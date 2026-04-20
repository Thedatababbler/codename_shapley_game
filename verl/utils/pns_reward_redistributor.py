"""End-to-end PNS reward redistribution: segment → score → allocate → token map.

This module provides the top-level function ``redistribute_token_rewards_with_pns``
that plugs into ``ray_trainer.py``'s advantage computation stage.  It reads
PNS scores from ``non_tensor_batch`` (produced by the reward function), splits
the response into reasoning steps, redistributes the trajectory-level reward
across steps, and rewrites ``token_level_scores`` in the batch.

If PNS scores are **not** pre-computed (i.e. not in ``non_tensor_batch``), the
module can call a user-supplied scoring function to produce them on the fly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import torch

from verl.protocol import DataProto
from verl.utils.pns_reward_allocation import (
    compute_pns_diagnostics,
    pns_baseline_normalized_surplus_allocation,
)
from verl.utils.pns_step_segmenter import StepSegment, segment_steps
from verl.utils.pns_token_mapping import (
    broadcast_step_rewards_to_tokens,
    map_steps_to_token_spans,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["redistribute_token_rewards_with_pns"]

PNSScorerFn = Callable[[list[str]], torch.Tensor]


def _extract_valid_response(
    batch_item_batch: dict,
) -> tuple[torch.Tensor, int, int]:
    """Extract valid response token ids and lengths from a single sample.

    Returns:
        (valid_response_ids, prompt_length, response_length)
    """
    responses = batch_item_batch["responses"]
    response_length = responses.shape[-1]
    attention_mask = batch_item_batch["attention_mask"]
    prompt_length = attention_mask.shape[-1] - response_length
    valid_len = attention_mask[prompt_length:].sum().int().item()
    valid_ids = responses[:valid_len]
    return valid_ids, prompt_length, response_length


def redistribute_token_rewards_with_pns(
    batch: DataProto,
    tokenizer: Any,
    pns_config: dict,
    pns_scorer: Optional[PNSScorerFn] = None,
) -> tuple[DataProto, dict[str, float]]:
    """Rewrite ``batch.batch["token_level_scores"]`` using PNS redistribution.

    This function is designed to be called inside ``ray_trainer.fit()``
    right after ``token_level_scores`` is assigned and before
    ``compute_advantage``.

    Args:
        batch: The training batch.  Must already have ``token_level_scores``
            (shape ``[B, response_length]``) and ``responses`` / ``attention_mask``.
        tokenizer: HuggingFace tokenizer used for the actor model.
        pns_config: Dict with keys:

            - ``alpha`` (float, default 0.5)
            - ``eps`` (float, default 1e-8)
            - ``mode`` (str, default "classification")
            - ``pns_values`` (list[float], default [0, 1/3, 2/3, 1])
            - ``variant`` (str, default "surplus")
            - ``step_segmenter`` (str, default "double_newline")
            - ``pns_score_key`` (str, default "pns_scores")
              Key in ``non_tensor_batch`` that holds pre-computed PNS scores.

        pns_scorer: Optional callable ``(step_texts: list[str]) -> Tensor[T]``
            that scores a list of step texts and returns per-step PNS scores.
            Used when PNS scores are **not** pre-computed in the batch.  This
            is where you plug in your pre-trained PNS scoring model.

    Returns:
        (batch, metrics): The batch with rewritten ``token_level_scores``,
        and a dict of PNS diagnostic metrics.
    """
    alpha = pns_config.get("alpha", 0.5)
    eps = pns_config.get("eps", 1e-8)
    mode = pns_config.get("mode", "regression")
    pns_values_list = pns_config.get("pns_values", [0.0, 1 / 3, 2 / 3, 1.0])
    variant = pns_config.get("variant", "surplus")
    segmenter_name = pns_config.get("step_segmenter", "double_newline")
    pns_score_key = pns_config.get("pns_score_key", "pns_scores")

    pns_values = torch.tensor(pns_values_list, dtype=torch.float32) if mode == "classification" else None

    B = batch.batch["token_level_scores"].shape[0]
    response_length = batch.batch["token_level_scores"].shape[1]

    all_scores_list: list[torch.Tensor] = []
    all_shares_list: list[torch.Tensor] = []
    all_step_rewards_list: list[torch.Tensor] = []
    final_rewards_list: list[float] = []

    new_token_level_scores = batch.batch["token_level_scores"].clone()

    for i in range(B):
        sample_batch = {k: v[i] for k, v in batch.batch.items()}
        valid_ids, prompt_length, resp_len = _extract_valid_response(sample_batch)

        R = batch.batch["token_level_scores"][i].sum()
        final_rewards_list.append(R.item())

        if valid_ids.numel() == 0:
            continue

        decoded_text = tokenizer.decode(valid_ids, skip_special_tokens=False)

        segments: list[StepSegment] = segment_steps(decoded_text, strategy=segmenter_name)
        if not segments:
            continue

        T_steps = len(segments)

        # --- Obtain PNS scores ---
        has_precomputed = (
            pns_score_key in batch.non_tensor_batch
            and batch.non_tensor_batch[pns_score_key][i] is not None
        )

        if has_precomputed:
            raw_scores = batch.non_tensor_batch[pns_score_key][i]
            if isinstance(raw_scores, np.ndarray):
                if np.isnan(raw_scores).any():
                    logger.warning(
                        "Sample %d: pns_scores contains NaN; "
                        "skipping PNS redistribution for this sample.",
                        i,
                    )
                    continue
                pns_output = torch.from_numpy(raw_scores).float()
            elif isinstance(raw_scores, torch.Tensor):
                if raw_scores.isnan().any():
                    logger.warning(
                        "Sample %d: pns_scores contains NaN; "
                        "skipping PNS redistribution for this sample.",
                        i,
                    )
                    continue
                pns_output = raw_scores.float()
            elif isinstance(raw_scores, (list, tuple)):
                if any(v is None for v in raw_scores):
                    logger.warning(
                        "Sample %d: pns_scores contains None values; "
                        "skipping PNS redistribution for this sample.",
                        i,
                    )
                    continue
                pns_output = torch.tensor(raw_scores, dtype=torch.float32)
            else:
                logger.warning(
                    "Sample %d: unexpected pns_scores type %s; skipping PNS redistribution.",
                    i, type(raw_scores).__name__,
                )
                continue
        elif pns_scorer is not None:
            step_texts = [seg.text for seg in segments]
            pns_output = pns_scorer(step_texts)
            if not isinstance(pns_output, torch.Tensor):
                pns_output = torch.tensor(pns_output, dtype=torch.float32)
        else:
            continue

        if pns_output.shape[0] != T_steps:
            if pns_output.shape[0] > T_steps:
                pns_output = pns_output[:T_steps]
            else:
                logger.warning(
                    "Sample %d: PNS scores length (%d) < segments (%d); skipping.",
                    i, pns_output.shape[0], T_steps,
                )
                continue

        result = pns_baseline_normalized_surplus_allocation(
            final_reward=R,
            pns_output=pns_output,
            mode=mode,
            pns_values=pns_values,
            alpha=alpha,
            eps=eps,
            variant=variant,
        )

        all_scores_list.append(result["scores"].detach())
        all_shares_list.append(result["shares"].detach())
        all_step_rewards_list.append(result["step_rewards"].detach())

        token_spans = map_steps_to_token_spans(
            response_ids=valid_ids,
            tokenizer=tokenizer,
            segments=segments,
            decoded_text=decoded_text,
        )

        token_rewards = broadcast_step_rewards_to_tokens(
            step_rewards=result["step_rewards"],
            token_spans=token_spans,
            response_length=resp_len,
            response_mask=sample_batch.get("response_mask"),
        )

        new_token_level_scores[i, :resp_len] = token_rewards

    batch.batch["token_level_scores"] = new_token_level_scores

    metrics: dict[str, float] = {}
    if all_scores_list:
        metrics = compute_pns_diagnostics(
            all_scores=all_scores_list,
            all_shares=all_shares_list,
            all_step_rewards=all_step_rewards_list,
            all_final_rewards=torch.tensor(final_rewards_list, dtype=torch.float32),
        )

    return batch, metrics
