"""Map step-level rewards to token-level reward tensors.

Two mapping strategies are provided:

1. **offset_mapping** (primary): Re-encode the decoded response text with
   ``return_offsets_mapping=True``, then use the character offsets reported
   by the tokenizer to determine which tokens belong to which step.

2. **proportional fallback**: When the re-encoded token ids do not match
   the original rollout token ids (edge case due to normalisation or special
   tokens), fall back to distributing steps proportionally by their character
   length.

Both strategies **distribute** the step reward across the tokens that make up
that step. To preserve reward conservation – i.e. ``sum_t r_t == sum_step
step_reward`` – every token in a step's span receives an *equal share* of the
step reward (``step_reward / span_length``), not the full ``step_reward``.
"""

from __future__ import annotations

import bisect
import logging
import warnings
from typing import Optional

import torch

from verl.utils.pns_step_segmenter import StepSegment

logger = logging.getLogger(__name__)

__all__ = [
    "map_steps_to_token_spans",
    "broadcast_step_rewards_to_tokens",
]


# ---------------------------------------------------------------------------
# Primary: offset_mapping-based mapping
# ---------------------------------------------------------------------------


def _build_char_to_token_map(
    response_ids: torch.Tensor,
    tokenizer,
    decoded_text: str,
) -> Optional[list[tuple[int, int]]]:
    """Re-encode ``decoded_text`` and return offset_mapping if token ids match.

    Returns ``None`` when the re-encoded ids diverge from the original
    ``response_ids`` (signals that the caller should fall back).
    """
    try:
        encoding = tokenizer(
            decoded_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
    except Exception:
        return None

    reencoded_ids = encoding["input_ids"].squeeze(0)
    offsets: list[tuple[int, int]] = encoding["offset_mapping"].squeeze(0).tolist()

    original = response_ids.cpu()
    if reencoded_ids.shape[0] != original.shape[0]:
        return None
    if not torch.equal(reencoded_ids, original):
        mismatch_pct = (reencoded_ids != original).float().mean().item()
        if mismatch_pct > 0.05:
            return None
        logger.debug(
            "Minor token mismatch (%.1f%%) between rollout ids and re-encoded ids; "
            "proceeding with offset_mapping anyway.",
            mismatch_pct * 100,
        )
    return offsets


def _char_offset_to_token_index(
    char_pos: int,
    offsets: list[tuple[int, int]],
    side: str = "left",
) -> int:
    """Binary search to find the token index covering ``char_pos``.

    ``side="left"``  → first token whose char range includes or follows char_pos.
    ``side="right"`` → last  token whose char range includes or precedes char_pos.
    """
    starts = [s for s, _ in offsets]
    ends = [e for _, e in offsets]

    if side == "left":
        idx = bisect.bisect_right(starts, char_pos) - 1
        return max(idx, 0)
    else:
        idx = bisect.bisect_left(ends, char_pos)
        return min(idx, len(offsets) - 1)


# ---------------------------------------------------------------------------
# Fallback: proportional mapping
# ---------------------------------------------------------------------------


def _proportional_token_spans(
    segments: list[StepSegment],
    total_text_len: int,
    num_tokens: int,
) -> list[tuple[int, int]]:
    """Distribute tokens across steps proportionally to their character length."""
    if total_text_len == 0 or num_tokens == 0:
        return [(0, num_tokens)] if segments else []

    spans: list[tuple[int, int]] = []
    used = 0
    for i, seg in enumerate(segments):
        seg_char_len = seg.char_end - seg.char_start
        if i == len(segments) - 1:
            token_count = num_tokens - used
        else:
            token_count = max(1, round(num_tokens * seg_char_len / total_text_len))
        start = used
        end = min(used + token_count, num_tokens)
        spans.append((start, end))
        used = end
    if used < num_tokens and spans:
        s, _ = spans[-1]
        spans[-1] = (s, num_tokens)
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_steps_to_token_spans(
    response_ids: torch.Tensor,
    tokenizer,
    segments: list[StepSegment],
    decoded_text: Optional[str] = None,
) -> list[tuple[int, int]]:
    """Map step segments (defined by character offsets) to token index spans.

    Args:
        response_ids: Valid response token ids from rollout, shape ``[L]``.
        tokenizer: HuggingFace tokenizer (should support ``return_offsets_mapping``).
        segments: Step segments with character offsets.
        decoded_text: Pre-decoded response text.  If ``None``, will be decoded
            from ``response_ids``.

    Returns:
        List of ``(token_start, token_end)`` spans (exclusive end), one per step.
        Token indices are relative to the beginning of ``response_ids``.
    """
    if not segments:
        return []

    num_tokens = response_ids.shape[0]
    if decoded_text is None:
        decoded_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    offsets = _build_char_to_token_map(response_ids, tokenizer, decoded_text)

    if offsets is not None:
        spans: list[tuple[int, int]] = []
        for seg in segments:
            tok_start = _char_offset_to_token_index(seg.char_start, offsets, side="left")
            tok_end = _char_offset_to_token_index(seg.char_end, offsets, side="right") + 1
            tok_end = min(tok_end, num_tokens)
            tok_start = min(tok_start, tok_end)
            spans.append((tok_start, tok_end))

        total_assigned = sum(e - s for s, e in spans)
        if total_assigned == 0:
            warnings.warn(
                "offset_mapping resulted in zero assigned tokens; falling back to proportional.",
                stacklevel=2,
            )
        else:
            return spans

    total_text_len = max(seg.char_end for seg in segments) - min(seg.char_start for seg in segments)
    return _proportional_token_spans(segments, total_text_len, num_tokens)


def broadcast_step_rewards_to_tokens(
    step_rewards: torch.Tensor,
    token_spans: list[tuple[int, int]],
    response_length: int,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Distribute step-level rewards across the tokens of each step span.

    For each step, the step reward is **evenly spread** over the tokens it
    covers: every token in a span of length ``L`` receives ``step_reward / L``.
    This preserves reward conservation: ``token_rewards.sum() ==
    step_rewards.sum()`` (modulo masking and tokens outside any span, which
    contribute zero).

    Args:
        step_rewards: Shape ``[T]``, one reward per step.
        token_spans: ``T`` tuples of ``(start, end)`` token indices.
        response_length: Total length of the padded response tensor.
        response_mask: Optional mask, shape ``[response_length]``.
            If provided, rewards are zeroed out on masked positions. Note
            that masking can break exact conservation if a step span overlaps
            masked-out tokens; in practice valid response tokens fully cover
            the relevant spans.

    Returns:
        Token-level reward tensor, shape ``[response_length]``.
    """
    token_rewards = torch.zeros(response_length, device=step_rewards.device, dtype=step_rewards.dtype)

    for step_idx, (start, end) in enumerate(token_spans):
        if start >= end or start >= response_length:
            continue
        actual_end = min(end, response_length)
        span_len = actual_end - start
        if span_len <= 0:
            continue
        token_rewards[start:actual_end] = step_rewards[step_idx] / span_len

    if response_mask is not None:
        token_rewards = token_rewards * response_mask.float()

    return token_rewards
