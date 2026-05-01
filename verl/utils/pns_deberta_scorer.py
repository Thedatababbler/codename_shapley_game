"""DeBERTa-based PNS step scorer for verl integration.

Loads a pre-trained 3-class DeBERTa PN scorer (low/mid/high) and provides
the ``score_steps`` interface expected by ``pns_reward_redistributor.py``.

3-class mapping:
    class 0 (PN_low)  -> score 0.0  (step is unnecessary)
    class 1 (PN_mid)  -> score 1.0  (step is partially necessary)
    class 2 (PN_high) -> score 2.0  (step is critical)

Usage:
    Set ``PNS_DEBERTA_CKPT`` env var to the checkpoint directory, then
    configure verl YAML::

        algorithm.pns_redistribution.pns_scorer_path: /path/to/pns_deberta_scorer.py
        algorithm.pns_redistribution.pns_scorer_name: score_steps
        algorithm.pns_redistribution.mode: classification
        algorithm.pns_redistribution.pns_values: [0.0, 1.0, 2.0]
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

NUM_CLASSES = 3
PN_BINS = [0.0, 1.0, 2.0]


def _resolve_dtype(name: Optional[str], device: torch.device) -> torch.dtype:
    """Resolve a string dtype spec (``bf16``/``fp16``/``fp32``) to a torch dtype.

    Defaults to bf16 on CUDA capability >= 8.0 (Ampere+), else fp16 on CUDA,
    else fp32 on CPU.
    """
    if name:
        key = name.strip().lower()
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp16", "half", "float16"):
            return torch.float16
        if key in ("fp32", "float32"):
            return torch.float32
    if device.type != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


class DeBERTaPNScorer:
    """Wraps a 3-class DeBERTa checkpoint for step-level PNS scoring."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 64,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        pad_to_multiple_of: int = 8,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.batch_size = batch_size
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))
        self.dtype = _resolve_dtype(dtype, self.device)

        logger.info(
            "Loading DeBERTa PNS scorer from %s on %s (dtype=%s, attn=%s, batch=%d, max_len=%d)",
            checkpoint_dir, self.device, self.dtype, attn_implementation or "default",
            self.batch_size, self.max_length,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        meta_path = Path(checkpoint_dir) / "train_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.num_classes = meta.get("num_classes", NUM_CLASSES)
            self.pn_bins = meta.get("pn_bins", PN_BINS)
            logger.info("Loaded meta: %d classes, bins=%s", self.num_classes, self.pn_bins)
        else:
            self.num_classes = NUM_CLASSES
            self.pn_bins = PN_BINS

        attn_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}

        ckpt_bin = Path(checkpoint_dir) / "pytorch_model.bin"
        if ckpt_bin.exists():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(checkpoint_dir)
            try:
                self.model = AutoModelForSequenceClassification.from_config(config, **attn_kwargs)
            except (TypeError, ValueError) as e:
                if attn_kwargs:
                    logger.warning(
                        "from_config with attn=%s failed (%s); falling back to default attention.",
                        attn_implementation, e,
                    )
                self.model = AutoModelForSequenceClassification.from_config(config)
            state_dict = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_dir, num_labels=self.num_classes, **attn_kwargs
                )
            except (TypeError, ValueError) as e:
                if attn_kwargs:
                    logger.warning(
                        "from_pretrained with attn=%s failed (%s); falling back to default attention.",
                        attn_implementation, e,
                    )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint_dir, num_labels=self.num_classes
                )
        self.model.to(self.device, dtype=self.dtype).eval()

        self.score_map = PN_BINS[:self.num_classes]
        logger.info("Score map: class -> %s", self.score_map)

    def _format_step(
        self,
        current_step: str,
        prefix_steps: list[str],
        question: Optional[str] = None,
    ) -> str:
        parts = []
        if question:
            parts.append(f"Question: {question}")

        if prefix_steps:
            reasoning = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(prefix_steps))
            parts.append(f"Reasoning:\n{reasoning}")

        step_idx = len(prefix_steps) + 1
        parts.append(f"Current Step {step_idx}: {current_step}")
        return "\n".join(parts)

    @torch.inference_mode()
    def _score_formatted(self, formatted: list[str]) -> torch.Tensor:
        """Score formatted texts using length-bucketed batching.

        We tokenize the whole list **without** padding once, then sort indices by
        token length and chunk them into batches. Each batch only pads to its
        own longest sequence (rounded up to ``pad_to_multiple_of`` for tensor
        cores). This dramatically reduces padding waste compared to the naive
        contiguous-slice batching when input lengths vary widely (which is the
        norm for step prefixes that grow with step index).
        """
        if not formatted:
            return torch.tensor([], dtype=torch.float32)

        enc = self.tokenizer(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        input_ids_list: list[list[int]] = enc["input_ids"]
        attention_mask_list: list[list[int]] = enc["attention_mask"]
        n = len(input_ids_list)
        lengths = [len(x) for x in input_ids_list]

        order = sorted(range(n), key=lambda i: lengths[i])

        scores_arr = [0.0] * n
        pad_mult = self.pad_to_multiple_of

        for start in range(0, n, self.batch_size):
            chunk = order[start : start + self.batch_size]
            chunk_len = max(lengths[i] for i in chunk)
            if pad_mult > 1:
                chunk_len = ((chunk_len + pad_mult - 1) // pad_mult) * pad_mult
            chunk_len = min(chunk_len, self.max_length)

            ids = torch.full(
                (len(chunk), chunk_len), self.pad_token_id, dtype=torch.long
            )
            mask = torch.zeros((len(chunk), chunk_len), dtype=torch.long)
            for j, i in enumerate(chunk):
                ll = lengths[i]
                ids[j, :ll] = torch.as_tensor(input_ids_list[i], dtype=torch.long)
                mask[j, :ll] = torch.as_tensor(attention_mask_list[i], dtype=torch.long)

            ids = ids.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            logits = self.model(input_ids=ids, attention_mask=mask).logits
            pred = logits.argmax(dim=-1).tolist()
            for j, i in enumerate(chunk):
                scores_arr[i] = self.score_map[pred[j]]

        return torch.tensor(scores_arr, dtype=torch.float32)

    def _format_steps_for_sample(
        self,
        step_texts: list[str],
        prompt_text: Optional[str] = None,
    ) -> list[str]:
        return [
            self._format_step(
                current_step=step.strip(),
                prefix_steps=[s.strip() for s in step_texts[:i]],
                question=prompt_text,
            )
            for i, step in enumerate(step_texts)
        ]

    @torch.no_grad()
    def __call__(
        self,
        step_texts: list[str],
        prompt_text: Optional[str] = None,
    ) -> torch.Tensor:
        """Score reasoning steps.

        Returns:
            Tensor of shape ``[T]`` with scores in {0.0, 1.0, 2.0}.
        """
        return self._score_formatted(self._format_steps_for_sample(step_texts, prompt_text))

    @torch.no_grad()
    def score_batches(
        self,
        step_text_batches: list[list[str]],
        prompt_texts: Optional[list[Optional[str]]] = None,
    ) -> list[torch.Tensor]:
        """Score multiple responses in one DeBERTa pass and split results back.

        Each sample keeps the same prefix-step formatting as ``__call__``; this
        method only batches all formatted examples together for throughput.
        """
        if prompt_texts is None:
            prompt_texts = [None] * len(step_text_batches)
        if len(prompt_texts) != len(step_text_batches):
            raise ValueError("prompt_texts must have the same length as step_text_batches")

        lengths: list[int] = []
        formatted_all: list[str] = []
        for step_texts, prompt_text in zip(step_text_batches, prompt_texts):
            formatted = self._format_steps_for_sample(step_texts, prompt_text)
            lengths.append(len(formatted))
            formatted_all.extend(formatted)

        flat_scores = self._score_formatted(formatted_all)
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            outputs.append(flat_scores[offset : offset + length])
            offset += length
        return outputs


_SINGLETON: Optional[DeBERTaPNScorer] = None


def _get_singleton() -> DeBERTaPNScorer:
    global _SINGLETON
    if _SINGLETON is None:
        ckpt = os.environ.get("PNS_DEBERTA_CKPT")
        if not ckpt:
            raise RuntimeError(
                "PNS_DEBERTA_CKPT env var must point to the DeBERTa checkpoint dir"
            )
        _SINGLETON = DeBERTaPNScorer(
            ckpt,
            device=os.environ.get("PNS_DEBERTA_DEVICE"),
            max_length=int(os.environ.get("PNS_DEBERTA_MAX_LENGTH", "512")),
            batch_size=int(os.environ.get("PNS_DEBERTA_BATCH_SIZE", "128")),
            dtype=os.environ.get("PNS_DEBERTA_DTYPE"),
            attn_implementation=os.environ.get("PNS_DEBERTA_ATTN", "sdpa"),
            pad_to_multiple_of=int(os.environ.get("PNS_DEBERTA_PAD_MULTIPLE", "8")),
        )
    return _SINGLETON


def score_steps(step_texts: list[str], **kwargs) -> torch.Tensor:
    """Module-level scorer function loaded by ``pns_scorer_path``."""
    return _get_singleton()(step_texts, **kwargs)


def score_step_batches(
    step_text_batches: list[list[str]],
    prompt_texts: Optional[list[Optional[str]]] = None,
) -> list[torch.Tensor]:
    """Batched scorer entry point used by PNS reward redistribution."""
    return _get_singleton().score_batches(step_text_batches, prompt_texts=prompt_texts)


score_steps.score_batches = score_step_batches
