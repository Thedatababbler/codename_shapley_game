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


class DeBERTaPNScorer:
    """Wraps a 3-class DeBERTa checkpoint for step-level PNS scoring."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 64,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info("Loading DeBERTa PNS scorer from %s on %s", checkpoint_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

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

        ckpt_bin = Path(checkpoint_dir) / "pytorch_model.bin"
        if ckpt_bin.exists():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(checkpoint_dir)
            self.model = AutoModelForSequenceClassification.from_config(config)
            state_dict = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir, num_labels=self.num_classes
            )
        self.model.to(self.device).eval()

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
        if not step_texts:
            return torch.tensor([], dtype=torch.float32)

        formatted = []
        for i, step in enumerate(step_texts):
            formatted.append(self._format_step(
                current_step=step.strip(),
                prefix_steps=[s.strip() for s in step_texts[:i]],
                question=prompt_text,
            ))

        all_scores: list[float] = []
        for start in range(0, len(formatted), self.batch_size):
            batch_texts = formatted[start : start + self.batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_cls = logits.argmax(dim=-1).cpu().tolist()
            all_scores.extend(self.score_map[c] for c in pred_cls)

        return torch.tensor(all_scores, dtype=torch.float32)


_SINGLETON: Optional[DeBERTaPNScorer] = None


def score_steps(step_texts: list[str], **kwargs) -> torch.Tensor:
    """Module-level scorer function loaded by ``pns_scorer_path``."""
    global _SINGLETON
    if _SINGLETON is None:
        ckpt = os.environ.get("PNS_DEBERTA_CKPT")
        if not ckpt:
            raise RuntimeError(
                "PNS_DEBERTA_CKPT env var must point to the DeBERTa checkpoint dir"
            )
        _SINGLETON = DeBERTaPNScorer(ckpt)
    return _SINGLETON(step_texts, **kwargs)
