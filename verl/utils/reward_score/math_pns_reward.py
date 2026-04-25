"""Math reward function with integrated PNS step scoring.

Returns ``{"score": float, "pns_scores": list[float]}`` so the PNS
redistribution hook reads scores from ``non_tensor_batch`` directly —
no external scorer needed.

Configure via ``reward.custom_reward_function``:

.. code-block:: yaml

    reward:
      custom_reward_function:
        path: verl/utils/reward_score/math_pns_reward.py
        name: compute_score
"""

from __future__ import annotations

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PNS_SCORER = None


def _get_pns_scorer():
    """Lazy-load the DeBERTa PNS scorer."""
    global _PNS_SCORER
    if _PNS_SCORER is not None:
        return _PNS_SCORER

    ckpt = os.environ.get("PNS_DEBERTA_CKPT")
    if not ckpt:
        return None

    from verl.utils.pns_deberta_scorer import DeBERTaPNScorer
    _PNS_SCORER = DeBERTaPNScorer(ckpt)
    return _PNS_SCORER


def _extract_math_answer(solution_str: str) -> Optional[str]:
    """Extract the answer from \\boxed{...}."""
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution_str)
    return matches[-1].strip() if matches else None


def _check_correctness(solution_str: str, ground_truth: str) -> float:
    """Check answer correctness, trying math_verify first, then regex."""
    try:
        from verl.utils.reward_score.math_verify import compute_score as verify_score
        return verify_score(solution_str, ground_truth)
    except Exception:
        pass

    try:
        from verl.utils.reward_score.math_reward import compute_score as regex_score
        return float(regex_score(solution_str, ground_truth))
    except Exception:
        pass

    extracted = _extract_math_answer(solution_str)
    if extracted is not None:
        return 1.0 if extracted.strip() == str(ground_truth).strip() else 0.0
    return 0.0


def _segment_response(solution_str: str) -> list[str]:
    """Split the solution into reasoning steps (double-newline or Step N:)."""
    step_marker = re.compile(r"(?i)Step\s+\d+\s*:")
    if step_marker.search(solution_str):
        parts = step_marker.split(solution_str)
        steps = [p.strip() for p in parts if p.strip()]
    else:
        steps = [s.strip() for s in solution_str.split("\n\n") if s.strip()]

    return steps if steps else [solution_str.strip()]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute math correctness and per-step PNS scores.

    Returns:
        dict with ``score`` (float) and ``pns_scores`` (list[float]).
    """
    score = _check_correctness(solution_str, ground_truth)

    pns_scores = None
    scorer = _get_pns_scorer()
    if scorer is not None:
        steps = _segment_response(solution_str)
        if steps:
            question = None
            if extra_info:
                question = extra_info.get("question") or extra_info.get("prompt_text")

            pn_tensor = scorer(steps, prompt_text=question)
            pns_scores = pn_tensor.tolist()

    result = {"score": float(score)}
    if pns_scores is not None:
        result["pns_scores"] = pns_scores
    return result
