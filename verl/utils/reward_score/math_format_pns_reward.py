"""Math/GSM8K reward wrapper with lightweight format and length penalties.

The base reward still comes from verl's default math/GSM8K correctness
functions. This wrapper only subtracts small penalties from otherwise correct
answers when the response exhibits degenerate formatting learned during RL:
duplicate answer markers, repeated final-answer boilerplate, and excessive
length.

It returns a dict so the reward manager logs both the final score and penalty
diagnostics into ``reward_extra_info`` / W&B.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from verl.utils.reward_score import default_compute_score


_FINAL_ANSWER_RE = re.compile(r"\b(final\s+answer|therefore,\s*the\s+final\s+answer)\b", re.IGNORECASE)
_ANSWER_MARKER_RE = re.compile(r"####|\b(final\s+answer|therefore,\s*the\s+final\s+answer)\b", re.IGNORECASE)
_BAD_HASH_RE = re.compile(r"####\s*####")


def _as_float_score(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("score", 0.0))
    if isinstance(value, (int, float, bool)):
        return float(value)
    return float(value[0])


def _linear_penalty(
    value: int,
    *,
    soft_limit: int,
    hard_limit: int,
    max_penalty: float,
) -> float:
    if max_penalty <= 0 or value <= soft_limit:
        return 0.0
    if hard_limit <= soft_limit:
        return max_penalty
    ratio = min(1.0, (value - soft_limit) / (hard_limit - soft_limit))
    return max_penalty * ratio


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    *,
    duplicate_hash_penalty: float = 0.1,
    duplicate_hash_max_penalty: float = 0.3,
    final_answer_penalty: float = 0.05,
    final_answer_max_penalty: float = 0.2,
    bad_hash_penalty: float = 0.1,
    post_answer_soft_chars: int = 240,
    post_answer_penalty: float = 0.1,
    length_soft_chars: int = 2400,
    length_hard_chars: int = 4200,
    length_max_penalty: float = 0.2,
    **kwargs,
) -> dict[str, float]:
    """Compute correctness reward minus small format/length penalties.

    Penalties are only applied to positive base rewards. Incorrect answers stay
    at 0 so we do not distort the binary correctness signal for failures.
    """
    raw_score = _as_float_score(
        default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    )

    num_hash_markers = solution_str.count("####")
    duplicate_hash_count = max(0, num_hash_markers - 1)
    duplicate_hash_pen = min(
        duplicate_hash_max_penalty,
        duplicate_hash_count * duplicate_hash_penalty,
    )

    num_final_answer_mentions = len(_FINAL_ANSWER_RE.findall(solution_str))
    duplicate_final_answer_count = max(0, num_final_answer_mentions - 1)
    final_answer_pen = min(
        final_answer_max_penalty,
        duplicate_final_answer_count * final_answer_penalty,
    )

    bad_hash_pen = bad_hash_penalty if _BAD_HASH_RE.search(solution_str) else 0.0
    answer_markers = list(_ANSWER_MARKER_RE.finditer(solution_str))
    post_answer_chars = 0
    if answer_markers:
        trailing_text = solution_str[answer_markers[-1].end() :].strip()
        post_answer_chars = len(trailing_text)
    post_answer_pen = post_answer_penalty if post_answer_chars > post_answer_soft_chars else 0.0
    length_pen = _linear_penalty(
        len(solution_str),
        soft_limit=length_soft_chars,
        hard_limit=length_hard_chars,
        max_penalty=length_max_penalty,
    )

    total_penalty = duplicate_hash_pen + final_answer_pen + bad_hash_pen + post_answer_pen + length_pen
    # Keep incorrect answers at 0. Positive partial rewards, if any are used by
    # the underlying scorer in the future, can also be softly penalized.
    penalized_score = raw_score if raw_score <= 0 else max(0.0, raw_score - total_penalty)

    return {
        "score": float(min(1.0, penalized_score)),
        "raw_score": float(raw_score),
        "format_penalty": float(duplicate_hash_pen + final_answer_pen + bad_hash_pen + post_answer_pen),
        "length_penalty": float(length_pen),
        "total_penalty": float(total_penalty if raw_score > 0 else 0.0),
        "num_hash_markers": float(num_hash_markers),
        "num_final_answer_mentions": float(num_final_answer_mentions),
        "post_answer_chars": float(post_answer_chars),
        "response_chars": float(len(solution_str)),
    }
