"""Extensible step segmentation for reasoning rollout text.

Splits a reasoning response into discrete steps (segments). Each segmenter
returns a list of ``StepSegment`` named tuples that record the step text
together with its **character-level** start/end offsets in the original text.

Usage:
    segmenter = get_segmenter("double_newline")
    segments = segmenter(response_text)
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple

__all__ = [
    "StepSegment",
    "register_segmenter",
    "get_segmenter",
    "segment_steps",
]


# Below this many segments we consider segmentation "too coarse" and try the
# next fallback strategy in the cascade segmenter.
_MIN_SEGMENTS_FOR_OK = 2


class StepSegment(NamedTuple):
    """A single reasoning step extracted from a response."""

    text: str
    char_start: int
    char_end: int


SegmenterFn = Callable[[str], list[StepSegment]]
_SEGMENTER_REGISTRY: dict[str, SegmenterFn] = {}


def register_segmenter(name: str):
    """Decorator to register a step segmenter function."""

    def decorator(fn: SegmenterFn) -> SegmenterFn:
        _SEGMENTER_REGISTRY[name] = fn
        return fn

    return decorator


def get_segmenter(name: str) -> SegmenterFn:
    """Retrieve a registered segmenter by name."""
    if name not in _SEGMENTER_REGISTRY:
        raise ValueError(
            f"Unknown segmenter: {name!r}. Available: {list(_SEGMENTER_REGISTRY.keys())}"
        )
    return _SEGMENTER_REGISTRY[name]


def segment_steps(text: str, strategy: str = "double_newline") -> list[StepSegment]:
    """Convenience wrapper: segment text using the named strategy."""
    return get_segmenter(strategy)(text)


# ---------------------------------------------------------------------------
# Built-in segmenters
# ---------------------------------------------------------------------------


@register_segmenter("double_newline")
def _segment_double_newline(text: str) -> list[StepSegment]:
    r"""Split on ``\n\n`` (two consecutive newlines).

    Consecutive delimiters are collapsed. Empty segments after stripping
    whitespace are discarded. Delimiter text is not included in any segment.
    """
    segments: list[StepSegment] = []
    for m in re.finditer(r"(?s)(.+?)(?:\n\n+|$)", text):
        seg_text = m.group(1)
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=m.start(1), char_end=m.end(1)))
    if not segments and text.strip():
        segments.append(StepSegment(text=text, char_start=0, char_end=len(text)))
    return segments


@register_segmenter("step_marker")
def _segment_step_marker(text: str) -> list[StepSegment]:
    r"""Split on ``Step N:`` markers (case-insensitive).

    Each segment runs from one marker to the next (or end of text).
    Text before the first marker is included as a separate segment if non-empty.
    """
    marker_pattern = re.compile(r"(?i)(?=Step\s+\d+\s*:)")
    positions = [m.start() for m in marker_pattern.finditer(text)]

    if not positions:
        if text.strip():
            return [StepSegment(text=text, char_start=0, char_end=len(text))]
        return []

    segments: list[StepSegment] = []
    if positions[0] > 0:
        prefix = text[: positions[0]]
        if prefix.strip():
            segments.append(StepSegment(text=prefix, char_start=0, char_end=positions[0]))

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        seg_text = text[start:end]
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=start, char_end=end))

    return segments


@register_segmenter("think_tag")
def _segment_think_tag(text: str) -> list[StepSegment]:
    r"""Split on ``<think>...</think>`` boundaries.

    Separates the thinking block from the final answer block.
    If no think tag is found, the entire text is one segment.
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    segments: list[StepSegment] = []
    last_end = 0

    for m in pattern.finditer(text):
        if m.start() > last_end:
            pre = text[last_end : m.start()]
            if pre.strip():
                segments.append(StepSegment(text=pre, char_start=last_end, char_end=m.start()))
        inner = m.group(1)
        if inner.strip():
            segments.append(StepSegment(text=inner, char_start=m.start(1), char_end=m.end(1)))
        last_end = m.end()

    if last_end < len(text):
        tail = text[last_end:]
        if tail.strip():
            segments.append(StepSegment(text=tail, char_start=last_end, char_end=len(text)))

    if not segments and text.strip():
        segments.append(StepSegment(text=text, char_start=0, char_end=len(text)))

    return segments


@register_segmenter("sentence")
def _segment_sentence(text: str) -> list[StepSegment]:
    """Split on sentence boundaries (period/exclamation/question mark followed by space or end).

    Useful as a fine-grained fallback.
    """
    segments: list[StepSegment] = []
    for m in re.finditer(r"(?s)(.+?[.!?])(?:\s+|$)", text):
        seg_text = m.group(1)
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=m.start(1), char_end=m.end(1)))
    if not segments and text.strip():
        segments.append(StepSegment(text=text, char_start=0, char_end=len(text)))
    return segments


# ---------------------------------------------------------------------------
# Additional fallback segmenters
# ---------------------------------------------------------------------------


# Common numbered/bulleted step markers, anchored at start-of-line.
# Examples that match: "1. ", "1) ", "(1) ", "Step 1:", "step 1.", "第 1 步"
_NUMBERED_MARKER_RE = re.compile(
    r"(?im)^[ \t]*("
    r"step\s+\d+\s*[:.\)]"          # Step 1:  / Step 2.
    r"|第\s*\d+\s*步\s*[:：.]?"      # 第 1 步:
    r"|\(?\d+\)[ \t]"               # 1) text  /  (1) text
    r"|\d+\.[ \t]"                   # 1. text
    r"|\d+、"                        # 1、text  (Chinese enumeration)
    r")"
)


@register_segmenter("numbered_marker")
def _segment_numbered_marker(text: str) -> list[StepSegment]:
    r"""Split on numbered/enumerated step markers at start of a line.

    Recognises ``1.``, ``1)``, ``(1)``, ``Step 1:``, ``第 1 步:``, ``1、`` etc.
    Each segment runs from one marker to the next (or end of text).
    Text before the first marker is included as a separate segment if non-empty.
    """
    positions = [m.start() for m in _NUMBERED_MARKER_RE.finditer(text)]

    if not positions:
        if text.strip():
            return [StepSegment(text=text, char_start=0, char_end=len(text))]
        return []

    segments: list[StepSegment] = []
    if positions[0] > 0:
        prefix = text[: positions[0]]
        if prefix.strip():
            segments.append(StepSegment(text=prefix, char_start=0, char_end=positions[0]))

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        seg_text = text[start:end]
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=start, char_end=end))

    return segments


# Conservative inline numbered step markers.
# Examples that match: "1. **Find the height**", "... area.2. Compute:"
# Examples intentionally not matched: decimals like "1.5", citations, or
# generic numbered text without a title cue.
_INLINE_NUMBERED_MARKER_RE = re.compile(
    r"(^|(?<=[.!?])\s*)"
    r"(?=\d{1,2}\.\s+(?:\*\*[^*\n]{1,80}\*\*|[A-Z][^:\n]{1,80}:))"
)


@register_segmenter("inline_numbered_marker")
def _segment_inline_numbered_marker(text: str) -> list[StepSegment]:
    r"""Split on inline numbered reasoning markers without requiring newlines.

    This catches rollout patterns like ``"... facts.1. **Determine ... 2. **Use ..."``
    where the model emits clear numbered steps but omits line breaks. Matching
    requires a title-like cue after ``N.`` to avoid splitting decimals or
    ordinary numeric mentions.
    """
    positions = [m.end(1) for m in _INLINE_NUMBERED_MARKER_RE.finditer(text)]

    if not positions:
        if text.strip():
            return [StepSegment(text=text, char_start=0, char_end=len(text))]
        return []

    segments: list[StepSegment] = []
    if positions[0] > 0:
        prefix = text[: positions[0]]
        if prefix.strip():
            segments.append(StepSegment(text=prefix, char_start=0, char_end=positions[0]))

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        seg_text = text[start:end]
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=start, char_end=end))

    return segments


@register_segmenter("single_newline")
def _segment_single_newline(text: str) -> list[StepSegment]:
    r"""Split on any single ``\n`` (treats every non-empty line as a step).

    Empty lines are skipped. Useful when the model emits one reasoning step
    per line without double newlines.
    """
    segments: list[StepSegment] = []
    for m in re.finditer(r"(?s)(.+?)(?:\n+|$)", text):
        seg_text = m.group(1)
        if seg_text.strip():
            segments.append(StepSegment(text=seg_text, char_start=m.start(1), char_end=m.end(1)))
    if not segments and text.strip():
        segments.append(StepSegment(text=text, char_start=0, char_end=len(text)))
    return segments


# Cascade order for the "auto" segmenter. We try coarse-grained markers first
# (preserve author intent if present) and then progressively finer fallbacks.
_AUTO_CASCADE: tuple[str, ...] = (
    "double_newline",
    "numbered_marker",
    "step_marker",
    "inline_numbered_marker",
    "single_newline",
    "sentence",
)


@register_segmenter("auto")
def _segment_auto(text: str) -> list[StepSegment]:
    """Cascade segmenter with fallbacks.

    Tries strategies in order: ``double_newline`` → ``numbered_marker`` →
    ``step_marker`` → ``inline_numbered_marker`` → ``single_newline`` →
    ``sentence``. Stops at the first strategy that produces at least
    ``_MIN_SEGMENTS_FOR_OK`` non-trivial segments. If none meet the threshold,
    returns the result of the last strategy attempted (which always covers the
    full text).
    """
    last_result: list[StepSegment] = []
    for name in _AUTO_CASCADE:
        result = _SEGMENTER_REGISTRY[name](text)
        last_result = result
        if len(result) >= _MIN_SEGMENTS_FOR_OK:
            return result
    return last_result
