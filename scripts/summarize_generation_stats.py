#!/usr/bin/env python3
"""Summarize rollout/validation JSONL dumps by training step.

The trainer writes files named ``<global_step>.jsonl`` under rollout_data_dir
and validation_data_dir. This script computes score, response length, token
length, and reasoning-step count trends for quick baseline comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verl.utils.pns_step_segmenter import segment_steps  # noqa: E402


def _maybe_load_tokenizer(model_name_or_path: str | None) -> Any | None:
    if not model_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - best effort CLI helper
        print(f"warning: failed to load tokenizer {model_name_or_path!r}: {exc}", file=sys.stderr)
        return None


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_records(paths: list[Path]):
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield file, json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) or dump directory/directories.")
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer path/name for exact token counts.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    tokenizer = _maybe_load_tokenizer(args.tokenizer)
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for file, record in _iter_records(args.paths):
        step_value = record.get("step")
        if step_value is None:
            try:
                step_value = int(file.stem)
            except ValueError:
                step_value = -1
        step = int(step_value)

        output = str(record.get("output", ""))
        score = _safe_float(record.get("score"))
        token_count = (
            len(tokenizer.encode(output, add_special_tokens=False))
            if tokenizer is not None
            else len(output.split())
        )
        step_count = len(segment_steps(output, strategy="auto"))

        bucket = by_step[step]
        bucket["n"].append(1.0)
        bucket["output_chars"].append(float(len(output)))
        bucket["output_tokens"].append(float(token_count))
        bucket["reasoning_steps"].append(float(step_count))
        if score is not None:
            bucket["score"].append(score)
        for key in ("raw_score", "format_penalty", "length_penalty", "total_penalty"):
            value = _safe_float(record.get(key))
            if value is not None:
                bucket[key].append(value)

    fieldnames = [
        "step",
        "n",
        "score_mean",
        "output_tokens_mean",
        "output_chars_mean",
        "reasoning_steps_mean",
        "raw_score_mean",
        "format_penalty_mean",
        "length_penalty_mean",
        "total_penalty_mean",
    ]
    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        bucket = by_step[step]
        rows.append(
            {
                "step": step,
                "n": int(sum(bucket["n"])),
                "score_mean": _mean(bucket.get("score", [])),
                "output_tokens_mean": _mean(bucket["output_tokens"]),
                "output_chars_mean": _mean(bucket["output_chars"]),
                "reasoning_steps_mean": _mean(bucket["reasoning_steps"]),
                "raw_score_mean": _mean(bucket.get("raw_score", [])),
                "format_penalty_mean": _mean(bucket.get("format_penalty", [])),
                "length_penalty_mean": _mean(bucket.get("length_penalty", [])),
                "total_penalty_mean": _mean(bucket.get("total_penalty", [])),
            }
        )

    output_file = args.output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out = output_file.open("w", encoding="utf-8", newline="")
    else:
        out = sys.stdout

    with out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
