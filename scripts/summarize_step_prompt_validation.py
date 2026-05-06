#!/usr/bin/env python3
"""Summarize step-prompt validation dumps, filtering out non-step responses."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any

_STEP_MARKER_RE = re.compile(r"(?im)^[ \t]*Step\s+\d+\s*:")
_GSM8K_VAL_SIZE = 1319


def _iter_jsonl(path: Path):
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    yield file, idx, json.loads(line)


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dataset_for_index(idx: int) -> str:
    return "gsm8k" if idx < _GSM8K_VAL_SIZE else "math"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for path in args.paths:
        run_name = path.name
        for file, idx, record in _iter_jsonl(path):
            step_value = record.get("step", file.stem)
            dataset = _dataset_for_index(idx)
            output = str(record.get("output", ""))
            score = _as_float(record.get("score"))
            marker_count = len(_STEP_MARKER_RE.findall(output))
            compliant = marker_count >= 2

            key = (run_name, dataset)
            bucket = buckets.setdefault(key, {})
            bucket.setdefault("n_total", []).append(1.0)
            bucket.setdefault("step_markers", []).append(float(marker_count))
            bucket.setdefault("output_chars_all", []).append(float(len(output)))
            if score is not None:
                bucket.setdefault("score_all", []).append(score)
            if compliant:
                bucket.setdefault("n_step_compliant", []).append(1.0)
                bucket.setdefault("output_chars_filtered", []).append(float(len(output)))
                bucket.setdefault("step_markers_filtered", []).append(float(marker_count))
                if score is not None:
                    bucket.setdefault("score_filtered", []).append(score)

    rows = []
    for (run_name, dataset), bucket in sorted(buckets.items()):
        n_total = int(sum(bucket.get("n_total", [])))
        n_step = int(sum(bucket.get("n_step_compliant", [])))
        rows.append(
            {
                "run": run_name,
                "dataset": dataset,
                "n_total": n_total,
                "n_step_compliant": n_step,
                "step_compliance_rate": n_step / n_total if n_total else 0.0,
                "score_all": _mean(bucket.get("score_all", [])),
                "score_filtered": _mean(bucket.get("score_filtered", [])),
                "output_chars_all": _mean(bucket.get("output_chars_all", [])),
                "output_chars_filtered": _mean(bucket.get("output_chars_filtered", [])),
                "step_markers_all": _mean(bucket.get("step_markers", [])),
                "step_markers_filtered": _mean(bucket.get("step_markers_filtered", [])),
            }
        )

    if rows:
        totals: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            run = row["run"]
            total = totals.setdefault(run, {})
            for key in ("n_total", "n_step_compliant"):
                total.setdefault(key, []).append(float(row[key]))
            for key in ("score_all", "score_filtered", "output_chars_all", "output_chars_filtered"):
                weight = float(row["n_total"] if key.endswith("_all") else row["n_step_compliant"])
                total.setdefault(key, []).append(float(row[key]) * weight)
        for run, total in sorted(totals.items()):
            n_total = int(sum(total.get("n_total", [])))
            n_step = int(sum(total.get("n_step_compliant", [])))
            rows.append(
                {
                    "run": run,
                    "dataset": "overall",
                    "n_total": n_total,
                    "n_step_compliant": n_step,
                    "step_compliance_rate": n_step / n_total if n_total else 0.0,
                    "score_all": sum(total.get("score_all", [])) / n_total if n_total else 0.0,
                    "score_filtered": sum(total.get("score_filtered", [])) / n_step if n_step else 0.0,
                    "output_chars_all": sum(total.get("output_chars_all", [])) / n_total if n_total else 0.0,
                    "output_chars_filtered": (
                        sum(total.get("output_chars_filtered", [])) / n_step if n_step else 0.0
                    ),
                    "step_markers_all": "",
                    "step_markers_filtered": "",
                }
            )

    fieldnames = [
        "run",
        "dataset",
        "n_total",
        "n_step_compliant",
        "step_compliance_rate",
        "score_all",
        "score_filtered",
        "output_chars_all",
        "output_chars_filtered",
        "step_markers_all",
        "step_markers_filtered",
    ]
    out = args.output.open("w", encoding="utf-8", newline="") if args.output else None
    try:
        writer = csv.DictWriter(out or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
