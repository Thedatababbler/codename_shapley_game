#!/usr/bin/env python3
"""Create validation parquet files that explicitly require step-formatted answers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GSM8K_STEP_INSTRUCTION = (
    "\n\nFor this evaluation, you must use explicit reasoning steps. "
    "Write at least two lines starting exactly with 'Step 1:', 'Step 2:', etc. "
    "Then end with exactly one final line in the form '#### <answer>'. "
    "Responses without explicit Step N lines will not be counted."
)

MATH_STEP_INSTRUCTION = (
    "\n\nFor this evaluation, you must use explicit reasoning steps. "
    "Write at least two lines starting exactly with 'Step 1:', 'Step 2:', etc. "
    "Then put the final answer in \\boxed{} at the end. "
    "Responses without explicit Step N lines will not be counted."
)


def _append_instruction(prompt_value, instruction: str):
    prompt = prompt_value.copy()
    if len(prompt) == 0:
        return prompt_value
    last = dict(prompt[-1])
    last["content"] = str(last.get("content", "")).rstrip() + instruction
    prompt[-1] = last
    return prompt


def convert_file(src: Path, dst: Path, instruction: str) -> None:
    df = pd.read_parquet(src)
    df = df.copy()
    df["prompt"] = df["prompt"].apply(lambda prompt: _append_instruction(prompt, instruction))
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"Wrote {len(df)} rows to {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/step_prompt_eval"))
    args = parser.parse_args()

    convert_file(
        args.data_dir / "gsm8k" / "test.parquet",
        args.output_dir / "gsm8k_test.parquet",
        GSM8K_STEP_INSTRUCTION,
    )
    convert_file(
        args.data_dir / "math" / "test.parquet",
        args.output_dir / "math_test.parquet",
        MATH_STEP_INSTRUCTION,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
