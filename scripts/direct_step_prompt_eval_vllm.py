#!/usr/bin/env python3
"""Run single-GPU step-prompt evaluation with vLLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.math_format_pns_reward import compute_score


_STEP_MARKER_RE = re.compile(r"(?im)^[ \t]*Step\s+\d+\s*:")


def _messages_to_text(tokenizer, messages: Any) -> str:
    return tokenizer.apply_chat_template(messages.tolist(), tokenize=False, add_generation_prompt=True)


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            reward_model = row["reward_model"]
            extra_info = row.get("extra_info", None)
            rows.append(
                {
                    "dataset_index": int(idx),
                    "data_source": row["data_source"],
                    "prompt": row["prompt"],
                    "ground_truth": reward_model["ground_truth"],
                    "extra_info": extra_info if isinstance(extra_info, dict) else None,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--val-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = _load_rows(args.val_files)
    prompts = [_messages_to_text(tokenizer, row["prompt"]) for row in rows]

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(rows), args.chunk_size), desc="vllm_chunks"):
            chunk_rows = rows[start : start + args.chunk_size]
            chunk_prompts = prompts[start : start + args.chunk_size]
            request_outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=True)
            for row, prompt, request_output in zip(chunk_rows, chunk_prompts, request_outputs):
                output = request_output.outputs[0].text if request_output.outputs else ""
                score_info = compute_score(
                    data_source=row["data_source"],
                    solution_str=output,
                    ground_truth=row["ground_truth"],
                    extra_info=row["extra_info"],
                )
                step_markers = len(_STEP_MARKER_RE.findall(output))
                record = {
                    "data_source": row["data_source"],
                    "dataset_index": row["dataset_index"],
                    "input": prompt,
                    "output": output,
                    "gts": row["ground_truth"],
                    "step_markers": step_markers,
                    "step_compliant": step_markers >= 2,
                    **score_info,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
