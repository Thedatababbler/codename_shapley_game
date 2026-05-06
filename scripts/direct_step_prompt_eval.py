#!/usr/bin/env python3
"""Run single-GPU step-prompt evaluation with a merged Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    rows = _load_rows(args.val_files)
    prompts = [_messages_to_text(tokenizer, row["prompt"]) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="generating"):
            batch_rows = rows[start : start + args.batch_size]
            batch_prompts = prompts[start : start + args.batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_ids = generated[:, prompt_len:]
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for row, prompt, output in zip(batch_rows, batch_prompts, outputs):
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
