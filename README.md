# PNS-Based Step-Level Reward Redistribution for Reasoning RL

This repository implements **PNS (Process Necessity Score) based reward redistribution** on top of [verl](https://github.com/volcengine/verl), a reinforcement learning training framework for LLMs.

## Overview

In reasoning RL (e.g. GRPO, PPO for chain-of-thought), the reward is typically a **single scalar** assigned at the trajectory level. This makes credit assignment across reasoning steps difficult — all steps receive the same advantage signal regardless of their individual contribution.

**PNS reward redistribution** solves this by:

1. Splitting the model's reasoning response into discrete **steps**
2. Scoring each step with a pre-trained **PNS model** (Process Necessity Score)
3. Redistributing the trajectory-level reward across steps using a **budget-preserving normalized surplus rule**
4. Broadcasting step-level rewards to **token-level** reward tensors for RL training

The key guarantee: **reward conservation** — the sum of redistributed step rewards always equals the original trajectory reward.

## Architecture

```
                        verl Training Loop (ray_trainer.py)
                        ┌──────────────────────────────────────┐
                        │                                      │
 Rollout (vLLM/SGLang)  │  ┌─────────────────────┐            │
 ───────────────────────►│  │  extract_reward()    │            │
                        │  │  rm_scores → R       │            │
                        │  └────────┬────────────┘            │
                        │           │                          │
                        │           ▼                          │
                        │  ┌─────────────────────────────┐    │
                        │  │  ★ PNS Redistribution Hook  │    │
                        │  │                             │    │
                        │  │  1. Segment response text   │    │
                        │  │  2. Score each step (PNS)   │    │
                        │  │  3. Compute surplus shares  │    │
                        │  │  4. Allocate step rewards   │    │
                        │  │  5. Map to token positions  │    │
                        │  │                             │    │
                        │  │  token_level_scores[i] =    │    │
                        │  │    broadcast(step_rewards)  │    │
                        │  └────────┬────────────────────┘    │
                        │           │                          │
                        │           ▼                          │
                        │  ┌─────────────────────┐            │
                        │  │  compute_advantage() │            │
                        │  │  (GRPO / GAE / ...)  │            │
                        │  └─────────────────────┘            │
                        └──────────────────────────────────────┘
```

## Module Structure

```
verl/
├── utils/
│   ├── pns_reward_allocation.py      # Core math: surplus shares, reward allocation, ablation variants
│   ├── pns_step_segmenter.py         # Extensible step segmentation (4 built-in strategies)
│   ├── pns_token_mapping.py          # Step→token mapping via offset_mapping + fallback
│   └── pns_reward_redistributor.py   # End-to-end: segment → score → allocate → broadcast
├── trainer/
│   ├── config/
│   │   └── algorithm.py              # PNSRedistributionConfig dataclass (modified)
│   └── ppo/
│       └── ray_trainer.py            # PNS hook integration point (modified)
└── ...

tests/
└── utils/
    └── test_pns_reward_allocation.py  # Unit tests (sanity checks from spec)
```

### Module Details

| Module | Purpose |
|--------|---------|
| `pns_reward_allocation.py` | Implements the core formula: `r_t = (1-α)·R/T + α·R·φ_t` where `φ_t` is the normalized surplus share. Includes 3 ablation variants (uniform, direct normalized, surplus). |
| `pns_step_segmenter.py` | Registry-based step segmentation. Built-in strategies: `double_newline`, `step_marker`, `think_tag`, `sentence`. Easy to extend with `@register_segmenter("name")`. |
| `pns_token_mapping.py` | Maps step character boundaries to token indices using HuggingFace `offset_mapping`. Falls back to proportional allocation when re-encoding doesn't match rollout tokens. |
| `pns_reward_redistributor.py` | Orchestrates the full pipeline per batch. Reads PNS scores from `non_tensor_batch` or calls a user-supplied scorer function. External scorer calls are batched across the whole RL batch when the scorer exposes `score_batches`. |
| `pns_deberta_scorer.py` | DeBERTa-v3-large external scorer wrapper. It supports both `score_steps(step_texts)` for compatibility and batched `score_step_batches(step_text_batches)` for faster online scoring. |

## Mathematical Formulation

For a trajectory with final reward `R`, `T` reasoning steps, and PNS scores `c_1, ..., c_T`:

**Step 1: Compute surplus**
```
u_t = max(c_t - min(c_j), 0)
```

**Step 2: Normalize**
```
φ_t = u_t / Σ u_j    (if Σ u_j > 0, else φ_t = 1/T)
```

**Step 3: Allocate**
```
r_t = (1-α) · R/T + α · R · φ_t
```

This guarantees `Σ r_t = R` (reward conservation) for any `α ∈ [0, 1]`.

## Quick Start

### 1. Enable PNS in YAML config

```yaml
algorithm:
  adv_estimator: grpo
  pns_redistribution:
    enable: true
    alpha: 0.5
    mode: regression          # or "classification"
    variant: surplus          # "uniform" | "direct_normalized" | "surplus"
    step_segmenter: double_newline
    pns_score_key: pns_scores  # key in non_tensor_batch
```

### 2. Provide PNS scores

**Option A: Return PNS scores from your reward function**

In your custom `compute_score` function (specified via `reward.custom_reward_function.path`):

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Your existing reward logic
    final_score = evaluate_answer(solution_str, ground_truth)

    # Split into steps and score each with your PNS model
    steps = solution_str.split("\n\n")
    pns_scores = pns_model.score(steps)  # Your pre-trained PNS model

    return {
        "score": final_score,
        "pns_scores": pns_scores,   # List[float], one per step
    }
```

The `pns_scores` will be automatically passed through `reward_extra_info` → `non_tensor_batch` to the PNS redistribution hook.

**Option B: Provide an external scorer function**

```yaml
algorithm:
  pns_redistribution:
    enable: true
    pns_scorer_path: /path/to/my_pns_scorer.py
    pns_scorer_name: score_steps
```

Where `my_pns_scorer.py` contains:

```python
def score_steps(step_texts: list[str]) -> list[float]:
    """Score each reasoning step and return PNS scores."""
    # Call your pre-trained model here
    return model.predict(step_texts)
```

For high-throughput online scoring, the scorer may also expose a batched API:

```python
def score_step_batches(step_text_batches: list[list[str]]) -> list[list[float]]:
    """Score all rollout samples in one batched model pass."""
    return model.predict_batches(step_text_batches)
```

`redistribute_token_rewards_with_pns()` automatically uses `score_steps.score_batches`
when present. The included DeBERTa scorer attaches this batched API by default, so
an RL batch is scored as one large DeBERTa workload instead of hundreds of small
per-sample calls.

The included DeBERTa scorer is configured through environment variables:

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `PNS_DEBERTA_CKPT` | required | Local checkpoint directory for the trained PN/PNS scorer |
| `PNS_DEBERTA_DEVICE` | auto | Override scorer device, e.g. `cuda` or `cpu` |
| `PNS_DEBERTA_BATCH_SIZE` | `128` | Batch size for DeBERTa step scoring |
| `PNS_DEBERTA_MAX_LENGTH` | `512` | Tokenizer truncation length for each formatted step example |

### 3. Run training

No other changes needed. Run verl training as usual, or use the provided launch
scripts:

| Script | Environment | Notes |
|--------|-------------|-------|
| `scripts/run_grpo_pns_official_sif.sh` | Singularity | Recommended cluster entrypoint. Mirrors verl's official example style while using the prebuilt `verl_sgl059.sif` image. |
| `scripts/run_grpo_pns.sh` | Python venv | Non-container fallback using the local `envs/verl_sglang` Python environment. |

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.pns_redistribution.enable=true \
    algorithm.pns_redistribution.alpha=0.5 \
    ...
```

## Running Tests

```bash
cd verl/
python -c "
import torch, sys; sys.path.insert(0, '.')
exec(open('verl/utils/pns_reward_allocation.py').read())
exec(open('verl/utils/pns_step_segmenter.py').read())

# Sanity check: reward conservation
R = torch.tensor(5.0)
scores = torch.tensor([0.2, 0.5, 0.8, 0.3])
shares = compute_normalized_surplus_shares(scores)
step_rewards = allocate_reward(R, shares, alpha=0.5)
assert torch.allclose(step_rewards.sum(), R, atol=1e-6)
print('Reward conservation: PASS')

# Sanity check: uniform fallback
scores_eq = torch.tensor([0.5, 0.5, 0.5, 0.5])
sr = allocate_reward(torch.tensor(4.0), compute_normalized_surplus_shares(scores_eq), alpha=0.5)
assert torch.allclose(sr, torch.tensor([1.0, 1.0, 1.0, 1.0]), atol=1e-6)
print('Uniform fallback: PASS')
"
```

With pytest (if available):

```bash
pytest tests/utils/test_pns_reward_allocation.py -v
```

## Step-Prompt Evaluation Scripts

The repository includes helper scripts for evaluating whether GRPO/PNS
checkpoints can solve the validation set while producing explicit, parseable
reasoning steps. These scripts are intended for controlled comparisons where
baseline checkpoints are prompted to output `Step 1:`, `Step 2:`, ... style
reasoning before the final answer.

### 1. Create step-prompt validation data

```bash
python scripts/create_step_prompt_validation_data.py
```

This writes:

| File | Description |
|------|-------------|
| `data/step_prompt_eval/gsm8k_test.parquet` | GSM8K validation prompts with explicit `Step N:` and `#### <answer>` requirements |
| `data/step_prompt_eval/math_test.parquet` | MATH validation prompts with explicit `Step N:` and `\boxed{}` requirements |

### 2. Direct single-GPU vLLM evaluation

For fast evaluation on a single GPU node, use the vLLM direct evaluator. The
script downloads the uploaded Verl actor checkpoint from Hugging Face if needed,
merges FSDP actor shards into a standard Hugging Face model, runs vLLM
generation, and writes JSONL outputs plus a CSV summary.

```bash
sbatch \
  --partition=rp6b-1-gm96-c8-m64 \
  --nodelist=rp6b-1-gm96-c8-m64-dy-g7e-2xlarge-1 \
  --cpus-per-task=8 \
  --mem=60G \
  --gpus=1 \
  --export=ALL,GRPO_STEP_PROMPT_VARIANT=native,HF_TOKEN="$HF_TOKEN",HF_HUB_DISABLE_XET=1 \
  scripts/run_direct_grpo_step_prompt_eval_vllm.sh
```

Set `GRPO_STEP_PROMPT_VARIANT` to:

| Variant | Checkpoint source |
|---------|-------------------|
| `native` | `drdoggo/pns_grpo_native/global_step_200` |
| `format_length` | `drdoggo/pns_grpo_format_length/global_step_200` |

Outputs are written under:

```text
outputs/direct_step_prompt_eval_vllm/<variant>_<job_id>/
├── global_step_200.jsonl
└── summary.csv
```

`summary.csv` reports per-dataset and overall metrics:

- total sample count
- number and rate of `Step N:` compliant responses
- score over all responses
- score over step-compliant responses only
- output length
- average number of step markers

### 3. Direct Transformers fallback

If vLLM is unavailable, the slower Transformers-based evaluator can be used:

```bash
bash scripts/run_direct_grpo_step_prompt_eval.sh
```

This path is useful for debugging but is much slower for full GSM8K+MATH
evaluation.

### 4. Verl val-only evaluation

For validation through the normal Verl trainer stack, use:

```bash
sbatch --array=0-1 scripts/validate_grpo_step_prompt.sh
```

This runs `trainer.val_only=True` on the baseline checkpoints. It requires the
checkpoint world size to match the requested GPU layout, so it is less flexible
than the direct vLLM evaluator for single-GPU nodes.

To validate existing PNS checkpoints, use:

```bash
sbatch --array=0-1 --export=ALL,PNS_VAL_STEPS="1500 1600" scripts/validate_pns_checkpoint.sh
```

### 5. Summarize JSONL outputs

Any step-prompt JSONL dump can be summarized with:

```bash
python scripts/summarize_step_prompt_validation.py \
  outputs/direct_step_prompt_eval_vllm/native_<job_id>/global_step_200.jsonl
```

To write a CSV file:

```bash
python scripts/summarize_step_prompt_validation.py \
  outputs/direct_step_prompt_eval_vllm/native_<job_id>/global_step_200.jsonl \
  --output outputs/direct_step_prompt_eval_vllm/native_<job_id>/summary.csv
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable PNS redistribution |
| `alpha` | float | `0.5` | Interpolation: 0=uniform, 1=fully PNS-driven |
| `eps` | float | `1e-8` | Numerical stability constant |
| `mode` | str | `"regression"` | PNS output type: `"regression"` or `"classification"` |
| `pns_values` | list | `[0, 1/3, 2/3, 1]` | Class-to-value mapping (classification mode only) |
| `variant` | str | `"surplus"` | Ablation: `"uniform"` / `"direct_normalized"` / `"surplus"` |
| `step_segmenter` | str | `"double_newline"` | Step segmentation strategy |
| `pns_score_key` | str | `"pns_scores"` | Key in `non_tensor_batch` for pre-computed scores |
| `pns_scorer_path` | str | `null` | Path to external scorer Python file |
| `pns_scorer_name` | str | `"score_steps"` | Function name in scorer file |
| `scorer_ray_actor` | bool | `false` | Run the external scorer in a dedicated Ray actor instead of the trainer process |
| `scorer_num_gpus` | float | `1` | GPU resources reserved for the scorer actor when `scorer_ray_actor=true` |
| `scorer_num_cpus` | int | `4` | CPU resources reserved for the scorer actor when `scorer_ray_actor=true` |

## Online DeBERTa Scoring Notes

The current trained scorer returns scalar scores in `{0.0, 1.0, 2.0}`, so the
training scripts set `++algorithm.pns_redistribution.mode=regression`. Do not use
`classification` unless the scorer returns class probabilities/logits shaped
`[num_steps, num_classes]`.

For online DeBERTa scoring during GRPO, prefer a dedicated Ray scorer actor:

```bash
++algorithm.pns_redistribution.scorer_ray_actor=true \
++algorithm.pns_redistribution.scorer_num_gpus=1 \
++algorithm.pns_redistribution.scorer_num_cpus=4
```

The provided Singularity launch script requests 5 GPUs but keeps
`trainer.n_gpus_per_node=4`: four GPUs are used by actor/rollout/ref workers and
one GPU is reserved for the DeBERTa scorer actor. This avoids the trainer driver
falling back to CPU for PNS scoring.

When using the bundled DeBERTa scorer online, PNS redistribution logs:

- `pns/external_scoring_seconds`
- `pns/external_scoring_samples`
- `pns/external_scoring_steps`

These metrics are useful for checking whether step scoring is the bottleneck.
Detailed per-step records are written to `outputs/pns_step_scores/*.jsonl` when
`PNS_STEP_SCORE_LOG_PATH` is set by the launch script.

## Step Segmentation Strategies

| Strategy | Delimiter | Use Case |
|----------|-----------|----------|
| `double_newline` | `\n\n` | General reasoning (default) |
| `step_marker` | `Step N:` | Structured step-by-step reasoning |
| `think_tag` | `<think>...</think>` | Models with thinking tags (e.g. Qwen3) |
| `sentence` | `.` / `!` / `?` | Fine-grained sentence-level |

Add custom strategies:

```python
from verl.utils.pns_step_segmenter import register_segmenter, StepSegment

@register_segmenter("my_strategy")
def my_segmenter(text: str) -> list[StepSegment]:
    # Your custom logic here
    return [StepSegment(text=..., char_start=..., char_end=...)]
```

## Ablation Variants

| Variant | Formula | Description |
|---------|---------|-------------|
| **A: uniform** | `r_t = R/T` | Equal allocation (baseline) |
| **B: direct_normalized** | `r_t = (1-α)R/T + αR·(c_t/Σc_j)` | Direct score normalization |
| **C: surplus** (default) | `r_t = (1-α)R/T + αR·φ_t` | Baseline + normalized surplus |

## Logged Metrics

When enabled, the following metrics are logged per batch:

- `pns/score_mean`, `pns/score_std` — PNS score statistics
- `pns/surplus_mean`, `pns/surplus_std` — Surplus value statistics
- `pns/share_entropy_mean` — Entropy of allocation shares
- `pns/max_share_mean` — Maximum share per trajectory
- `pns/uniform_fallback_pct` — Percentage of trajectories with flat PNS scores
- `pns/score_reward_correlation` — Correlation between PNS scores and allocated rewards

## License

This project is built on top of [verl](https://github.com/volcengine/verl), licensed under Apache 2.0.
