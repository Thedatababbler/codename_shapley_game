#!/bin/bash
#SBATCH --job-name=direct_grpo_vllm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/direct_grpo_vllm_eval-%j.out
#SBATCH --error=scripts/direct_grpo_vllm_eval-%j.err
#SBATCH --gpus=1

set -euo pipefail
set -x

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/zqin30/condaenvs/verl
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

VERL_DIR="/scratch/zqin30/project/repo/codename_shapley_game"
cd "${VERL_DIR}"

export HF_HOME=/scratch/zqin30/.cache/hf
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

variant="${GRPO_STEP_PROMPT_VARIANT:?Set GRPO_STEP_PROMPT_VARIANT to native or format_length}"
case "${variant}" in
    native)
        experiment_name="qwen2.5_7b_grpo_native_baseline_2gpu"
        hf_repo_id="drdoggo/pns_grpo_native"
        ;;
    format_length)
        experiment_name="qwen2.5_7b_grpo_format_length_2gpu"
        hf_repo_id="drdoggo/pns_grpo_format_length"
        ;;
    *)
        echo "Unknown variant: ${variant}" >&2
        exit 2
        ;;
esac

checkpoint_dir="${VERL_DIR}/checkpoints/pns rl/${experiment_name}"
actor_dir="${checkpoint_dir}/global_step_200/actor"
merged_dir="${VERL_DIR}/outputs/merged_hf_baselines/${variant}_global_step_200"
output_dir="${VERL_DIR}/outputs/direct_step_prompt_eval_vllm/${variant}_${SLURM_JOB_ID}"
output_jsonl="${output_dir}/global_step_200.jsonl"

if [[ ! -f "${VERL_DIR}/data/step_prompt_eval/gsm8k_test.parquet" || ! -f "${VERL_DIR}/data/step_prompt_eval/math_test.parquet" ]]; then
    "${PYTHON_BIN}" scripts/create_step_prompt_validation_data.py
fi

if [[ ! -f "${actor_dir}/model_world_size_2_rank_0.pt" || ! -f "${actor_dir}/model_world_size_2_rank_1.pt" ]]; then
    for attempt in 1 2 3 4 5; do
        if "${PYTHON_BIN}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${hf_repo_id}",
    repo_type="model",
    allow_patterns=[
        "global_step_200/actor/model_world_size_2_rank_*.pt",
        "global_step_200/actor/fsdp_config.json",
        "global_step_200/actor/huggingface/**",
    ],
    local_dir="${checkpoint_dir}",
    max_workers=1,
)
PY
        then
            break
        fi
        echo "snapshot_download failed on attempt ${attempt}; retrying after sleep"
        sleep $((attempt * 60))
    done
fi

if [[ ! -f "${actor_dir}/model_world_size_2_rank_0.pt" || ! -f "${actor_dir}/model_world_size_2_rank_1.pt" ]]; then
    echo "Missing actor shards under ${actor_dir}" >&2
    exit 2
fi

if [[ ! -f "${merged_dir}/model.safetensors" && ! -f "${merged_dir}/pytorch_model.bin" && ! -f "${merged_dir}/model-00001-of-00004.safetensors" ]]; then
    rm -rf "${merged_dir}"
    "${PYTHON_BIN}" scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${merged_dir}"
fi

mkdir -p "${output_dir}"
"${PYTHON_BIN}" scripts/direct_step_prompt_eval_vllm.py \
    --model "${merged_dir}" \
    --val-files data/step_prompt_eval/gsm8k_test.parquet data/step_prompt_eval/math_test.parquet \
    --output "${output_jsonl}" \
    --chunk-size "${DIRECT_EVAL_CHUNK_SIZE:-512}" \
    --max-new-tokens "${DIRECT_EVAL_MAX_NEW_TOKENS:-2048}" \
    --max-model-len "${DIRECT_EVAL_MAX_MODEL_LEN:-4096}"

"${PYTHON_BIN}" scripts/summarize_step_prompt_validation.py "${output_jsonl}" \
    --output "${output_dir}/summary.csv"

echo "Wrote ${output_jsonl}"
echo "Wrote ${output_dir}/summary.csv"
