#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6
#SBATCH --exclude=aisct03
#SBATCH --job-name=grpo_pns_official
#SBATCH --output=scripts/grpo_pns_official-%j.log
#SBATCH --error=scripts/grpo_pns_official-%j.log

# Official-example-style GRPO + PNS script.
# Shape follows examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh:
# keep the base verl overrides recognizable, use Singularity only as the
# environment wrapper, and append PNS-specific overrides at the end.

set -euo pipefail
set -x

module purge 2>/dev/null || true
module load singularity/4.3.4

ulimit -n 65535

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
SIF="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sgl059.sif"
cd "${VERL_DIR}"

if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: SIF not found: ${SIF}" >&2
    exit 1
fi

WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be exported or passed via sbatch --export}"
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export WANDB_API_KEY
export HYDRA_FULL_ERROR=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_LOGGING_LEVEL=INFO
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

export RAY_TMPDIR="/tmp/ray_verl_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((30*1024*1024*1024))

# PNS scorer and detailed per-step logging.
export PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/causal_rl/pns_scorer/checkpoints/deberta_pn_scorer_v3/best}"
export PNS_STEP_SCORE_LOG_DIR="${VERL_DIR}/outputs/pns_step_scores"
mkdir -p "${PNS_STEP_SCORE_LOG_DIR}"
export PNS_STEP_SCORE_LOG_PATH="${PNS_STEP_SCORE_LOG_DIR}/grpo_pns_${SLURM_JOB_ID}.jsonl"
export PNS_STEP_SCORE_STDOUT_SAMPLES="${PNS_STEP_SCORE_STDOUT_SAMPLES:-2}"
export PNS_STEP_SCORE_PREVIEW_CHARS="${PNS_STEP_SCORE_PREVIEW_CHARS:-200}"

model_path="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
gsm8k_train_path="${VERL_DIR}/data/gsm8k/train.parquet"
gsm8k_test_path="${VERL_DIR}/data/gsm8k/test.parquet"
math_train_path="${VERL_DIR}/data/math/train.parquet"
math_test_path="${VERL_DIR}/data/math/test.parquet"
train_files="['${gsm8k_train_path}','${math_train_path}']"
test_files="['${gsm8k_test_path}','${math_test_path}']"

if [[ ! -f "${gsm8k_train_path}" || ! -f "${math_train_path}" ]]; then
    singularity exec --nv \
        --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
        --env HF_HOME="${HF_HOME}" \
        --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
        --pwd "${VERL_DIR}" \
        "${SIF}" bash -lc "
            python3 examples/data_preprocess/gsm8k.py --local_save_dir ${VERL_DIR}/data/gsm8k
            python3 examples/data_preprocess/math_dataset.py --local_save_dir ${VERL_DIR}/data/math
        "
fi

echo "=========================================="
echo "GRPO + PNS official-style Singularity run"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "model=${model_path}"
echo "train_files=${train_files}"
echo "PNS_DEBERTA_CKPT=${PNS_DEBERTA_CKPT}"
echo "PNS_STEP_SCORE_LOG_PATH=${PNS_STEP_SCORE_LOG_PATH}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

singularity exec --nv \
    --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
    --bind /tmp:/tmp \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR}" \
    --env NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE}" \
    --env CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
    --env VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL}" \
    --env RAY_TMPDIR="${RAY_TMPDIR}" \
    --env RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS}" \
    --env RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    --env PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT}" \
    --env PNS_STEP_SCORE_LOG_PATH="${PNS_STEP_SCORE_LOG_PATH}" \
    --env PNS_STEP_SCORE_STDOUT_SAMPLES="${PNS_STEP_SCORE_STDOUT_SAMPLES}" \
    --env PNS_STEP_SCORE_PREVIEW_CHARS="${PNS_STEP_SCORE_PREVIEW_CHARS}" \
    --pwd "${VERL_DIR}" \
    "${SIF}" \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${train_files}" \
        data.val_files="${test_files}" \
        data.train_batch_size=96 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="${model_path}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=24 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=6 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.multi_stage_wake_up=True \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='pns rl' \
        trainer.experiment_name='qwen2.5_7b_grpo_pns_scorer_v3_official_style' \
        trainer.n_gpus_per_node=6 \
        trainer.nnodes=1 \
        trainer.save_freq=5 \
        trainer.test_freq=3 \
        trainer.total_epochs=15 \
        ray_kwargs.ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-48}" \
        ++ray_kwargs.ray_init.include_dashboard=false \
        ++algorithm.pns_redistribution.enable=true \
        ++algorithm.pns_redistribution.alpha=0.5 \
        ++algorithm.pns_redistribution.mode=classification \
        ++algorithm.pns_redistribution.pns_values='[0.0,1.0,2.0]' \
        ++algorithm.pns_redistribution.variant=surplus \
        ++algorithm.pns_redistribution.step_segmenter=double_newline \
        ++algorithm.pns_redistribution.pns_scorer_path="${VERL_DIR}/verl/utils/pns_deberta_scorer.py" \
        ++algorithm.pns_redistribution.pns_scorer_name=score_steps

exit_code=$?
rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
exit "${exit_code}"
