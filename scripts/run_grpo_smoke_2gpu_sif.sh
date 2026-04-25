#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=grpo_smoke2
#SBATCH --output=scripts/grpo_smoke_2gpu-%j.log
#SBATCH --error=scripts/grpo_smoke_2gpu-%j.log

# ============================================================
# 更轻的 smoke test: 2 GPU, Qwen2.5-1.5B, 用 gpu 分区 (有空闲)
# 目的: 验证 singularity + verl + sglang pipeline 能跑通
# ============================================================

set -euo pipefail

module purge 2>/dev/null || true
module load singularity/4.3.4

SIF="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sgl059.sif"
if [ ! -f "${SIF}" ]; then
    echo "ERROR: sif not found at ${SIF}"
    exit 1
fi

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"

export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

export RAY_TMPDIR="/tmp/ray_verl_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0

export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# 轻量参数
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
N_GPUS=2
TP_SIZE=1
TRAIN_BATCH_SIZE=16
ROLLOUT_N=2
MINI_BATCH_SIZE=8
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=1024
TOTAL_EPOCHS=1
SAVE_FREQ=10000
TEST_FREQ=10000
LR=1e-6

DATA_DIR="${VERL_DIR}/data"
GSM8K_TRAIN="${DATA_DIR}/gsm8k/train.parquet"
GSM8K_TEST="${DATA_DIR}/gsm8k/test.parquet"

echo "============================================"
echo "Date:    $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "SIF:     ${SIF} ($(du -h ${SIF} | cut -f1))"
echo "Model:   ${ACTOR_MODEL}"
echo "============================================"

if [[ ! -f "${GSM8K_TRAIN}" ]]; then
    echo ">>> GSM8K data missing — preparing inside container..."
    singularity exec --nv \
        --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
        --env HF_HOME="${HF_HOME}" \
        --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
        --pwd "${VERL_DIR}" \
        "${SIF}" \
        python3 examples/data_preprocess/gsm8k.py --local_save_dir "${DATA_DIR}/gsm8k"
fi

echo ""
echo ">>> CUDA compat check inside container..."
singularity exec --nv \
    --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
    "${SIF}" bash -c '
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
        python3 -c "
import torch
print(f\"  torch={torch.__version__}\")
print(f\"  cuda_runtime={torch.version.cuda}\")
print(f\"  cuda_available={torch.cuda.is_available()}\")
print(f\"  device_count={torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"  device_0={torch.cuda.get_device_name(0)}\")
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f\"  matmul OK, result shape={y.shape}\")
"
    '

echo ""
echo ">>> Launching training..."

CMD=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    data.train_files="['${GSM8K_TRAIN}']"
    data.val_files="['${GSM8K_TEST}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LEN}
    data.max_response_length=${MAX_RESPONSE_LEN}
    data.filter_overlong_prompts=True
    data.truncation=error

    actor_rollout_ref.model.path=${ACTOR_MODEL}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8000

    actor_rollout_ref.ref.fsdp_config.param_offload=True

    algorithm.use_kl_in_reward=False
    ++algorithm.pns_redistribution.enable=false

    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=verl_grpo_smoke
    trainer.experiment_name=qwen2.5_1.5b_smoke2gpu
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}

    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-16}
    ++ray_kwargs.ray_init.include_dashboard=false
)

singularity exec --nv \
    --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
    --bind /tmp:/tmp \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE}" \
    --env CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
    --env HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR}" \
    --env RAY_TMPDIR="${RAY_TMPDIR}" \
    --env RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    --pwd "${VERL_DIR}" \
    "${SIF}" \
    "${CMD[@]}"

EXIT_CODE=$?
rm -rf "${RAY_TMPDIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Smoke2GPU test finished (exit: ${EXIT_CODE})"
echo "=========================================="
exit ${EXIT_CODE}
