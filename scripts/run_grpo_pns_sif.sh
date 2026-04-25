#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=grpo_pns_sif
#SBATCH --output=scripts/grpo_pns_sif-%j.log
#SBATCH --error=scripts/grpo_pns_sif-%j.log

# ============================================================
# GRPO + PNS Step-Level Reward Redistribution (Singularity 版)
# 用 verl 官方 sglang059 镜像, 把本地改过的 verl 源码 (含 PNS)
# 通过 PYTHONPATH 优先挂载, 绕开所有装环境问题
# ============================================================

set -euo pipefail

module purge 2>/dev/null || true
module load singularity/4.3.4

SIF="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sgl059.sif"
if [ ! -f "${SIF}" ]; then
    echo "ERROR: sif not found at ${SIF}"
    echo "请先 sbatch scripts/pull_verl_sif.sh 拉镜像"
    exit 1
fi

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
cd "${VERL_DIR}"

# ─── Env passed into container ───
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set; run: export WANDB_API_KEY=... before sbatch}"
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

export RAY_TMPDIR="/tmp/ray_verl_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((30*1024*1024*1024))

# PYTHONPATH 让容器内 python 优先用 RDS 上我们改过的 verl
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# ─── Models & data ───
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/causal_rl/pns_scorer/checkpoints/deberta_pn_scorer_v3/best}"
INTEGRATION_MODE="${INTEGRATION_MODE:-external_scorer}"

DATA_DIR="${VERL_DIR}/data"
GSM8K_TRAIN="${DATA_DIR}/gsm8k/train.parquet"
GSM8K_TEST="${DATA_DIR}/gsm8k/test.parquet"
MATH_TRAIN="${DATA_DIR}/math/train.parquet"
MATH_TEST="${DATA_DIR}/math/test.parquet"

prepare_data_in_container() {
    echo "=== 数据准备 (容器内) ==="
    if [[ -f "${GSM8K_TRAIN}" && -f "${MATH_TRAIN}" ]]; then
        echo "数据已存在, 跳过"
        return
    fi
    mkdir -p "${DATA_DIR}/gsm8k" "${DATA_DIR}/math"
    singularity exec --nv \
        --bind /mnt/rds/VipinRDS:/mnt/rds/VipinRDS \
        --env HF_HOME="${HF_HOME}" \
        --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
        --pwd "${VERL_DIR}" \
        "${SIF}" bash -c "
            python3 examples/data_preprocess/gsm8k.py --local_save_dir ${DATA_DIR}/gsm8k
            python3 examples/data_preprocess/math_dataset.py --local_save_dir ${DATA_DIR}/math
        "
}
prepare_data_in_container

# ─── 训练超参 ───
N_GPUS=8
TP_SIZE=2
TRAIN_BATCH_SIZE=256
ROLLOUT_N=5
MINI_BATCH_SIZE=64
MICRO_BATCH_SIZE=8
MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=2048
TOTAL_EPOCHS=15
SAVE_FREQ=5
TEST_FREQ=3
LR=1e-6

# PNS 参数
PNS_ALPHA=0.5
PNS_MODE="classification"
PNS_VALUES="[0.0,1.0,2.0]"
PNS_VARIANT="surplus"
PNS_SEGMENTER="double_newline"

echo ""
echo "=========================================="
echo "GRPO + PNS Training (Singularity)"
echo "=========================================="
echo "Actor Model:  ${ACTOR_MODEL}"
echo "PNS Scorer:   ${PNS_DEBERTA_CKPT}"
echo "Integration:  ${INTEGRATION_MODE}"
echo "GPUs:         ${N_GPUS} × $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "TP Size:      ${TP_SIZE}"
echo "Batch Size:   ${TRAIN_BATCH_SIZE} (mini=${MINI_BATCH_SIZE})"
echo "Rollout N:    ${ROLLOUT_N}"
echo "Epochs:       ${TOTAL_EPOCHS}"
echo "PNS α:        ${PNS_ALPHA}"
echo "SIF:          ${SIF}"
echo "=========================================="
echo ""

# ─── PNS ckpt 检查 ───
if [[ ! -d "${PNS_DEBERTA_CKPT}" ]]; then
    echo "[WARN] PNS ckpt not found. Disabling PNS redistribution."
    PNS_ENABLE="false"
else
    echo "[OK] PNS ckpt: ${PNS_DEBERTA_CKPT}"
    PNS_ENABLE="true"
fi

TRAIN_FILES="['${GSM8K_TRAIN}','${MATH_TRAIN}']"
VAL_FILES="['${GSM8K_TEST}','${MATH_TEST}']"

CMD=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000
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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000

    actor_rollout_ref.ref.fsdp_config.param_offload=True

    algorithm.use_kl_in_reward=False

    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_grpo_pns
    trainer.experiment_name=qwen2.5_7b_grpo_pns
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}

    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-64}
    ++ray_kwargs.ray_init.include_dashboard=false
)

if [[ "${PNS_ENABLE}" == "true" ]]; then
    CMD+=(
        ++algorithm.pns_redistribution.enable=true
        ++algorithm.pns_redistribution.alpha=${PNS_ALPHA}
        ++algorithm.pns_redistribution.mode=${PNS_MODE}
        "++algorithm.pns_redistribution.pns_values=${PNS_VALUES}"
        ++algorithm.pns_redistribution.variant=${PNS_VARIANT}
        ++algorithm.pns_redistribution.step_segmenter=${PNS_SEGMENTER}
    )
    if [[ "${INTEGRATION_MODE}" == "external_scorer" ]]; then
        PNS_SCORER_PATH="${VERL_DIR}/verl/utils/pns_deberta_scorer.py"
        CMD+=(
            ++algorithm.pns_redistribution.pns_scorer_path="${PNS_SCORER_PATH}"
            ++algorithm.pns_redistribution.pns_scorer_name=score_steps
        )
        echo "[PNS] external scorer: ${PNS_SCORER_PATH}"
    else
        CMD+=(
            reward.custom_reward_function.path=verl/utils/reward_score/math_pns_reward.py
            reward.custom_reward_function.name=compute_score
            ++algorithm.pns_redistribution.pns_score_key=pns_scores
        )
        echo "[PNS] custom reward function mode"
    fi
else
    CMD+=(++algorithm.pns_redistribution.enable=false)
fi

echo "Command: ${CMD[*]}"
echo ""

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
    --env RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY}" \
    --env PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    --pwd "${VERL_DIR}" \
    "${SIF}" \
    "${CMD[@]}"

EXIT_CODE=$?
rm -rf "${RAY_TMPDIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Training finished (exit: ${EXIT_CODE})"
echo "=========================================="
exit ${EXIT_CODE}
