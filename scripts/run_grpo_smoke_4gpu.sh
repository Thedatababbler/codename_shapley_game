#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=grpo_smoke4
#SBATCH --output=scripts/grpo_smoke_4gpu-%j.log
#SBATCH --error=scripts/grpo_smoke_4gpu-%j.log

# ============================================================
# 4 卡 GRPO + SGLang 冒烟（小模型、短序列、少 epoch）
# 目的：先跑通 verl 栈，再开 8 卡 + 大模型 + PNS。
#
# 可选环境变量：
#   ENABLE_PNS=1          打开 PNS（默认关闭）
#   ACTOR_MODEL=...       覆盖默认小模型
# ============================================================

set -euo pipefail

# 环境使用 python 3.12 + torch 2.8 + cu126 (pre-built wheels)
# driver 12.7 能跑 cu126 runtime，不再需要 module load CUDA
# 先 module load (带上所有运行依赖 libffi/SQLite/OpenSSL 等)，再 activate venv
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sglang/bin/activate

export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0

PYTHON_ROOT=/usr/local/easybuild_allnodes/software/Python/3.12.3-GCCcore-13.3.0
GCC_ROOT=/usr/local/easybuild_allnodes/software/GCCcore/13.3.0
PYTHON_INCLUDE="${PYTHON_ROOT}/include/python3.12"
export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"
# Triton JIT 需要 Python.h
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"
export C_INCLUDE_PATH="${PYTHON_INCLUDE}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"

# A100 (sm80) 不支持 FlashAttention3；verl 默认 mm_attention_backend=fa3 会导致 SGLang
# scheduler 在 init_torch_distributed 时报 cudaErrorDevicesUnavailable。
# 另外加几个 CUDA 稳定性 env 变量，避免 NCCL / 内存池与 FSDP 冲突。
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
cd "${VERL_DIR}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# 冒烟前快速校验（与训练同解释器）
python3 - <<'PY'
import importlib
for m in ("cachetools", "PIL.Image", "yaml", "sglang", "ray", "torch", "flash_attn"):
    importlib.import_module(m)
from verl.experimental.agent_loop import AgentLoopManager
print("[smoke] imports OK, AgentLoopManager loadable")
PY

export RAY_TMPDIR="/tmp/ray_verl_smoke"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((8*1024*1024*1024))
export HYDRA_FULL_ERROR=1

ray stop --force 2>/dev/null || true
sleep 1
rm -rf "${RAY_TMPDIR}"/session_* 2>/dev/null || true

echo "Node: $(hostname)"
nvidia-smi -L | head -4

N_GPUS=4
TP_SIZE=1
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_BATCH_SIZE=32
ROLLOUT_N=4
MINI_BATCH_SIZE=16
MAX_PROMPT_LEN=512
MAX_RESPONSE_LEN=512
TOTAL_EPOCHS=1
SAVE_FREQ=1
TEST_FREQ=1
LR=5e-6

DATA_DIR="${VERL_DIR}/data"
GSM8K_TRAIN="${DATA_DIR}/gsm8k/train.parquet"
GSM8K_TEST="${DATA_DIR}/gsm8k/test.parquet"

if [[ ! -f "${GSM8K_TRAIN}" ]]; then
    echo "[ERR] Missing ${GSM8K_TRAIN}. Run full data prep or: python3 examples/data_preprocess/gsm8k.py --local_save_dir ${DATA_DIR}/gsm8k"
    exit 1
fi

TRAIN_FILES="['${GSM8K_TRAIN}']"
VAL_FILES="['${GSM8K_TEST}']"

PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/causal_rl/pns_scorer/checkpoints/deberta_pn_scorer_v3/best}"
PNS_SCORER_PATH="${VERL_DIR}/verl/utils/pns_deberta_scorer.py"

if [[ "${ENABLE_PNS:-0}" == "1" ]] && [[ -d "${PNS_DEBERTA_CKPT}" ]]; then
    PNS_ENABLE="true"
else
    PNS_ENABLE="false"
fi

echo ""
echo "======== GRPO smoke (4 GPU) ========"
echo "Model:     ${ACTOR_MODEL}"
echo "GPUs:      ${N_GPUS}  TP=${TP_SIZE}"
echo "Seq:       prompt=${MAX_PROMPT_LEN} response=${MAX_RESPONSE_LEN}"
echo "Batch:     train=${TRAIN_BATCH_SIZE} mini=${MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"
echo "Epochs:    ${TOTAL_EPOCHS}"
echo "PNS:       ${PNS_ENABLE} (set ENABLE_PNS=1 to enable)"
echo "======================================"
echo ""

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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192

    actor_rollout_ref.ref.fsdp_config.param_offload=True

    algorithm.use_kl_in_reward=False

    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_grpo_smoke
    trainer.experiment_name=smoke_4gpu_1p5b_sglang
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}

    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-32}
    ++ray_kwargs.ray_init.include_dashboard=false
)

if [[ "${PNS_ENABLE}" == "true" ]]; then
    CMD+=(
        ++algorithm.pns_redistribution.enable=true
        ++algorithm.pns_redistribution.alpha=0.5
        ++algorithm.pns_redistribution.mode=classification
        "++algorithm.pns_redistribution.pns_values=[0.0,1.0,2.0]"
        ++algorithm.pns_redistribution.variant=surplus
        ++algorithm.pns_redistribution.step_segmenter=double_newline
        ++algorithm.pns_redistribution.pns_scorer_path="${PNS_SCORER_PATH}"
        ++algorithm.pns_redistribution.pns_scorer_name=score_steps
    )
else
    CMD+=(++algorithm.pns_redistribution.enable=false)
fi

echo "Starting: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "======== Smoke finished OK ========"
