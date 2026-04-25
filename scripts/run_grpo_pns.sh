#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=grpo_pns
#SBATCH --output=scripts/grpo_pns-%j.log
#SBATCH --error=scripts/grpo_pns-%j.log

# ============================================================
# GRPO + PNS Step-Level Reward Redistribution
#
# 在 verl 框架上用 GRPO 训练 LLM，同时用预训练的 DeBERTa PNS
# 打分器进行步级奖励再分配。
#
# 集群: 8× A100-80GB (single node)
# Actor/Rollout 模型: Qwen/Qwen2.5-7B-Instruct (可换)
# PNS Scorer: microsoft/deberta-v3-large 4-class (预训练 ckpt)
# 数据: GSM8k + MATH
#
# 集成方式 (二选一, 默认 Option B):
#   A) Custom reward function — 在 compute_score 中同时返回
#      correctness + pns_scores，PNS scores 通过 non_tensor_batch 传入
#   B) External scorer — pns_scorer_path 指向 DeBERTa wrapper，
#      在 redistribute 阶段动态打分
#
# ============================================================

set -euo pipefail

# ─── 环境 ───
# 环境用 python 3.12 + torch 2.8 + cu126 (wheel 自带 runtime, driver 12.7 兼容)
# 先 module load (带上 libffi/SQLite/OpenSSL 等)，再 activate venv
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
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"
export C_INCLUDE_PATH="${PYTHON_INCLUDE}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"

export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# WandB
# WandB API key — 建议通过 `export WANDB_API_KEY=...` 在提交前设置, 不要写进 repo
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set; run: export WANDB_API_KEY=... before sbatch}"

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
cd "${VERL_DIR}"
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# Ray 配置: 使用本地短路径 /tmp/ray_verl 避免 Unix socket 路径过长
export RAY_TMPDIR="/tmp/ray_verl"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((30*1024*1024*1024))
export HYDRA_FULL_ERROR=1

# 清理残留 Ray 进程
ray stop --force 2>/dev/null || true
sleep 2
# 清理旧 Ray 临时文件
rm -rf /tmp/ray_verl/session_* 2>/dev/null || true
rm -rf /tmp/ray /tmp/job.*/ray 2>/dev/null || true

echo "Node: $(hostname), /dev/shm size:"
df -h /dev/shm 2>/dev/null || echo "N/A"
echo "ulimit -n: $(ulimit -n)"

# ─── 模型选择 ───
# 训练目标模型 (actor)
#   - Qwen2.5-7B-Instruct:  适合 8 卡, TP=2 rollout
#   - Qwen2.5-3B-Instruct:  更快收敛, 可 TP=1
#   - Qwen2.5-1.5B-Instruct: 最轻量, 快速验证
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# PNS DeBERTa 打分器 checkpoint (v3: 3-class, freeze-tuned)
export PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/causal_rl/pns_scorer/checkpoints/deberta_pn_scorer_v3/best}"

# ─── 集成方式选择 ───
# "external_scorer" — 使用 pns_scorer_path (推荐, 自动获取 prompt 上下文)
# "reward_function" — 在 compute_score 返回 pns_scores
INTEGRATION_MODE="${INTEGRATION_MODE:-external_scorer}"

# ─── 数据准备 ───
DATA_DIR="${VERL_DIR}/data"
GSM8K_TRAIN="${DATA_DIR}/gsm8k/train.parquet"
GSM8K_TEST="${DATA_DIR}/gsm8k/test.parquet"
MATH_TRAIN="${DATA_DIR}/math/train.parquet"
MATH_TEST="${DATA_DIR}/math/test.parquet"

prepare_data() {
    echo "=== 数据准备 ==="
    if [[ -f "${GSM8K_TRAIN}" && -f "${MATH_TRAIN}" ]]; then
        echo "数据已存在, 跳过下载"
        return
    fi

    mkdir -p "${DATA_DIR}/gsm8k" "${DATA_DIR}/math"

    echo "下载 GSM8k..."
    python3 examples/data_preprocess/gsm8k.py \
        --local_save_dir "${DATA_DIR}/gsm8k"

    echo "下载 MATH..."
    python3 examples/data_preprocess/math_dataset.py \
        --local_save_dir "${DATA_DIR}/math"

    echo "数据准备完成"
    ls -lh "${DATA_DIR}/gsm8k/"
    ls -lh "${DATA_DIR}/math/"
}

prepare_data

# ─── 训练超参 ───
N_GPUS=8
TP_SIZE=2                    # Rollout tensor parallelism (7B 用 2, 3B 用 1)
TRAIN_BATCH_SIZE=256         # 每个 iteration 的 prompt 数
ROLLOUT_N=5                  # 每个 prompt 采样 N 条 response (GRPO 需要)
MINI_BATCH_SIZE=64           # PPO mini-batch
MICRO_BATCH_SIZE=8           # 每 GPU 的 micro-batch
MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=2048        # 数学推理需要较长的 response
TOTAL_EPOCHS=15
SAVE_FREQ=5
TEST_FREQ=3
LR=1e-6

# PNS 参数
PNS_ALPHA=0.5                # 0 = 纯 uniform, 1 = 纯 PNS 驱动
PNS_MODE="classification"    # DeBERTa 输出 3 类
PNS_VALUES="[0.0,1.0,2.0]"  # low=0, mid=1, high=2
PNS_VARIANT="surplus"        # baseline + normalized surplus 分配
PNS_SEGMENTER="double_newline"

# ─── 环境信息 ───
echo ""
echo "=========================================="
echo "GRPO + PNS Training"
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
echo "PNS Mode:     ${PNS_MODE} / ${PNS_VARIANT}"
echo "=========================================="
echo ""

# ─── 验证 PNS checkpoint ───
if [[ ! -d "${PNS_DEBERTA_CKPT}" ]]; then
    echo "[WARN] PNS checkpoint not found: ${PNS_DEBERTA_CKPT}"
    echo "       PNS redistribution will be DISABLED until checkpoint is available."
    echo "       To train the scorer, run: pns_scorer/scripts/submit_train_deberta.sh"
    PNS_ENABLE="false"
else
    echo "[OK] PNS checkpoint found: ${PNS_DEBERTA_CKPT}"
    ls -lh "${PNS_DEBERTA_CKPT}/"
    PNS_ENABLE="true"
fi

# ─── 构建训练命令 ───
TRAIN_FILES="['${GSM8K_TRAIN}','${MATH_TRAIN}']"
VAL_FILES="['${GSM8K_TEST}','${MATH_TEST}']"

# 基础参数
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

    # Actor 模型
    actor_rollout_ref.model.path=${ACTOR_MODEL}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # Actor 优化器
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

    # Rollout (SGLang)
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000

    # Reference model
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    # Algorithm
    algorithm.use_kl_in_reward=False

    # Trainer
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_grpo_pns
    trainer.experiment_name=qwen2.5_7b_grpo_pns
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}

    # SLURM 环境下需要显式指定 CPU 数量, 否则 Ray 无法正确检测
    ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-64}
    ++ray_kwargs.ray_init.include_dashboard=false
)

# PNS redistribution 参数 (用 ++ 前缀强制创建/覆盖 Hydra struct 中的 Optional 字段)
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
        # Option B: 外部 scorer (推荐)
        PNS_SCORER_PATH="${VERL_DIR}/verl/utils/pns_deberta_scorer.py"
        CMD+=(
            ++algorithm.pns_redistribution.pns_scorer_path="${PNS_SCORER_PATH}"
            ++algorithm.pns_redistribution.pns_scorer_name=score_steps
        )
        echo "[PNS] Using external scorer: ${PNS_SCORER_PATH}"
    else
        # Option A: 通过 custom reward function 传递 pns_scores
        CMD+=(
            reward.custom_reward_function.path=verl/utils/reward_score/math_pns_reward.py
            reward.custom_reward_function.name=compute_score
            ++algorithm.pns_redistribution.pns_score_key=pns_scores
        )
        echo "[PNS] Using custom reward function with embedded PNS scoring"
    fi
else
    CMD+=(
        ++algorithm.pns_redistribution.enable=false
    )
    echo "[PNS] DISABLED — no checkpoint found"
fi

echo ""
echo "Starting training..."
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
