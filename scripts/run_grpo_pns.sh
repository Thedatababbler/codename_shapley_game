#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6
#SBATCH --exclude=aisct03
#SBATCH --job-name=grpo_pns
#SBATCH --output=scripts/grpo_pns-%j.log
#SBATCH --error=scripts/grpo_pns-%j.log

# Official-example-style GRPO + PNS launch script using the Python/uv venv.
# This intentionally mirrors verl/examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh:
# activate the environment first, then call `python3 -m verl.trainer.main_ppo`
# with flat Hydra overrides.  PNS-specific overrides are appended at the end.

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

source /mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sglang/bin/activate

ulimit -n 65535

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
JOB_ID="${SLURM_JOB_ID:-manual}"
cd "${VERL_DIR}"

export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=0
export HYDRA_FULL_ERROR=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_LOGGING_LEVEL=INFO
export RAY_TMPDIR="/tmp/ray_verl_${JOB_ID}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((30*1024*1024*1024))
mkdir -p "${RAY_TMPDIR}"
trap 'rm -rf "${RAY_TMPDIR}" 2>/dev/null || true' EXIT

PYTHON_ROOT=/usr/local/easybuild_allnodes/software/Python/3.12.3-GCCcore-13.3.0
GCC_ROOT=/usr/local/easybuild_allnodes/software/GCCcore/13.3.0
PYTHON_INCLUDE="${PYTHON_ROOT}/include/python3.12"
export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"
export C_INCLUDE_PATH="${PYTHON_INCLUDE}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"

# Keep logger local by default to avoid requiring network/W&B credentials.
# To enable W&B from sbatch: TRAIN_LOGGER='["console","wandb"]' WANDB_API_KEY=... sbatch ...
TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\"]}"

# PNS scorer and detailed per-step logging.
export PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/causal_rl/pns_scorer/checkpoints/deberta_pn_scorer_v3/best}"
export PNS_STEP_SCORE_LOG_DIR="${VERL_DIR}/outputs/pns_step_scores"
mkdir -p "${PNS_STEP_SCORE_LOG_DIR}"
export PNS_STEP_SCORE_LOG_PATH="${PNS_STEP_SCORE_LOG_DIR}/grpo_pns_${JOB_ID}.jsonl"
export PNS_STEP_SCORE_STDOUT_SAMPLES="${PNS_STEP_SCORE_STDOUT_SAMPLES:-2}"
export PNS_STEP_SCORE_PREVIEW_CHARS="${PNS_STEP_SCORE_PREVIEW_CHARS:-200}"
export PNS_DEBERTA_BATCH_SIZE="${PNS_DEBERTA_BATCH_SIZE:-128}"
export PNS_DEBERTA_MAX_LENGTH="${PNS_DEBERTA_MAX_LENGTH:-512}"

model_path="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
gsm8k_train_path="${VERL_DIR}/data/gsm8k/train.parquet"
gsm8k_test_path="${VERL_DIR}/data/gsm8k/test.parquet"
math_train_path="${VERL_DIR}/data/math/train.parquet"
math_test_path="${VERL_DIR}/data/math/test.parquet"
train_files="['${gsm8k_train_path}','${math_train_path}']"
test_files="['${gsm8k_test_path}','${math_test_path}']"

if [[ ! -f "${gsm8k_train_path}" || ! -f "${math_train_path}" ]]; then
    python3 examples/data_preprocess/gsm8k.py --local_save_dir "${VERL_DIR}/data/gsm8k"
    python3 examples/data_preprocess/math_dataset.py --local_save_dir "${VERL_DIR}/data/math"
fi

echo "=========================================="
echo "GRPO + PNS official-style Python venv run"
echo "job_id=${JOB_ID}"
echo "node=$(hostname)"
echo "python=$(which python3)"
echo "model=${model_path}"
echo "train_files=${train_files}"
echo "PNS_DEBERTA_CKPT=${PNS_DEBERTA_CKPT}"
echo "PNS_STEP_SCORE_LOG_PATH=${PNS_STEP_SCORE_LOG_PATH}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAIN_LOGGER}" \
    trainer.project_name='pns rl' \
    trainer.experiment_name='qwen2.5_7b_grpo_pns_scorer_v3_python_official_style' \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=3 \
    trainer.total_epochs=15 \
    ray_kwargs.ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-48}" \
    ++ray_kwargs.ray_init.include_dashboard=false \
    ++algorithm.pns_redistribution.enable=true \
    ++algorithm.pns_redistribution.alpha=0.5 \
    ++algorithm.pns_redistribution.mode=regression \
    ++algorithm.pns_redistribution.variant=surplus \
    ++algorithm.pns_redistribution.step_segmenter=double_newline \
    ++algorithm.pns_redistribution.pns_scorer_path="${VERL_DIR}/verl/utils/pns_deberta_scorer.py" \
    ++algorithm.pns_redistribution.pns_scorer_name=score_steps
