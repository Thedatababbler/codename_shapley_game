#!/bin/bash
#SBATCH --job-name=pns_val_ckpt
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/pns_val_ckpt-%A_%a.out
#SBATCH --error=scripts/pns_val_ckpt-%A_%a.err
#SBATCH --gpus=2

set -euo pipefail
set -x

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/zqin30/condaenvs/verl
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
"${PYTHON_BIN}" -c 'import sys, ray; print(f"python={sys.executable} ray={ray.__version__}")'
ulimit -n 65535

VERL_DIR="/scratch/zqin30/project/repo/codename_shapley_game"
cd "${VERL_DIR}"

export HF_HOME=/scratch/zqin30/.cache/hf
export HF_HUB_CACHE="${HF_HOME}/hub"
export HYDRA_FULL_ERROR=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_LOGGING_LEVEL=INFO
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="/tmp/pv${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((24*1024*1024*1024))

steps=(${PNS_VAL_STEPS:-1500 1600})
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    step="${steps[$SLURM_ARRAY_TASK_ID]}"
else
    step="${PNS_VAL_STEP:?Set PNS_VAL_STEP when not running as an array job}"
fi

model_path="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
experiment_name="qwen2.5_7b_grpo_pns_scorer_v3_format_fix_from800"
resume_ckpt_path="${VERL_DIR}/checkpoints/pns rl/${experiment_name}/global_step_${step}"
validation_data_dir="${VERL_DIR}/outputs/validation_pns_ckpt/step_${step}_${SLURM_JOB_ID}"
gsm8k_train_path="${VERL_DIR}/data/gsm8k/train.parquet"
gsm8k_test_path="${VERL_DIR}/data/gsm8k/test.parquet"
math_train_path="${VERL_DIR}/data/math/train.parquet"
math_test_path="${VERL_DIR}/data/math/test.parquet"
train_files="['${gsm8k_train_path}','${math_train_path}']"
test_files="['${gsm8k_test_path}','${math_test_path}']"

if [[ ! -d "${resume_ckpt_path}/actor" ]]; then
    echo "Missing actor checkpoint under ${resume_ckpt_path}" >&2
    exit 2
fi

echo "=========================================="
echo "PNS checkpoint validation"
echo "job_id=${SLURM_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
echo "node=$(hostname)"
echo "model=${model_path}"
echo "resume_ckpt_path=${resume_ckpt_path}"
echo "validation_data_dir=${validation_data_dir}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

cleanup() {
    rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${train_files}" \
        data.val_files="${test_files}" \
        data.train_batch_size=8 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        reward.custom_reward_function.path="${VERL_DIR}/verl/utils/reward_score/math_format_pns_reward.py" \
        reward.custom_reward_function.name=compute_score \
        ++reward.custom_reward_function.reward_kwargs.length_soft_chars=2400 \
        ++reward.custom_reward_function.reward_kwargs.length_hard_chars=4200 \
        actor_rollout_ref.model.path="${model_path}" \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.multi_stage_wake_up=True \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name='pns rl' \
        trainer.experiment_name="${experiment_name}_val_only" \
        trainer.validation_data_dir="${validation_data_dir}" \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${resume_ckpt_path}" \
        trainer.val_only=True \
        ray_kwargs.ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-36}" \
        ++ray_kwargs.ray_init.include_dashboard=false
