#!/bin/bash
#SBATCH --job-name=rloo_native_baseline
#SBATCH --partition=rp6b-8-gm768-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=scripts/rloo_native_baseline-%j.out
#SBATCH --error=scripts/rloo_native_baseline-%j.err
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
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be exported or passed via sbatch --export}"
export HYDRA_FULL_ERROR=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_LOGGING_LEVEL=INFO
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="/scratch/zqin30/tmp/ray_verl_rloo_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((24*1024*1024*1024))

model_path="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
rollout_data_dir="${VERL_DIR}/outputs/rollouts_rloo/${SLURM_JOB_ID}"
validation_data_dir="${VERL_DIR}/outputs/validation_rloo/${SLURM_JOB_ID}"
gsm8k_train_path="${VERL_DIR}/data/gsm8k/train.parquet"
gsm8k_test_path="${VERL_DIR}/data/gsm8k/test.parquet"
math_train_path="${VERL_DIR}/data/math/train.parquet"
math_test_path="${VERL_DIR}/data/math/test.parquet"
train_files="['${gsm8k_train_path}','${math_train_path}']"
test_files="['${gsm8k_test_path}','${math_test_path}']"

if [[ ! -f "${gsm8k_train_path}" || ! -f "${math_train_path}" ]]; then
    "${PYTHON_BIN}" examples/data_preprocess/gsm8k.py --local_save_dir "${VERL_DIR}/data/gsm8k"
    "${PYTHON_BIN}" examples/data_preprocess/math_dataset.py --local_save_dir "${VERL_DIR}/data/math"
fi

echo "=========================================="
echo "Native RLOO baseline conda run"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "model=${model_path}"
echo "train_files=${train_files}"
echo "rollout_data_dir=${rollout_data_dir}"
echo "validation_data_dir=${validation_data_dir}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
        algorithm.adv_estimator=rloo \
        data.train_files="${train_files}" \
        data.val_files="${test_files}" \
        data.train_batch_size=8 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
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
        trainer.logger='["console","wandb"]' \
        trainer.project_name='pns rl' \
        trainer.experiment_name='qwen2.5_7b_rloo_native_baseline_2gpu' \
        trainer.rollout_data_dir="${rollout_data_dir}" \
        trainer.validation_data_dir="${validation_data_dir}" \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.max_actor_ckpt_to_keep=2 \
        trainer.max_critic_ckpt_to_keep=2 \
        trainer.test_freq=200 \
        trainer.total_epochs=1 \
        trainer.resume_mode=disable \
        ray_kwargs.ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-48}" \
        ++ray_kwargs.ray_init.include_dashboard=false

exit_code=$?
rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
exit "${exit_code}"
