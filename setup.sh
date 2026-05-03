#!/bin/bash
#SBATCH --job-name=grpo_pns_official          # Descriptive job name
#SBATCH --partition=rp6b-8-gm768-c192-m2048           # Target partition
#SBATCH --nodes=1                       # Number of nodes required
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=36              # CPU cores per task
#SBATCH --mem=256G                        # Memory requirement
#SBATCH --time=12:00:00                # Maximum runtime (HH:MM:SS)
#SBATCH --output=scripts/grpo_pns_official-%j.out             # Output file (%x=job name, %j=job ID)
#SBATCH --error=scripts/grpo_pns_official-%j.err              # Error file
#SBATCH --gpus=3

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

# This is an NVIDIA CUDA job. Some cluster images export ROCm/HIP variables,
# and verl rejects having ROCR_VISIBLE_DEVICES together with CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export TORCHINDUCTOR_COMPILE_THREADS=1
export RAY_TMPDIR="/scratch/zqin30/tmp/ray_verl_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_MEMORY=$((24*1024*1024*1024))

# PNS scorer and detailed per-step logging.
export PNS_DEBERTA_CKPT="${PNS_DEBERTA_CKPT:-/scratch/zqin30/.cache/hf/hub/models--drdoggo--deberta-v3-large-pn-scorer-3class/snapshots/3371fe6c8654ec3dd0fad46f254ed60cc51b9536}"
export PNS_DEBERTA_BATCH_SIZE="${PNS_DEBERTA_BATCH_SIZE:-256}"
export PNS_DEBERTA_MAX_LENGTH="${PNS_DEBERTA_MAX_LENGTH:-512}"
export PNS_DEBERTA_DTYPE="${PNS_DEBERTA_DTYPE:-bf16}"
export PNS_DEBERTA_ATTN="${PNS_DEBERTA_ATTN:-eager}"
export PNS_DEBERTA_PAD_MULTIPLE="${PNS_DEBERTA_PAD_MULTIPLE:-8}"
export PNS_STEP_SCORE_LOG_DIR="${VERL_DIR}/outputs/pns_step_scores"
mkdir -p "${PNS_STEP_SCORE_LOG_DIR}"
export PNS_STEP_SCORE_LOG_PATH="${PNS_STEP_SCORE_LOG_DIR}/grpo_pns_${SLURM_JOB_ID}.jsonl"
export PNS_STEP_SCORE_STDOUT_SAMPLES="${PNS_STEP_SCORE_STDOUT_SAMPLES:-2}"
export PNS_STEP_SCORE_PREVIEW_CHARS="${PNS_STEP_SCORE_PREVIEW_CHARS:-200}"

model_path="${ACTOR_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
resume_ckpt_path="${RESUME_CKPT_PATH:-${VERL_DIR}/checkpoints/pns rl/qwen2.5_7b_grpo_pns_scorer_v3_format_fix_from800/global_step_900}"
rollout_data_dir="${VERL_DIR}/outputs/rollouts/${SLURM_JOB_ID}"
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
echo "GRPO + PNS official-style conda run"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "model=${model_path}"
echo "train_files=${train_files}"
echo "resume_ckpt_path=${resume_ckpt_path}"
echo "rollout_data_dir=${rollout_data_dir}"
echo "PNS_DEBERTA_CKPT=${PNS_DEBERTA_CKPT}"
echo "PNS_DEBERTA_BATCH_SIZE=${PNS_DEBERTA_BATCH_SIZE}"
echo "PNS_DEBERTA_MAX_LENGTH=${PNS_DEBERTA_MAX_LENGTH}"
echo "PNS_STEP_SCORE_LOG_PATH=${PNS_STEP_SCORE_LOG_PATH}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "=========================================="

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
        trainer.logger='["console","wandb"]' \
        trainer.project_name='pns rl' \
        trainer.experiment_name='qwen2.5_7b_grpo_pns_scorer_v3_format_fix_from800' \
        trainer.rollout_data_dir="${rollout_data_dir}" \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.max_actor_ckpt_to_keep=2 \
        trainer.max_critic_ckpt_to_keep=2 \
        trainer.test_freq=1000 \
        trainer.total_epochs=1 \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${resume_ckpt_path}" \
        ray_kwargs.ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-48}" \
        ++ray_kwargs.ray_init.include_dashboard=false \
        ++algorithm.pns_redistribution.enable=true \
        ++algorithm.pns_redistribution.alpha=0.5 \
        ++algorithm.pns_redistribution.mode=regression \
        ++algorithm.pns_redistribution.pns_values='[0.0,1.0,2.0]' \
        ++algorithm.pns_redistribution.variant=surplus \
        ++algorithm.pns_redistribution.step_segmenter=auto \
        ++algorithm.pns_redistribution.pns_scorer_path="${VERL_DIR}/verl/utils/pns_deberta_scorer.py" \
        ++algorithm.pns_redistribution.pns_scorer_name=score_steps \
        ++algorithm.pns_redistribution.scorer_ray_actor=true \
        ++algorithm.pns_redistribution.scorer_num_gpus=1 \
        ++algorithm.pns_redistribution.scorer_num_cpus=4

exit_code=$?
rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
exit "${exit_code}"