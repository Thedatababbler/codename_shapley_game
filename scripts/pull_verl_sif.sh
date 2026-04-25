#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=pull_verl_sif
#SBATCH --output=scripts/pull_verl_sif-%j.log
#SBATCH --error=scripts/pull_verl_sif-%j.log

# ============================================================
# 一次性拉 verl 官方 docker 镜像 (sglang 0.5.9 backend) 并转成 .sif
# 存到 RDS, 以后训练直接 singularity exec 用, 永不再装环境
# 镜像: verlai/verl:sgl059.latest  (20 GB compressed)
# ============================================================

set -euo pipefail

module purge 2>/dev/null || true
module load singularity/4.3.4

SIF_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs"
SIF_PATH="${SIF_DIR}/verl_sgl059.sif"
mkdir -p "${SIF_DIR}"

# Singularity 自己的临时 + 缓存目录 - 用 /tmp SSD 避免 RDS I/O
export SINGULARITY_TMPDIR=/tmp/sing_tmp_${SLURM_JOB_ID}
export SINGULARITY_CACHEDIR=/tmp/sing_cache_${SLURM_JOB_ID}
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

echo "============================================"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "singularity: $(singularity --version)"
echo "Target:     ${SIF_PATH}"
echo "Tmp:        ${SINGULARITY_TMPDIR}"
echo "============================================"

# 如果旧 sif 存在, 删掉重来
if [ -f "${SIF_PATH}" ]; then
    echo ">>> Removing old sif..."
    rm -f "${SIF_PATH}"
fi

echo ""
echo ">>> Pulling docker image (verlai/verl:sgl059.latest, ~20 GB compressed)..."
echo "    (预计 20-60 分钟, 取决于网速和 RDS 写入速度)"

time singularity pull --force "${SIF_PATH}" docker://verlai/verl:sgl059.latest

echo ""
echo ">>> Checking sif..."
ls -lh "${SIF_PATH}"
singularity inspect "${SIF_PATH}" | head -20 || true

echo ""
echo ">>> Smoke test inside container..."
singularity exec --nv "${SIF_PATH}" bash -c '
  which python
  python --version
  python -c "import torch; print(f\"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_avail={torch.cuda.is_available()}\")"
  python -c "import sglang; print(f\"sglang={sglang.__version__}\")"
  python -c "import flash_attn; print(f\"flash_attn={flash_attn.__version__}\")"
  python -c "import verl; print(f\"verl (built-in) imported OK\")" || echo "verl not installed in image, will use bind-mount"
'

# cleanup
rm -rf "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}" 2>/dev/null || true

echo ""
echo "============================================"
echo "DONE. .sif at ${SIF_PATH}"
echo "============================================"
