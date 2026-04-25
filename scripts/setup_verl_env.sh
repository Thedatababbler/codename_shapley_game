#!/bin/bash
#SBATCH -p aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=setup_verl
#SBATCH --output=scripts/setup_verl-%j.log
#SBATCH --error=scripts/setup_verl-%j.log

set -euo pipefail

# ============================================================
# verl 训练环境 (SGLang backend, Python 3.12, torch 2.8 + cu126)
#
# 重要: 集群 driver 版本是 12.7, 所以 torch 必须是 cu126 runtime
# (cu128 会报 cudaErrorDevicesUnavailable).
#
# 组合参考 verl 官方 install_vllm_sglang_mcore.sh:
#   - torch 2.8.0 + cu126
#   - sglang[all] == 0.5.2
#   - flash-attn 2.8.1 预编译 wheel (cu12torch2.8 cxx11abiFALSE cp312)
#
# 策略: 在本地 /tmp (SSD) 构建 venv，然后一次性 cp -a 到 RDS
# 教训: 直接装 RDS 会非常慢 (4 个小包就要 13 分钟，torch 上不封顶)
#       所以必须走 /tmp 构建，最后 cp。
# 关键: 整个过程不要从外部杀任何进程（包括 rm），否则会破坏 cp 的目标路径状态
# ============================================================

VERL_DIR="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/verl"
FINAL_VENV="/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/envs/verl_sglang"
LOCAL_VENV="/tmp/verl_sglang_${SLURM_JOB_ID}"

# UV cache 放在本地 /tmp，避免 RDS 上历史缓存权限损坏（archive-v0 被 root 占用）
# 每次 job 重新下载但 /tmp 是 SSD，总体反而更快
export UV_CACHE_DIR="/tmp/uv_cache_${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export HF_HOME=/mnt/rds/VipinRDS/VipinRDS/users/yxs1432/.cache/

# --- toolchain (Py 3.12 对应 GCCcore 13.3.0) ---
# 直接 module load Python 会自动带上所有运行依赖:
#   zlib, binutils, bzip2, ncurses, libreadline, Tcl, SQLite, XZ,
#   libffi, OpenSSL, GCCcore
# 这样就不用一个个手动加 LD_LIBRARY_PATH
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load Python/3.12.3-GCCcore-13.3.0
# CUDA 12.6 nvcc 用于 flash-attn 源码编译，且与 torch cu126 runtime 一致
module load CUDA/12.6.0

PYTHON_ROOT=/usr/local/easybuild_allnodes/software/Python/3.12.3-GCCcore-13.3.0
GCC_ROOT=/usr/local/easybuild_allnodes/software/GCCcore/13.3.0
PYTHON_INCLUDE="${PYTHON_ROOT}/include/python3.12"

export CC="${GCC_ROOT}/bin/gcc"
export CXX="${GCC_ROOT}/bin/g++"
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"
export C_INCLUDE_PATH="${PYTHON_INCLUDE}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"

# --- CUDA (不用 module load CUDA, 因为只装 pre-built wheel) ---
# pre-built flash-attn / torch wheels 自带 cu12.6 libs, 不需要外部 nvcc

echo "============================================"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "gcc:        $(gcc --version | head -1)"
echo "g++:        $(g++ --version | head -1)"
echo "as:         $(as --version | head -1)"
echo "python:     $(python3 --version) at $(which python3)"
echo "Date:       $(date)"
echo "============================================"

# ─── Step 0a: 清理 RDS 上的旧/损坏目录 (先同步做完，避免 cp 时发现还有残留) ───
echo ""
echo ">>> [0a/6] Cleaning old RDS venv: ${FINAL_VENV}"
if [ -e "${FINAL_VENV}" ]; then
  echo "    removing existing venv (RDS 小文件删除可能慢, 请耐心等)..."
  time rm -rf "${FINAL_VENV}"
  echo "    removed."
fi
# 确认父目录存在、干净
mkdir -p "$(dirname "${FINAL_VENV}")"
# 再次校验目标真的不存在
[ -e "${FINAL_VENV}" ] && { echo "ERROR: ${FINAL_VENV} still exists after rm!"; exit 1; }

# ─── Step 0b: 在 /tmp (SSD) 创建 venv ───
echo ""
echo ">>> [0b/6] Creating venv at ${LOCAL_VENV} (python 3.12, local SSD)"
rm -rf "${LOCAL_VENV}" 2>/dev/null || true
uv venv "${LOCAL_VENV}" --python "${PYTHON_ROOT}/bin/python3.12"
source "${LOCAL_VENV}/bin/activate"
export PATH="${LOCAL_VENV}/bin:${PATH}"
uv pip install pip setuptools wheel
echo "    Python: $(which python) ($(python --version))"

# ─── Step 1: PyTorch 2.8.0 + cu126 (driver 12.7 can run this runtime) ───
echo ""
echo ">>> [1/6] Installing torch 2.8.0 + cu126 (on /tmp)"
uv pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0
python -c "import torch; print(f'    torch={torch.__version__}, cuda_build={torch.version.cuda}')"

# ─── Step 2: SGLang 0.5.2 (same version as verl官方脚本) ───
echo ""
echo ">>> [2/6] Installing sglang[all]==0.5.2 (on /tmp)"
uv pip install --no-cache-dir "sglang[all]==0.5.2" torch-memory-saver
python -c "import sglang; print(f'    sglang={sglang.__version__}')"

# ─── Step 3: flash-attn 2.8.1 (源码编译, 因为集群 glibc 2.28 无法用官方 wheel) ───
echo ""
echo ">>> [3/6] Compiling flash-attn 2.8.1 from source (~30 min, on /tmp)"
# 先装编译依赖
uv pip install --no-cache-dir packaging ninja psutil
# CUDA 环境变量
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "    CUDA_HOME=${CUDA_HOME}  nvcc=$(nvcc --version | tail -1)"
# A100 (sm_80) 目标架构；同时保留 sm_90 兼容性
export TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"
# 限制并行避免 OOM（flash-attn 单 TU 编译可到 10+GB）
export MAX_JOBS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# 从 git 源码编译
FA_SRC_DIR="/tmp/flash_attn_src_${SLURM_JOB_ID}"
rm -rf "${FA_SRC_DIR}"
git clone --depth 1 --branch v2.8.1 https://github.com/Dao-AILab/flash-attention.git "${FA_SRC_DIR}"
cd "${FA_SRC_DIR}"
# 用 pip (不是 uv) 避免 isolated-build 丢失 MAX_JOBS 等环境变量
python -m pip install --no-build-isolation --no-cache-dir -v .
cd -
rm -rf "${FA_SRC_DIR}"
python -c "import flash_attn; print(f'    flash_attn={flash_attn.__version__}')"

# ─── Step 4: verl 其它依赖 + wandb ───
echo ""
echo ">>> [4/6] Installing verl deps + wandb (on /tmp)"
uv pip install --no-cache-dir \
    "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc dill pybind11 \
    liger-kernel sentencepiece protobuf mathruler \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
    tensorboard packaging uvicorn omegaconf cachetools Pillow pyyaml \
    wandb
echo "    done"

# ─── Step 5: 安装 verl (editable, no-deps, 指向 RDS 源码) ───
echo ""
echo ">>> [5/6] Installing verl from source (on /tmp venv, RDS source)"
cd "${VERL_DIR}"
uv pip install --no-cache-dir --no-deps -e .
python -c "import verl; print('    verl installed')"

# ─── Step 6: 将 /tmp venv 拷贝到 RDS (一次性大拷贝, 期间不要干预) ───
echo ""
echo ">>> [6/6] Copying venv from /tmp to RDS: ${FINAL_VENV}"
echo "    (这步耗时 ~15-25 分钟, 请不要 scancel / kill 任何进程)"
time cp -a "${LOCAL_VENV}" "${FINAL_VENV}"

# 修正 venv 内脚本的路径引用 (pyvenv.cfg + 所有 bin 脚本的 shebang/硬编码路径)
echo "    fixing paths in ${FINAL_VENV}..."
sed -i "s|${LOCAL_VENV}|${FINAL_VENV}|g" "${FINAL_VENV}/pyvenv.cfg" || true
for f in "${FINAL_VENV}/bin/"*; do
    if file "$f" 2>/dev/null | grep -q "text"; then
        sed -i "s|${LOCAL_VENV}|${FINAL_VENV}|g" "$f" 2>/dev/null || true
    fi
done

# 用新路径的 venv 重新 editable-install verl (更新 .pth 指向)
source "${FINAL_VENV}/bin/activate"
cd "${VERL_DIR}"
uv pip install --no-cache-dir --no-deps -e .

# 验证
echo ""
echo "    === 验证 (使用 RDS venv) ==="
python -c "import torch; print(f'    torch={torch.__version__}, cuda={torch.version.cuda}')"
python -c "import sglang; print(f'    sglang={sglang.__version__}')"
python -c "import flash_attn; print(f'    flash_attn={flash_attn.__version__}')"
python -c "from verl.trainer.main_ppo import main; from verl.experimental.agent_loop import AgentLoopManager; print('    verl OK')"

# 清理 /tmp 资源
rm -rf "${LOCAL_VENV}" 2>/dev/null || true
rm -rf "${UV_CACHE_DIR}" 2>/dev/null || true

echo ""
echo "============================================"
echo "Environment setup complete!"
echo "Activate with: source ${FINAL_VENV}/bin/activate"
echo "Total packages: $(ls "${FINAL_VENV}/lib/python3.12/site-packages/" 2>/dev/null | grep -cE 'dist-info')"
echo "============================================"
