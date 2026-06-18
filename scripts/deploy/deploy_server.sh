#!/usr/bin/env bash
# ============================================================
# DocVisRAG 服务器一键部署脚本
# 用法：在 VSCode 终端 / PowerShell 中执行此脚本
#   bash scripts/deploy/deploy_server.sh
# ============================================================
set -e

# ── 配置 ──────────────────────────────────────────────
SERVER_USER="tangbaizhen-27"
SERVER_IP="172.18.168.30"
SERVER_PASS="tangbaizhen-27"
SERVER_DIR="/home/${SERVER_USER}/DocVisRAG"

# ── 颜色 ──────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 0: 测试 SSH ──────────────────────────────────
info "Step 0: 测试 SSH 连接..."
if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o PasswordAuthentication=yes "${SERVER_USER}@${SERVER_IP}" \
    "echo SSH_OK" 2>/dev/null; then
    warn "SSH 免密登录未配置，请手动输入密码（仅需一次）"
fi

# ── Step 1: 服务器环境检查 ────────────────────────────
info "Step 1: 检查服务器环境..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_CHECK'
echo "=== OS ==="
cat /etc/os-release | head -4
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "NO GPU"
echo ""
echo "=== CPU ==="
echo "Cores: $(nproc)"
echo ""
echo "=== RAM ==="
free -h | grep ^Mem
echo ""
echo "=== Disk ==="
df -h /home/
echo ""
echo "=== Python ==="
python3 --version 2>&1 || python --version 2>&1 || echo "NO PYTHON"
echo ""
echo "=== Git ==="
git --version 2>&1 || echo "NO GIT"
echo ""
echo "=== CUDA ==="
nvcc --version 2>&1 | tail -1 || echo "NO NVCC"
REMOTE_CHECK

# ── Step 2: 安装系统依赖 ──────────────────────────────
info "Step 2: 安装系统依赖..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_INSTALL'
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    git curl wget ca-certificates build-essential \
    poppler-utils tesseract-ocr tesseract-ocr-chi-sim \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
echo "System packages installed."
REMOTE_INSTALL

# ── Step 3: 上传项目代码 ──────────────────────────────
info "Step 3: 上传项目代码..."
if git rev-parse --show-toplevel >/dev/null 2>&1; then
    PROJECT_ROOT=$(git rev-parse --show-toplevel)
else
    PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
fi

# 检查服务器是否已有代码
if ssh "${SERVER_USER}@${SERVER_IP}" "[ -d ${SERVER_DIR}/.git ]" 2>/dev/null; then
    info "服务器上已有 git 仓库，执行 git pull..."
    ssh "${SERVER_USER}@${SERVER_IP}" "cd ${SERVER_DIR} && git pull"
else
    info "首次部署，通过 rsync 上传代码..."
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude '.cache' \
        --exclude 'data/outputs' \
        --exclude 'data/indexes' \
        --exclude 'data/bench_runs' \
        "${PROJECT_ROOT}/" \
        "${SERVER_USER}@${SERVER_IP}:${SERVER_DIR}/"
fi

# ── Step 4: 创建虚拟环境并安装依赖 ────────────────────
info "Step 4: 创建 Python 虚拟环境..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_VENV'
set -e
cd ~/DocVisRAG

# 创建 venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# 安装 PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements-base.txt

echo "Base dependencies installed."
REMOTE_VENV

# ── Step 5: 安装可选 visual 依赖 ──────────────────────
info "Step 5: 安装 visual 检索依赖（可选）..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_VISUAL'
set -e
cd ~/DocVisRAG
source .venv/bin/activate
pip install -r requirements-visual.txt
echo "Visual dependencies installed."
REMOTE_VISUAL

# ── Step 6: 配置环境变量 ──────────────────────────────
info "Step 6: 配置环境变量..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_ENV'
cd ~/DocVisRAG

# 添加环境变量到 .bashrc（如果不存在）
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# DocVisRAG 环境变量
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
EOF
fi

# 当前 session 生效
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p "$HF_HOME"
echo "Environment configured."
REMOTE_ENV

# ── Step 7: 预下载模型（可选，首次较慢） ──────────────
info "Step 7: 预下载模型（这可能需要几分钟）..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_MODEL'
set -e
cd ~/DocVisRAG
source .venv/bin/activate

export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p "$HF_HOME"

# 预下载 embedding 模型（小，几秒）
echo "Downloading embedding model..."
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('Embedding model loaded.')
"

# 预下载 VLM 模型（大，可能需要几分钟）
echo "Downloading VLM model (this may take a while)..."
python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoProcessor, AutoConfig
# 先只下载配置和 processor，不加载权重
cfg = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
print('VLM config loaded. Model type:', cfg.model_type)
# 可选：预下载完整模型（取消注释下一行）
# from transformers import Qwen2_5_VLForConditionalGeneration
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', device_map='cpu')
"

echo "Model pre-download completed."
REMOTE_MODEL

# ── Step 8: 验证部署 ──────────────────────────────────
info "Step 8: 验证部署..."
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE_VERIFY'
set -e
cd ~/DocVisRAG
source .venv/bin/activate

export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com

echo "=== Environment Check ==="
python scripts/env/check_env.py

echo ""
echo "=== Python Imports ==="
python -c "
from src.docvisrag.ingest import PageImage, OCRBlock, LayoutRegion, LayoutAnalyzer
from src.docvisrag.retrieve import TextIndex, HybridPageIndex, VisualPageIndex, reciprocal_rank_fusion
from src.docvisrag.qa import DocQAEngine, TextDocQAEngine, parse_citations
from src.docvisrag.vlm import QwenVLClient
from src.docvisrag.eval import recall_at_k, mrr, exact_match, token_f1, simple_anls
print('All imports OK')
"

echo ""
echo "=== GPU Check ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== Verification Complete ==="
REMOTE_VERIFY

# ── 完成 ──────────────────────────────────────────────
info ""
info "============================================"
info "  部署完成！"
info "============================================"
info ""
info "SSH 登录:  ssh ${SERVER_USER}@${SERVER_IP}"
info "项目目录:  ${SERVER_DIR}"
info ""
info "登录后运行项目:"
info "  cd ~/DocVisRAG"
info "  source .venv/bin/activate"
info "  python scripts/env/check_env.py"
info ""
info "快速验证:"
info "  python scripts/qa/vlm_qa.py --image data/samples/test_stage1.png --question '请概括这张图片'"
info "============================================"
