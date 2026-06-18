#!/bin/bash
# ================================================================
# DocVisRAG 服务器一键安装脚本
# 在服务器上运行: bash ~/DocVisRAG/scripts/deploy/server_setup.sh
# ================================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

cd ~/DocVisRAG

# ── 1. 环境变量 ──────────────────────────────────────
info "配置环境变量..."
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 持久化到 .bashrc
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'
# DocVisRAG
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
EOF
    info "环境变量已写入 ~/.bashrc"
fi

# ── 2. 安装基础依赖 ──────────────────────────────────
info "安装 requirements-base.txt（约 5-10 分钟）..."
pip install -r requirements-base.txt --no-cache-dir 2>&1 | tail -20
info "基础依赖安装完成"

# ── 3. 安装 visual 依赖（可选） ──────────────────────
info "安装 requirements-visual.txt（约 2 分钟）..."
pip install -r requirements-visual.txt --no-cache-dir 2>&1 | tail -10
info "Visual 依赖安装完成"

# ── 4. 验证关键包 ────────────────────────────────────
info "验证关键包..."
python -c "
import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, gpus: {torch.cuda.device_count()}')
import transformers; print(f'transformers: {transformers.__version__}')
from sentence_transformers import SentenceTransformer; print('sentence-transformers: OK')
import faiss; print('faiss: OK')
import pymupdf; print('pymupdf: OK')
import gradio; print(f'gradio: {gradio.__version__}')
print('All core imports: OK')
"

# ── 5. 下载 Embedding 模型 ───────────────────────────
info "下载 embedding 模型 BAAI/bge-small-zh-v1.5（~500MB）..."
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print(f'bge-small-zh-v1.5 OK, dim={m.get_sentence_embedding_dimension()}')
"

# ── 6. 下载 Qwen2.5-VL-7B（完整模型） ───────────────
info "下载 Qwen2.5-VL-7B-Instruct（~15GB，约 10-30 分钟）..."
python -c "
import os, time
start = time.time()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
print('Loading processor...')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
print('Loading model (7B, this takes a while)...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype='auto',
    device_map='auto'
)
elapsed = time.time() - start
print(f'Qwen2.5-VL-7B-Instruct loaded in {elapsed:.0f}s')
print(f'Model device: {model.device}')
"

# ── 7. 运行环境检查 ──────────────────────────────────
info "运行环境检查..."
python scripts/env/check_env.py
python scripts/env/check_env.py --visual 2>&1 | tail -20

# ── 8. 磁盘使用 ──────────────────────────────────────
info "磁盘使用情况:"
df -h /home/tangbaizhen-27/ | tail -1
du -sh ~/DocVisRAG/.cache/ 2>/dev/null || echo "cache empty"
du -sh ~/DocVisRAG/.venv/ 2>/dev/null

info "============================================"
info "  安装全部完成！"
info "============================================"
info "快速验证:"
info "  cd ~/DocVisRAG && source .venv/bin/activate"
info "  python scripts/qa/vlm_qa.py --image data/samples/test_stage1.png --question '请概括这张图片' --model-id Qwen/Qwen2.5-VL-7B-Instruct --max-new-tokens 256"
