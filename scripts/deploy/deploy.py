#!/usr/bin/env python
"""DocVisRAG 服务器部署脚本 — 通过 SSH 自动部署整个项目。

用法: python scripts/deploy/deploy.py
"""

import os
import sys
import time
from pathlib import Path

import paramiko
from scp import SCPClient


# ═══ 配置 ═══════════════════════════════════════════════════
HOST = "172.18.168.30"
USER = "tangbaizhen-27"
PASS = "tangbaizhen-27"
SERVER_DIR = f"/home/{USER}/DocVisRAG"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# HF 镜像配置
HF_ENV = (
    'export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"\n'
    'export HF_HUB_CACHE="$HF_HOME/hub"\n'
    'export TRANSFORMERS_CACHE="$HF_HUB_CACHE"\n'
    'export HF_ENDPOINT=https://hf-mirror.com\n'
    'export HF_HUB_DISABLE_XET=1\n'
)

EXCLUDE_PATTERNS = [
    "__pycache__", "*.pyc", ".git", ".venv", ".cache",
    "data/outputs", "data/indexes", "data/bench_runs",
    "node_modules", "*.egg-info", ".claude",
    "test.png", "开题报告.pdf", "开题报告.txt",
]


# ═══ SSH 客户端 ══════════════════════════════════════════════

class SSH:
    def __init__(self, host, user, password):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"[SSH] 连接 {user}@{host} ...")
        self.client.connect(host, username=user, password=password, timeout=15)
        print("[SSH] 已连接")

    def run(self, cmd, desc="", show_output=True):
        if desc:
            print(f"\n{'='*60}\n  {desc}\n{'='*60}")
        print(f"  $ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
        stdin, stdout, stderr = self.client.exec_command(cmd, get_pty=True)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if out and show_output:
            print(out)
        if err and "WARNING" not in err and "Warning" not in err:
            if show_output:
                print(f"  [stderr]: {err[:500]}")
        return out, err

    def upload_dir(self, local_dir, remote_dir):
        print(f"\n{'='*60}")
        print(f"  上传代码: {local_dir} → {remote_dir}")
        print(f"{'='*60}")
        with SCPClient(self.client.get_transport(), socket_timeout=30) as scp:
            scp.put(
                str(local_dir),
                remote_path=remote_dir,
                recursive=True,
            )
        print("[OK] 上传完成")

    def close(self):
        self.client.close()
        print("[SSH] 断开")


# ═══ 部署步骤 ════════════════════════════════════════════════

def step_check_env(ssh: SSH):
    """检查服务器环境"""
    ssh.run("""
echo "=== OS ===" && cat /etc/os-release | head -4 && echo "" && \
echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 && echo "" && \
echo "=== CPU Cores ===" && nproc && echo "" && \
echo "=== RAM ===" && free -h | grep ^Mem && echo "" && \
echo "=== Disk /home ===" && df -h /home && echo "" && \
echo "=== Python ===" && python3 --version 2>&1 && echo "" && \
echo "=== Git ===" && git --version 2>&1 && echo "" && \
echo "=== NVCC ===" && nvcc --version 2>&1 | tail -1 || echo "no nvcc"
""", desc="Step 1: 服务器环境检查")


def step_install_system(ssh: SSH):
    """安装系统依赖"""
    ssh.run("""
sudo apt-get update -qq 2>&1 | tail -1 && \
sudo apt-get install -y -qq --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    git curl wget ca-certificates build-essential \
    poppler-utils tesseract-ocr tesseract-ocr-chi-sim \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 2>&1 | tail -3 && \
echo "System packages installed."
""", desc="Step 2: 安装系统依赖（apt-get）")


def step_upload_code(ssh: SSH):
    """上传代码（排除不需要的文件）"""
    # 先确保目标目录存在
    ssh.run(f"mkdir -p {SERVER_DIR}")

    # 通过 scp 上传
    local = str(PROJECT_ROOT)
    # 创建临时 exclude list for scp
    ssh.upload_dir(local, SERVER_DIR)


def step_install_python(ssh: SSH):
    """创建 venv 并安装 Python 依赖"""
    ssh.run(f"""
cd {SERVER_DIR} && \
python3.10 -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1 && \
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5 && \
pip install -r requirements-base.txt 2>&1 | tail -10 && \
echo "Base packages installed."
""", desc="Step 3: 创建 venv + 安装基础依赖", show_output=True)


def step_install_visual(ssh: SSH):
    """安装 visual 检索可选依赖"""
    ssh.run(f"""
cd {SERVER_DIR} && source .venv/bin/activate && \
pip install -r requirements-visual.txt 2>&1 | tail -5 && \
echo "Visual packages installed."
""", desc="Step 4: 安装 visual 检索依赖")


def step_setup_env(ssh: SSH):
    """配置环境变量"""
    ssh.run(f"""
cd {SERVER_DIR} && \
mkdir -p .cache/huggingface && \
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'
# DocVisRAG
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
EOF
    echo "Added env vars to .bashrc"
else
    echo "Env vars already in .bashrc"
fi
""", desc="Step 5: 配置环境变量")


def step_preload_models(ssh: SSH):
    """预下载关键模型"""
    ssh.run(f"""
cd {SERVER_DIR} && source .venv/bin/activate && \
{HF_ENV}
echo "Downloading embedding model (bge-small-zh-v1.5)..." && \
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('OK: embedding model loaded')
" 2>&1
echo "" && \
echo "Pre-downloading VLM config (Qwen2.5-VL-3B-Instruct)..." && \
python -c "
from transformers import AutoConfig, AutoProcessor
cfg = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
print(f'OK: model_type={cfg.model_type}')
" 2>&1
""", desc="Step 6: 预下载模型")


def step_verify(ssh: SSH):
    """验证部署"""
    out, _ = ssh.run(f"""
cd {SERVER_DIR} && source .venv/bin/activate && \
{HF_ENV}
echo "=== Environment Check ===" && \
python scripts/env/check_env.py 2>&1 && \
echo "" && \
echo "=== Python Imports ===" && \
python -c "
from src.docvisrag.ingest import PageImage, OCRBlock, LayoutRegion, LayoutAnalyzer
from src.docvisrag.retrieve import TextIndex, HybridPageIndex
from src.docvisrag.qa import DocQAEngine, TextDocQAEngine, parse_citations
from src.docvisrag.vlm import QwenVLClient
from src.docvisrag.eval import recall_at_k, mrr, exact_match
print('All core imports: OK')
" 2>&1 && \
echo "" && \
echo "=== GPU ===" && \
python -c "
import torch
print(f'CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU count: {{torch.cuda.device_count()}}')
    for i in range(torch.cuda.device_count()):
        gb = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'GPU {{i}}: {{torch.cuda.get_device_name(i)}} ({{gb:.1f}} GB)')
else:
    print('CUDA NOT AVAILABLE - model inference will NOT work')
" 2>&1
""", desc="Step 7: 验证部署")
    return out


def main():
    print("=" * 60)
    print("  DocVisRAG 服务器自动部署")
    print(f"  目标: {USER}@{HOST}:{SERVER_DIR}")
    print("=" * 60)

    ssh = SSH(HOST, USER, PASS)

    try:
        step_check_env(ssh)
        step_install_system(ssh)
        step_upload_code(ssh)
        step_install_python(ssh)
        step_install_visual(ssh)
        step_setup_env(ssh)
        step_preload_models(ssh)
        out = step_verify(ssh)

        print("\n" + "=" * 60)
        print("  部署完成！")
        print("=" * 60)
        print(f"""
SSH 登录:    ssh {USER}@{HOST}
项目目录:    {SERVER_DIR}

使用方式:
  cd {SERVER_DIR}
  source .venv/bin/activate
  python scripts/env/check_env.py

快速验证:
  python scripts/qa/vlm_qa.py \\
    --image data/samples/test_stage1.png \\
    --question "请概括这张图片" --max-new-tokens 256

跑完整管线:
  bash scripts/loopback_test.sh
""")
    except Exception as exc:
        print(f"\n[ERROR] 部署失败: {exc}")
        return 1
    finally:
        ssh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
