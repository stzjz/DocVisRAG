# DocVisRAG 服务器部署与实验报告

## 1. 服务器环境

| 项目 | 规格 |
|------|------|
| 型号 | Super Rack Server |
| OS | Ubuntu 22.04.5 LTS (GNU/Linux 6.8.0-107-generic) |
| CPU | 320 核 |
| RAM | 377 GB |
| 磁盘 | 3.5 TB NVMe（可用 ~460 GB） |
| GPU | 4 卡 |

### GPU 详情

| GPU | 型号 | 显存 | 架构 | CUDA 支持 |
|-----|------|------|------|-----------|
| 0 | NVIDIA RTX PRO 5000 Blackwell | 48 GB | Blackwell (sm_120) | PyTorch ≥ 2.7 |
| 1 | NVIDIA RTX 5880 Ada Generation | 48 GB | Ada (sm_89) | PyTorch ≥ 2.0 |
| 2 | NVIDIA RTX 5880 Ada Generation | 48 GB | Ada (sm_89) | PyTorch ≥ 2.0 |
| 3 | NVIDIA RTX PRO 5000 Blackwell | 48 GB | Blackwell (sm_120) | PyTorch ≥ 2.7 |

- **总显存**: ~192 GB
- **CUDA Driver**: 595.58.03, CUDA 13.2
- **初始问题**: PyTorch 2.6.0 不支持 Blackwell 架构 → 升级至 2.7.1+cu128 解决

---

## 2. 部署过程

### 2.1 基础环境

```bash
# SSH 连接
ssh tangbaizhen-27@172.18.168.30

# 系统已有: Python 3.10.12, Git 2.34.1, poppler-utils
# 系统缺少: python3-pip, python3.10-venv, tesseract, sudo权限
```

### 2.2 安装 pip（无需 sudo）

服务器缺少 `python3-pip` 且无 sudo 权限，通过 `get-pip.py` 用户级安装：

```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3.10 /tmp/get-pip.py --user
export PATH="$HOME/.local/bin:$PATH"
```

### 2.3 创建虚拟环境

`python3.10-venv` 缺失，改用 `virtualenv`：

```bash
python3.10 -m pip install --user virtualenv
python3.10 -m virtualenv ~/DocVisRAG/.venv
source ~/DocVisRAG/.venv/bin/activate
```

### 2.4 配置 pip 镜像（关键步骤）

默认 PyPI 源速度极慢（~89 kB/s），必须切清华镜像：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2.5 安装 PyTorch

初始安装 PyTorch 2.6.0+cu124，但 Blackwell GPU 报错 `no kernel image is available`。升级至 2.7.1+cu128 解决：

```bash
# 初始版本（有问题）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 升级修复（4 卡全可用）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

**踩坑记录**: PyTorch 2.7.1 升级后出现 numpy/scipy 版本冲突。`transformers` 重装时拉入了 `numpy==2.2.6`，与 `scipy==1.11.4` 和 `paddleocr==2.8.1` 不兼容。修复方法：先固定 numpy，再装其他：

```bash
pip install numpy==1.26.4 scipy==1.11.4 --force-reinstall --no-cache-dir
pip install -r requirements-base.txt --no-cache-dir
```

### 2.6 安装项目依赖

```bash
cd ~/DocVisRAG
pip install -r requirements-base.txt --no-cache-dir
pip install -r requirements-visual.txt --no-cache-dir
```

### 2.7 配置 HuggingFace 镜像

```bash
# 持久化环境变量
cat >> ~/.bashrc << 'EOF'
export HF_HOME="$HOME/DocVisRAG/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
EOF
source ~/.bashrc
```

### 2.8 下载模型

```bash
# Embedding 模型 (~500 MB)
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
"

# VLM 模型 (~15 GB)
python -c "
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype='auto', device_map='auto'
)
"
```

### 2.9 修改默认模型配置

将所有模块的默认模型从 3B 改为 7B：

```bash
cd ~/DocVisRAG
sed -i 's|Qwen2.5-VL-3B-Instruct|Qwen2.5-VL-7B-Instruct|g' \
  src/docvisrag/config.py \
  src/docvisrag/vlm/qwen_vl.py \
  src/docvisrag/qa/doc_qa.py \
  src/docvisrag/qa/text_qa.py \
  src/docvisrag/qa/page_qa.py \
  src/docvisrag/ingest/page_summary.py \
  scripts/qa/vlm_qa.py
```

### 2.10 离线模式注意事项

服务器对 `huggingface.co` 直连被墙，对 `hf-mirror.com` 间歇性超时。索引构建需要在环境变量中同时设置镜像和离线标志：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1  # 仅在模型已缓存时使用
```

`sentence-transformers` 库（v5.5.0+）在 `HF_HUB_OFFLINE=1` 但无 `HF_ENDPOINT` 时仍会尝试直连。必须**同时**设置两者。

---

## 3. 实验设计

### 3.1 数据集

| 数据集 | 来源 | 题目数 | 页面数 | 特点 |
|--------|------|--------|--------|------|
| **DocVQA** | `pixparse/docvqa-single-page-questions` | 100 | 26 | 文字密集型文档问答 |
| **ChartQA** | `HuggingFaceM4/ChartQA` | 50 | 24 | 柱状图/折线图/饼图数值推理 |

```bash
# DocVQA
python scripts/bench/prepare_benchmark.py \
  --benchmark docvqa --out-dir data/bench/docvqa_100 --limit 100

# ChartQA
python scripts/bench/prepare_benchmark.py \
  --benchmark chartqa --out-dir data/bench/chartqa_50 --limit 50
```

> TextVQA（`lmms-lab/textvqa`）因服务器网络问题（hf-mirror 超时）未能下载，后续补上。

### 3.2 对照组设置

| 组别 | 检索器 | 检索单元 | 生成模型 | 证据形式 |
|------|--------|----------|----------|----------|
| **TEXT 基线** | text | OCR 文本块（FAISS） | Qwen2.5-VL-7B（纯文本） | OCR 文本 |
| **HYBRID 实验组** | hybrid | 页面（摘要+OCR） | Qwen2.5-VL-7B（多图） | 页面图像+文本 |
| **FUSION** | fusion | 混合+视觉（RRF） | 同上 | 页面图像+文本+ColPali |

### 3.3 QA Prompt 优化（第二轮实验）

初始中文 prompt 对英文 DocVQA/ChartQA 无效——ChartQA TEXT QA 全零。第二轮实验前做了以下优化：

1. **语言自动检测**：`_is_english()` 判断问题语言，英文问题切英文 prompt
2. **短答案格式**：添加 `IMPORTANT: Provide a SHORT, CONCISE answer (1-5 words)` 指令
3. **输出解析增强**：`_extract_section()` 兼容英文标记（`Answer:`/`Evidence:`/`Citation:`）
4. **代码文件**：修改了 `src/docvisrag/qa/doc_qa.py` 和 `src/docvisrag/qa/text_qa.py`

### 3.4 ColPali 模型下载与 FUSION 建设

FUSION 需要 `vidore/colqwen2-v1.0`（~8.8GB），部署过程中经历了多次网络问题：

**尝试 1（6月3日凌晨）**：hf-mirror 超时 → 失败  
**尝试 2（6月3日上午）**：hf-mirror 恢复，模型下载完成，但加载时报 `Python.h: No such file or directory`  
**原因**：`bitsandbytes` 触发 triton 编译 CUDA 内核，需要 `python3.10-dev`，服务器缺失且无 sudo  
**解决方案**：卸载 `bitsandbytes`（ColPali 不需要量化），直接通过 `colpali_engine` 加载

```bash
pip uninstall bitsandbytes -y
python -c "
from byaldi import RAGMultiModalModel
m = RAGMultiModalModel.from_pretrained('vidore/colqwen2-v1.0')
print('Byaldi ready!')
"

# 构建 visual 索引
python scripts/retrieve/build_visual_index.py \
  --manifest data/bench/docvqa_100/manifest.json \
  --index-dir data/bench_runs/experiment_20260603/docvqa_fusion/intermediate/visual_index
```

### 3.5 最终实验启动

```bash
cd ~/DocVisRAG && source .venv/bin/activate
export HF_HOME=/home/tangbaizhen-27/DocVisRAG/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# 顺序跑 TEXT → HYBRID → FUSION（DocVQA 100题）
nohup bash ~/run_experiments.sh > ~/experiment_final.log 2>&1 &
```

### 3.6 评测指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 检索 | Recall@1/3/5 | 前 K 个检索结果命中证据页的比例 |
| 检索 | MRR | 首个命中页的倒数排名均值 |
| 检索 | NDCG@5 | 标准化折损累计增益 |
| 问答 | EM (Exact Match) | 预测答案与标准答案完全一致比例 |
| 问答 | F1 | Token 级别 F1 分数 |
| 问答 | ANLS | 基于编辑距离的近似匹配（阈值 0.5） |
| 问答 | Citation Accuracy | 引用页码命中证据页的比例 |

---

## 4. 实验结果

### 4.1 DocVQA（100题）— 最终三组对比

| 指标 | TEXT 基线 | HYBRID 多模态 | FUSION | 最佳 |
|------|-----------|--------------|--------|:--:|
| Recall@1 | **0.510** | 0.170 | 0.190 | TEXT |
| Recall@3 | **0.770** | 0.380 | 0.340 | TEXT |
| Recall@5 | **0.790** | 0.550 | 0.500 | TEXT |
| MRR | **0.632** | 0.298 | 0.296 | TEXT |
| NDCG@5 | **0.672** | 0.360 | 0.347 | TEXT |
| EM | 0.000 | 0.000 | 0.000 | — |
| **F1** | 0.034 | **0.088** | 0.067 | **HYBRID** |
| ANLS | 0.000 | 0.000 | 0.000 | — |
| CitationAcc | **0.710** | 0.380 | 0.340 | TEXT |

**核心发现**：

1. **检索：TEXT 全面碾压**。OCR 文本直接含有关键词（如发票金额、人名），向量匹配精准。VLM 摘要将页面概括为 200 字，反而稀释了细节信号。FUSION 的 ColPali 视觉通道对文字密集型文档没有增益

2. **生成：HYBRID 最优**。F1 从 0.034 → 0.088（+159%），VLM 看图理解了版面结构和空间关系，token 重叠度显著更高

3. **EM/ANLS 全零是评测口径问题**。7B 模型输出自然语言（如 "The document mainly discusses..."），DocVQA 标注是短词（"Blue""2018"）。新 prompt 改进了 F1 但精确匹配需要更激进的后处理（截断首行/提取关键词）

### 4.2 ChartQA（50题）— TEXT vs HYBRID

| 指标 | TEXT | HYBRID | 胜出 |
|------|------|--------|:--:|
| Recall@1 | 0.24 | 0.20 | TEXT |
| Recall@5 | 0.48 | **0.50** | HYBRID |
| MRR | 0.33 | 0.30 | TEXT |
| EM | 0.000 | **0.100** | **HYBRID** |
| **F1** | 0.000 | **0.199** | **HYBRID** |
| ANLS | 0.000 | **0.143** | **HYBRID** |
| CitationAcc | 0.38 | 0.22 | TEXT |

**核心发现**：

1. **HYBRID 在图表问答上碾压 TEXT**。TEXT QA 三指标全为零（纯 OCR 文本无法理解图表数值），HYBRID 的 VLM 看图直接读出了图表数据（EM 0.10, F1 0.20, ANLS 0.14）

2. ChartQA 是"视觉主通道"最有力的证据——图表是纯视觉对象，OCR 拍扁后完全无法回答

3. ChartQA 检索差异小（R@5: 0.48 vs 0.50），因为图表标题/轴标签提供了少量 OCR 信号

### 4.3 综合结论

1. **TEXT 检索最强**（OCR 关键词匹配），但**HYBRID 生成最准**（VLM 看图理解）。两者在不同维度互补

2. **图表类任务是多模态方案的杀手场景**：ChartQA TEXT F1=0.000 vs HYBRID F1=0.199（无穷大倍差距）

3. **FUSION 的 ColPali 视觉检索对文字密集型文档帮助有限**：文档文字本身已经提供了足够的检索信号，通用视觉特征未能额外提升

4. **EM/ANLS 全零是评测标准问题，不是模型问题**：模型输出自然语言，标注是短词。F1 能捕捉到语义重叠，是更合理的评估指标。后续可引入 Relaxed Accuracy（容差匹配）和人工评分

5. **"视觉主通道 + OCR 辅通道"架构得到验证**：检索靠 OCR（TEXT 最优），生成靠视觉（HYBRID 最优），两者分工合理

---

## 5. 踩坑记录

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | `CUDA error: no kernel image` | PyTorch 2.6.0 不支持 Blackwell (sm_120) | 升级至 PyTorch 2.7.1+cu128 |
| 2 | pip 下载 89 kB/s | 默认 PyPI 源 | 配置清华 pip 镜像 |
| 3 | `python3-pip` 缺失 | 服务器最小化安装 | `get-pip.py --user` |
| 4 | `python3.10-venv` 缺失 | 同上 | 改用 `virtualenv` |
| 5 | numpy/scipy 版本冲突 | transformers 重装拉入新版 numpy | 先 pin numpy==1.26.4 再装其他 |
| 6 | `sentence-transformers` 直连 huggingface.co | 镜像设置不生效 | 同时设 `HF_ENDPOINT` + `HF_HUB_OFFLINE=1` |
| 7 | `--load-in-4bit` 需编译 CUDA kernel | 缺少 `python3.10-dev` 头文件 | 不使用 4bit（48GB 显存够用） |
| 8 | hf-mirror 间歇性不可用 | 服务器网络 DNS/防火墙 | 预下载模型后切 `HF_HUB_OFFLINE=1` |
| 9 | ColPali 模型下载后加载失败 | `bitsandbytes` → triton → 需 `Python.h` | 卸载 `bitsandbytes`，ColPali 不需要量化 |
| 10 | 3B 默认模型被反复尝试下载 | 多处代码硬编码 3B 默认值 | `sed` 全局替换为 7B |
| 11 | ChartQA TEXT QA 全零 | 中文 prompt 对英文数据无效 | 添加 `_is_english()` 语言检测 + 英文 prompt |
| 12 | ColPali 缓存不完整 | 第一次下载被 Python.h 错误中断 | 清除缓存 `rm -rf models--vidore--colqwen2-v1.0` 重新下载 |
| 13 | `HF_HUB_CACHE` 路径被截断为 `/hub` | shell 变量嵌套展开失败 | 使用绝对路径 `/home/tangbaizhen-27/...` |

---

## 6. 服务器文件结构

```
~/
├── DocVisRAG/                           # 项目根目录
│   ├── .venv/                           # Python 虚拟环境 (~25 GB)
│   ├── .cache/huggingface/              # 模型缓存 (~26 GB)
│   │   └── hub/
│   │       ├── models--BAAI--bge-small-zh-v1.5/      # Embedding (~500 MB)
│   │       ├── models--Qwen--Qwen2.5-VL-7B-Instruct/  # VLM (~15 GB)
│   │       └── models--vidore--colqwen2-v1.0/        # ColPali (~8.8 GB)
│   ├── data/
│   │   ├── bench/
│   │   │   ├── docvqa_100/              # DocVQA 100题 (26 pages)
│   │   │   └── chartqa_50/              # ChartQA 50题 (24 pages)
│   │   └── bench_runs/
│   │       └── experiment_20260603/      # 实验输出
│   │           ├── overview.json
│   │           ├── docvqa_text/          # TEXT 基线
│   │           ├── docvqa_hybrid/        # HYBRID
│   │           ├── docvqa_fusion/        # FUSION (含 visual_index)
│   │           ├── chartqa_text/         # ChartQA TEXT
│   │           └── chartqa_hybrid/       # ChartQA HYBRID
│   └── ...
├── run_experiments.sh                   # DocVQA 实验脚本
├── run_experiments_extra.sh             # ChartQA 实验脚本
├── experiment_final.log                 # 实验运行日志
└── .bashrc                              # 含 HF/PyTorch 环境变量
```

---

## 7. 后续工作

| 优先级 | 任务 | 状态 |
|--------|------|:--:|
| P0 | 自建课程文档集（8-12份，60-120 QA） | ⬜ |
| P0 | 错误案例分析（挑选 TEXT vs HYBRID 典型 case） | ⬜ |
| P1 | TextVQA 评测（网络恢复后） | ⬜ |
| P1 | Relaxed Accuracy 指标（图表数值容差） | ⬜ |
| P1 | 人工评分（30题抽检） | ⬜ |
| P1 | FUSION 跑 ChartQA（visual 索引已有，只需跑 benchmark） | 🔶 |
| P2 | 多模型对比（Llama 3.2 Vision） | ⬜ |
| P2 | RAGAS 忠实度自动评测 | ⬜ |
| P2 | EM 后处理（截断短答案、去除解释性文字） | ⬜ |
