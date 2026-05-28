# DocVisRAG

DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态文档问答系统。

当前能力覆盖：
- PDF/图片摄入与页面渲染
- OCR（PaddleOCR 优先，失败自动回退 pytesseract）
- 版面分析（PP-Structure 优先，OpenCV 回退）：检测文本/表格/图表/公式等语义区域
- 版面感知分块：将 OCR 文本按语义区域归并，生成类型标签化文档块
- 页面级混合检索（VLM 页面摘要 + OCR 文本）
- 纯文本 RAG 基线（仅 OCR 文本，无图像/VLM 摘要）
- 端到端文档问答（检索 + VLM 生成 + 页码引用）
- 实验评测（Recall@K / MRR / EM / F1 / ANLS + 错误分析）
- 纯文本 vs 多模态对比评测
- 阶段 9 可选增强：视觉检索（Byaldi/ColPali 风格）与 fusion 融合检索

说明：视觉检索是可选增强，不影响主线 hybrid。即使 visual 依赖不可用，阶段 6/7/8 的 hybrid 流程仍可运行。

---

## 项目进度总览（对照开题报告需求）

### 表 A：系统功能需求完成度（对应开题报告 表 2）

| 功能模块 | 优先级 | 状态 | 实现文件 |
|----------|--------|------|----------|
| 文档上传（PDF/图片/PPT） | 高 | ✅ 完成 | `ingest/render.py`, `app.py` |
| 文档解析（渲染+OCR+版面+元数据） | 高 | ✅ 完成 | `render.py` / `ocr.py` / `layout.py` |
| 多模态分块（文本/表格/图表/公式） | 高 | ✅ 完成 | `layout.py` + `layout_chunk.py` |
| 向量检索（文本+图像+混合+重排） | 高 | ✅ 完成 | `hybrid_index.py` / `visual_index.py` / `fusion.py` |
| 问答生成（基于证据+少幻觉） | 高 | ✅ 完成 | `qa/doc_qa.py` / `qa/text_qa.py` |
| 引用定位（页码+区域高亮） | 高 | ✅ 完成 | DocQA 输出 "第 X 页" + OCR bbox 高亮 |
| 可视化展示（Web UI） | 中 | ✅ 完成 | `app.py` (Gradio) |
| 实验评测（召回+准确率+效率） | 高 | ✅ 完成 | `eval/` + `bench/` |

### 表 B：研究目标完成度（对应开题报告 3.1 节）

| 研究目标 | 状态 | 说明 |
|----------|------|------|
| 1. 多页 PDF/扫描/PPT 导入预处理 | ✅ | PyMuPDF 渲染 + PaddleOCR + PP-Structure 版面分析 |
| 2. 多模态检索模块（视觉+文本） | ✅ | hybrid / visual / fusion 三种检索模式 |
| 3. 多模态问答生成+来源定位 | ✅ | Qwen2.5-VL 生成 + 页码引用 + bbox 证据高亮 |
| 4. 纯文本 vs 多模态对比实验 | ✅ | `compare_rag.py` 一键对比，含检索+QA双维度 |

### 表 C：实验设计与评测完成度（对应开题报告 5.2–5.4 节）

| 实验/指标 | 状态 | 说明 |
|-----------|------|------|
| 纯文本 RAG 基线 | ✅ | `TextDocQAEngine` + `text_qa.py` |
| 多模态 RAG 实验组 | ✅ | `DocQAEngine` (hybrid/visual/fusion) |
| Recall@K / MRR / NDCG@5 | ✅ | `eval_retrieval.py` 含分类型统计 |
| EM / F1 / ANLS | ✅ | `eval_qa.py` 含 Citation Accuracy |
| 错误分析 | ✅ | `make_error_analysis.py` 自动分类 |
| 对比实验一键脚本 | ✅ | `compare_rag.py` + `run_benchmark_suite.py` |
| DocVQA 评测 | ✅ | 已完成 100 题 benchmark 运行 |
| ChartQA 评测 | ⬜ | 框架就绪，数据集待运行 |
| TextVQA 评测 | ⬜ | 框架就绪，数据集待运行 |
| 自建课程文档集 (60-120 QA) | ⬜ | 待创建 |
| Relaxed Accuracy | ⬜ | 图表数值评测指标待实现 |
| RAGAS / Faithfulness | ⬜ | 忠实度自动评测待引入 |

### 表 D：预期成果完成度（对应开题报告 7.3 节）

| 预期成果 | 状态 | 说明 |
|----------|------|------|
| 1. 可运行的多模态问答原型 | ✅ | `app.py` Gradio Demo |
| 2. 文档索引流水线 | ✅ | 渲染→OCR→版面→摘要→索引 全链路 |
| 3. 带引用来源的问答模块 | ✅ | 答案+依据+页码引用+bbox 高亮 |
| 4. 纯文本 vs 多模态对比结果 | 🔶 | 框架就绪，完整实验待运行 |
| 5. 成功/失败案例分析 | 🔶 | 分析框架就绪，待积累案例 |
| 6. 开题/结题报告 | 🔶 | 开题报告已完成，结题报告待撰写 |

> 图例：✅ 已完成 &nbsp; 🔶 部分完成/框架就绪 &nbsp; ⬜ 待实现

### 后续优先工作

1. **P0** — 运行 ChartQA + TextVQA 完整评测，产出对比实验数据
2. **P0** — 创建自建课程文档集（8-12 份文档，60-120 条 QA）
3. **P1** — 实现 Relaxed Accuracy 指标，补充 RAGAS 忠实度评测
4. **P1** — 图号/表号引用增强（当前仅页码，缺少 "图 Y""表 Z" 格式）
5. **P2** — 引入 BM25 / 关键词检索通道做混合重排
6. **P2** — 多模型对比（Llama 3.2 Vision / DeepSeek-VL）
7. **P2** — 图表专项处理管线（chart-to-table / chart QA）

---

## 系统架构与数据流

### 总体架构

项目采用 **"离线建库 + 在线问答"** 两段式结构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        离线建库阶段                                │
│                                                                   │
│  输入文档 ──→ [渲染] ──→ [OCR] ──→ [版面分析] ──→ [摘要]            │
│  (PDF/图片)     │          │          │              │             │
│                 v          v          v              v             │
│           manifest.json  ocr.jsonl  layout.jsonl  page_summaries  │
│                 │          │          │              │             │
│                 │          v          │              v             │
│                 │    [text_index]     │       [hybrid_index]       │
│                 │    FAISS·纯文本     │       FAISS·摘要+OCR       │
│                 │                     │                            │
│                 └─────────────────────┘                            │
│                           │                                       │
│                           v                                       │
│                     [visual_index]                                │
│                   Byaldi·ColPali·可选                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        在线问答阶段                                │
│                                                                   │
│  用户问题 ──→ [检索] ──→ Top-K 页面 ──→ [VLM 生成] ──→ 答案+引用   │
│                │                          │                       │
│                ├─ hybrid (FAISS·摘要+OCR)  │                       │
│                ├─ visual (ColPali·视觉)    ├─ Qwen2.5-VL 多图推理  │
│                ├─ fusion (RRF·混合重排)    │                       │
│                └─ text   (FAISS·纯OCR)    └─ 纯文本 LLM 生成       │
│                                                                   │
│  输出：答案 + 依据 + 引用（第 X 页）+ 不确定性说明                    │
└─────────────────────────────────────────────────────────────────┘
```

### 项目文件结构

```
DocVisRAG/
├── src/docvisrag/                  # 核心 Python 包
│   ├── config.py                   # ProjectConfig（模型ID、设备、top_k）
│   ├── ingest/                     # 文档摄入层
│   │   ├── render.py               #   PDF渲染→页面图像 + manifest
│   │   ├── ocr.py                  #   PaddleOCR / Tesseract 文本提取
│   │   ├── layout.py               #   版面分析（PP-Structure / OpenCV）
│   │   ├── layout_chunk.py         #   版面感知分块
│   │   └── page_summary.py         #   VLM 页面摘要生成
│   ├── retrieve/                   # 检索层
│   │   ├── base.py                 #   BaseRetriever 抽象基类
│   │   ├── text_index.py           #   纯文本 FAISS 索引
│   │   ├── hybrid_index.py         #   混合页面 FAISS 索引（摘要+OCR）
│   │   ├── visual_index.py         #   ColPali/Byaldi 视觉索引
│   │   └── fusion.py               #   RRF 融合排序（k=60）
│   ├── qa/                         # 问答层
│   │   ├── doc_qa.py               #   多模态文档QA（检索+多图VLM生成）
│   │   ├── text_qa.py              #   纯文本RAG基线（检索+纯文本LLM生成）
│   │   └── page_qa.py              #   单页面QA
│   ├── vlm/                        # 视觉语言模型层
│   │   └── qwen_vl.py              #   Qwen2.5-VL / Qwen3-VL 封装
│   ├── eval/                       # 评测层
│   │   └── metrics.py              #   R@K / MRR / NDCG / EM / F1 / ANLS / CitationAcc
│   └── ui/                         # UI层（预留）
├── scripts/                        # CLI 脚本（按功能分层）
│   ├── ingest/                     #   摄入：render / ocr / layout / chunk / summary
│   ├── retrieve/                   #   检索：build_*_index / *_search
│   ├── qa/                         #   问答：vlm_qa / page_qa / doc_qa / text_qa
│   ├── eval/                       #   评测：eval_retrieval / eval_qa / error_analysis
│   ├── bench/                      #   基准：prepare_benchmark / run_suite / compare_rag
│   ├── env/                        #   环境：check_env.py
│   └── loopback_test.sh            #   端到端冒烟测试
├── app.py                          # Gradio Web UI
├── configs/default.yaml            # 默认配置
├── data/                           # 数据目录
│   ├── samples/                    #   测试样本（PDF / 图片）
│   ├── bench_runs/                 #   benchmark 运行输出
│   ├── outputs/                    #   管线中间产物（pages / ocr / layout / summary / index）
│   └── indexes/                    #   索引文件（FAISS / Byaldi）
├── eval/                           # 评测配置文件
│   ├── questions.example.jsonl     #   问题样例格式
│   └── bench_suite.example.json    #   benchmark 套件定义
├── requirements-base.txt           # 核心依赖
├── requirements-visual.txt         # 可选视觉检索依赖
└── Dockerfile                      # Docker 构建（备用）
```

### 数据格式规范

#### 核心中间文件

| 文件 | 格式 | 关键字段 | 生成者 | 消费者 |
|------|------|----------|--------|--------|
| `manifest.json` | JSON数组 | `doc_id, page_index, image_path, width, height` | `render.py` | 全部后续管线 |
| `ocr.jsonl` | JSONL | `doc_id, page_index, text, bbox[x0,y0,x1,y1], confidence` | `ocr.py` | text_index, hybrid_index, layout_chunk |
| `layout.jsonl` | JSONL | `region_id, region_type, bbox[0-1], confidence` | `layout.py` | layout_chunk |
| `layout_chunks.jsonl` | JSONL | `chunk_id, chunk_type, text, bbox, page_index` | `layout_chunk.py` | build_layout_index |
| `page_summaries.jsonl` | JSONL | `doc_id, page_index, summary(≤200字)` | `page_summary.py` | hybrid_index |
| `index.faiss` | 二进制 | FAISS IndexFlatIP (余弦内积) | build_*_index.py | *_search, QA |
| `metadata.jsonl` | JSONL | `id, doc_id, page_index, score` | build_*_index.py | 检索结果组装 |
| `config.json` | JSON | `model_name, dimension, metric, num_vectors` | build_*_index.py | index.load() |

#### 评测数据格式

```jsonl
# questions.jsonl（每行一个QA样本）
{"id": "q001", "doc_path": "data/samples/demo.pdf", "question": "...",
 "answer": "标准答案", "evidence_pages": [1, 2], "type": "text|table|chart|summary|layout"}

# predictions.jsonl（每行一个预测结果）
{"id": "q001", "pred_answer": "...", "gold_answer": "...", "em": 0.0,
 "f1": 0.85, "anls": 0.72, "recall@3": 1.0, "citation_accuracy": 1.0}
```

### 检索器对比矩阵

| 特性 | text | hybrid | visual | fusion |
|------|------|--------|--------|--------|
| 检索单元 | OCR文本行 | 页面（摘要+OCR） | 页面图像 | 页面（hybrid+visual融合） |
| 嵌入模型 | bge-small-zh | bge-small-zh | ColQwen2-V1.0 | 两者结合 |
| 向量库 | FAISS·512d | FAISS·512d | Byaldi·多向量 | FAISS + Byaldi |
| 需要GPU | 否 | 否 | 是 | 是 |
| 图像作为证据 | 否 | 是 | 是 | 是 |
| 版面结构保留 | 否（线性文本） | 部分（摘要描述） | 是（原始页面） | 是 |
| 典型用例 | 纯文本RAG基线 | 通用文档QA | 图表/表格密集型 | 最全面 |

### 生成模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_id` | `Qwen/Qwen2.5-VL-3B-Instruct` | 可通过 `DOCVISRAG_MODEL_ID` 环境变量覆盖 |
| `device_map` | `auto` | accelerate 自动分配 |
| `torch_dtype` | `auto` | 默认 fp16 |
| `load_in_4bit` | `False` | BitsAndBytes 4-bit 量化，约节省 60% 显存 |
| `max_new_tokens` | 512 | 生成回答的最大 token 数 |

### 环境变量一览

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `DOCVISRAG_MODEL_ID` | 覆盖 VLM 模型ID | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `DOCVISRAG_LOCAL_FILES_ONLY` | 强制离线模式（仅本地缓存） | `false` |
| `DOCVISRAG_OCR_BACKEND` | OCR后端选择 | `auto` (paddle→tesseract) |
| `DOCVISRAG_LAYOUT_BACKEND` | 版面检测后端 | `auto` (ppstructure→opencv) |
| `DOCVISRAG_STRICT_VISUAL_CHECK` | 严格 peft 兼容性检查 | `false` (relaxed) |
| `HF_ENDPOINT` | HuggingFace 镜像 | `https://hf-mirror.com` |
| `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` | 模型缓存路径 | `~/.cache/huggingface` |

---

## 1. 环境与虚拟环境

推荐直接使用 `python -m venv` 安装和运行项目，不再依赖 Docker。

### 1.1 系统依赖
Ubuntu 22.04 上建议先安装：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  poppler-utils tesseract-ocr tesseract-ocr-chi-sim \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  build-essential git curl wget ca-certificates
```

### 1.2 创建虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-base.txt
```

阶段 9 的 visual / fusion 检索依赖可选安装：

```bash
python -m pip install -r requirements-visual.txt
```

### 1.3 建议环境变量
```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.4 安装后建议检查
```bash
env | grep -E "HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|HF_ENDPOINT"
python scripts/env/check_env.py
python scripts/env/check_env.py --visual
```

说明：
- `python scripts/env/check_env.py --visual` 仅在安装了 `requirements-visual.txt` 后运行。
- 当前固定的 relaxed visual 依赖栈里，`peft.tp_helper : NO` 可能是预期现象，不影响继续构建 visual index。

---

## 本地开发指南：有限资源下能做什么

本项目的 GPU 密集型操作（VLM 推理、视觉索引构建）需要在配备 CUDA GPU 的服务器上运行。但大量开发准备工作可以在本地（CPU only / 小显存）完成。

### 操作分级

| 等级 | 操作 | 需要 GPU | 可在本地完成 |
|------|------|----------|-------------|
| 🔵 CPU | PDF 渲染 (ingest_render) | 否 | ✅ |
| 🔵 CPU | OCR 文本提取 (run_ocr) | 否 | ✅ |
| 🔵 CPU | 版面分析·OpenCV 回退 (run_layout --backend opencv) | 否 | ✅ |
| 🔵 CPU | 文本索引构建 (build_text_index) | 否 | ✅ |
| 🔵 CPU | 混合索引构建 (build_hybrid_index) | 否 | ✅ |
| 🔵 CPU | 版面索引构建 (build_layout_index) | 否 | ✅ |
| 🔵 CPU | 文本/混合检索 (text_search / hybrid_search) | 否 | ✅ |
| 🔵 CPU | 纯文本 QA (text_qa) — 调用本地嵌入模型 | 否 | ✅ |
| 🔵 CPU | 检索评测 (eval_retrieval) | 否 | ✅ |
| 🟡 CPU† | 版面分析·PP-Structure (run_layout --backend ppstructure) | 否† | ✅ |
| 🟡 CPU† | 纯文本 QA — 使用小模型或 CPU 推理 | 否† | ✅ |
| 🔴 GPU | VLM 页面摘要生成 (build_page_summaries) | 是 | ❌ |
| 🔴 GPU | 视觉索引构建 (build_visual_index) | 是 | ❌ |
| 🔴 GPU | 多模态 QA (doc_qa --retriever-type hybrid/visual/fusion) | 是 | ❌ |
| 🔴 GPU | QA 评测 (eval_qa) | 是 | ❌ |
| 🔴 GPU | 对比评测 QA 部分 (compare_rag) | 是 | ❌ |
| 🔴 GPU | Gradio Demo (app.py) | 是 | ❌ |

> † PP-Structure 可能首次下载模型；纯文本 QA 的 LLM 生成部分在 CPU 上会很慢，可用 `--limit 1` 验证链路。

### 本地可完成的 8 件事

#### 1. 验证代码链路完整性

在 CPU 环境安装基础依赖后，用单张小图片跑通全链路（跳过 VLM 环节）：

```bash
# 安装 CPU 依赖（跳过 CUDA PyTorch）
python -m pip install -r requirements-base.txt

# 跑环境检查
python scripts/env/check_env.py

# 用 test_stage1.png 跑通基础链路
python scripts/ingest/ingest_render.py \
  --input data/samples/test_stage1.png \
  --output data/outputs/local_test

python scripts/ingest/run_ocr.py \
  --manifest data/outputs/local_test/manifest.json \
  --out data/outputs/local_test/ocr.jsonl

python scripts/ingest/run_layout.py \
  --manifest data/outputs/local_test/manifest.json \
  --out data/outputs/local_test/layout.jsonl \
  --backend opencv

python scripts/retrieve/build_text_index.py \
  --ocr data/outputs/local_test/ocr.jsonl \
  --index-dir data/indexes/local_test_text

# 纯文本检索验证
python scripts/retrieve/text_search.py \
  --index-dir data/indexes/local_test_text \
  --question "这张图片的主要内容是什么？" --top-k 3
```

#### 2. 构建自建课程文档集（P0 待办）

这是完全不需要 GPU 的工作，却是开题报告的核心交付物之一：

- 收集 8–12 份不同类型的文档：课程 PPT（导出为PDF）、作业说明 PDF、论文 PDF、扫描件、带图表的报告
- 每份文档设计 5–10 个问题，覆盖 text / table / chart / summary / layout 五种类型
- 标注 `evidence_pages` 和标准答案
- 参照格式：`eval/questions.example.jsonl`

```jsonl
{"id": "self001", "doc_path": "data/self_built/xxx.pdf", "question": "...",
 "answer": "...", "evidence_pages": [3], "type": "text"}
```

#### 3. 准备公开数据集子集

下载 DocVQA、ChartQA、TextVQA 的小规模子集（例如各 50 题），在本地整理好 `questions.jsonl`。数据集下载后提前上传到服务器，避免在服务器上临时下载。

```bash
# 查看 benchmark 配置格式
cat eval/bench_suite.example.json

# 本地准备问题文件（纯文本工作，无需 GPU）
# 参照 data/bench_runs/suite_20260511_131616/docvqa_small/inputs/questions.jsonl
```

#### 4. 完善提示词模板

提示词模板在 `src/docvisrag/qa/doc_qa.py:_build_prompt()` 中，修改不涉及模型推理：

- 调整系统指令的表达方式
- 为不同类型的文档设计专用 prompt（论文 vs PPT vs 财报）
- 增加 "图 Y""表 Z" 引用格式的 prompt 指令
- 测试中文/英文双语 prompt 效果

#### 5. 增强评测指标

在 `src/docvisrag/eval/metrics.py` 中添加新指标，纯代码逻辑不依赖 GPU：

- `relaxed_accuracy()` — 图表数值问答的容差匹配
- 评测分组细化 — 按 region_type 统计（text / table / chart 分报告）
- 在 `make_error_analysis.py` 中增加版面相关的错误分类

#### 6. 扩展引用格式（图号/表号）

当前引用仅支持 "第 X 页"。可在本地设计并实现 "图 Y"、"表 Z" 的引用解析逻辑：

- 扩展 `_parse_citations()` 正则支持 `图\s*\d+`、`表\s*\d+`
- 设计从 layout regions 中提取图号/表号的逻辑
- 在 prompt 中增加图号表号输出要求

#### 7. 编写单元测试

为各个独立模块编写测试（不需要模型）：

- `render.py`：PDF 渲染、图片复制、manifest 序列化
- `ocr.py`：JSONL 解析、OCR block 数据类
- `layout.py`：IoU 计算、bbox 归一化、区域分类逻辑
- `text_index.py`：JSONL 加载、格式验证
- `metrics.py`：各指标的边界条件测试
- `fusion.py`：RRF 排序边界条件

#### 8. 代码审查和文档完善

- 跑 `scripts/loopback_test.sh` 的 CPU 子集（设置 `RUN_VISUAL=0 RUN_DEMO=0`）
- 阅读并理解队友新增的 `check_env.py` 和 `loopback_test.sh`
- 更新 README 中各阶段的用法说明
- 整理服务器部署 checklist

### 本地 → 服务器迁移检查清单

当需要在服务器上运行时，确保以下准备就绪：

| # | 事项 | 本地完成 | 服务器验证 |
|---|------|----------|-----------|
| 1 | 代码推送到 GitHub | ✅ 本地 commit & push | `git pull` |
| 2 | 测试样本上传 | `data/samples/` 就绪 | 检查路径 |
| 3 | 预下载模型 | 通过 HF mirror 缓存 | 检查 `~/.cache/huggingface` |
| 4 | 虚拟环境配置 | `requirements-base.txt` 锁定版本 | `pip install` 验证 |
| 5 | 评测问题集 | `eval/questions.jsonl` 就绪 | 路径验证 |
| 6 | 自建文档集 | `data/self_built/` 就绪 | 路径验证 |
| 7 | 环境变量设置 | 确认 `HF_ENDPOINT` 等 | `check_env.py` 验证 |
| 8 | GPU 可用性 | — | `nvidia-smi` + `torch.cuda.is_available()` |
| 9 | visual 依赖检查 | — | `check_env.py --visual` |
| 10 | 端到端冒烟测试 | — | `bash scripts/loopback_test.sh` |

---

## 2. 阶段式功能与命令

## 阶段 1：单图 VLM 问答
```bash
python scripts/qa/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "请概括这张图片的主要内容。" \
  --max-new-tokens 1024
```

## 阶段 2：文档摄入与页面渲染
```bash
python scripts/ingest/ingest_render.py \
  --input data/samples/开题报告.pdf \
  --output data/outputs/demo_pages \
  --dpi 180
```

## 阶段 3：指定页面问答
```bash
python scripts/qa/page_qa.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --page 5 \
  --question "这一页主要讲了什么？"
```

## 阶段 3b：版面分析与语义区域检测

检测页面中的文本、标题、表格、图表、公式、页眉、页脚等语义区域。

### 3b.1 版面检测
```bash
python scripts/ingest/run_layout.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/layout.jsonl \
  --backend auto
```

输出 `layout.jsonl` 每行包含：
- `region_id` / `doc_id` / `page_index`
- `region_type`：`text` | `title` | `table` | `chart` | `figure` | `formula` | `header` | `footer` | `page_number`
- `bbox`：归一化坐标 `[x0, y0, x1, y1]`（0-1）
- `confidence`：检测置信度

可选参数：
- `--backend auto|ppstructure|opencv`：检测后端。`auto` 优选 PP-Structure（PaddleOCR 生态），失败回退 OpenCV 启发式检测
- `--ocr <path>`：可选 OCR JSONL，用于后续归并
- `--enriched-out <path>`：可选输出，保存 OCR 文本归并后的富化区域

### 3b.2 版面感知分块
```bash
python scripts/ingest/run_layout_chunk.py \
  --layout data/outputs/demo_pages/layout.jsonl \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --out data/outputs/demo_pages/layout_chunks.jsonl \
  --min-text-length 10
```

输出 `layout_chunks.jsonl` 每行包含：
- `chunk_id`：唯一块 ID
- `chunk_type`：`text` | `title` | `table` | `formula` 等
- `text`：该区域内的聚合 OCR 文本
- `bbox` / `page_index` / `doc_id`

### 3b.3 版面感知索引
```bash
python scripts/retrieve/build_layout_index.py \
  --chunks data/outputs/demo_pages/layout_chunks.jsonl \
  --index-dir data/indexes/demo_layout
```

与普通 text index 的区别：每个向量对应一个**语义区域**（如一个表格、一个段落），而非单个 OCR 文本行。元数据中保留 `chunk_type` 和 `bbox`。

## 阶段 4：OCR 与纯文本检索基线
```bash
python scripts/ingest/run_ocr.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/ocr.jsonl

python scripts/retrieve/build_text_index.py \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --index-dir data/indexes/demo_text

python scripts/retrieve/text_search.py \
  --index-dir data/indexes/demo_text \
  --question "本文档的主要内容是什么？" \
  --top-k 5
```

## 阶段 5：轻量多模态页面索引（Hybrid）

### 5.1 生成每页 VLM 摘要
```bash
python scripts/ingest/build_page_summaries.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/page_summaries.jsonl
```

`page_summaries.jsonl` 每行包含：
- `doc_id`
- `page_index`
- `image_path`
- `summary`

### 5.2 建立页面级混合索引（摘要 + OCR）
```bash
python scripts/retrieve/build_hybrid_index.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --summaries data/outputs/demo_pages/page_summaries.jsonl \
  --index-dir data/indexes/demo_hybrid
```

索引目录包含：
- `index.faiss`
- `metadata.jsonl`
- `config.json`

### 5.3 页面级检索
```bash
python scripts/retrieve/hybrid_search.py \
  --index-dir data/indexes/demo_hybrid \
  --question "图表展示了什么趋势？" \
  --top-k 3
```

返回字段包含：
- `page_index`
- `image_path`
- `summary`
- `ocr_text_preview`
- `score`

## 阶段 5b：纯文本 RAG 基线（仅 OCR 文本）

纯文本 RAG 仅使用 OCR 文本进行检索和生成，不使用页面图像或 VLM 摘要。
这是开题报告要求的基线对照组，用于与多模态 RAG 做对比实验。

### 5b.1 纯文本 QA
```bash
python scripts/qa/text_qa.py \
  --index-dir data/indexes/demo_text \
  --question "本文档的主要内容是什么？" \
  --top-k 5
```

CLI 参数：
- `--index-dir`：文本索引目录（由 `build_text_index.py` 构建）
- `--question`：用户问题
- `--top-k`：检索文本块数量（默认 5）
- `--model-id`：可选 LLM model id 覆盖
- `--load-in-4bit`：启用 4-bit 量化
- `--max-new-tokens`：最大生成 token 数（默认 512）

### 5b.2 纯文本检索评测
```bash
python scripts/eval/eval_retrieval.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_text \
  --retriever-type text \
  --out data/outputs/eval_retrieval_text.json
```

Text 模式下，检索返回的文本块按 `page_index` 去重后映射为页面列表，再与 `evidence_pages` 计算 Recall@K / MRR / NDCG@5。

### 5b.3 纯文本 QA 评测
```bash
python scripts/eval/eval_qa.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_text \
  --retriever-type text \
  --out data/outputs/predictions_text.jsonl \
  --limit 10
```

### 5b.4 纯文本 vs 多模态对比评测
```bash
python scripts/bench/compare_rag.py \
  --questions eval/questions.example.jsonl \
  --text-index-dir data/indexes/demo_text \
  --hybrid-index-dir data/indexes/demo_hybrid \
  --visual-index-dir data/indexes/demo_visual \
  --multimodal-type fusion \
  --out data/outputs/comparison_report.json \
  --top-k 5 --qa-top-k 3
```

输出对比报告包含：
- 各模式检索指标（R@1/3/5, MRR, NDCG@5）
- 各模式 QA 指标（EM, F1, ANLS, Recall@3, Citation Accuracy, 平均延迟）
- 多模态 vs 纯文本差值（正值表示多模态更优）

可选参数：
- `--skip-qa`：仅做检索对比，跳过 QA 评测（节省时间/显存）
- `--limit N`：仅评测前 N 条问题
- `--multimodal-type hybrid|fusion`：多模态检索器类型
- `--load-in-4bit`：启用 4-bit 量化

### 5b.5 纯文本 benchmark 一键评测
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type text
```

使用 `--retriever-type text` 即可将整个 benchmark 管线切换为纯文本模式：跳过 VLM 摘要和 hybrid/visual 索引构建，直接构建 text index 并评测。

## 阶段 6：端到端多模态 RAG 问答（默认 hybrid）

输入问题后，系统会自动：
1. 从索引检索 top-k 页面。
2. 读取候选页面图像、摘要、OCR 文本。
3. 交给 VLM 生成带引用页码的答案。

```bash
python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --question "第 2 节的主要结论是什么？" \
  --top-k 3
```

可选参数：
- `--model-id`
- `--load-in-4bit`
- `--max-new-tokens`
- `--retriever-type`（阶段9新增：`hybrid|visual|fusion|text`）
- `--visual-index-dir`（visual/fusion 时）

注意：`text` 检索模式请使用专用脚本 `scripts/qa/text_qa.py`，它仅基于 OCR 文本无需页面图像。

输出结构包含：
- `答案：`
- `依据：`
- `引用：`
- `不确定性：`

其中引用按“第 X 页”格式输出。

## 阶段 7：Gradio Demo
```bash
python app.py
```
浏览器打开：`http://localhost:7860`

Demo 支持：
- 上传 PDF/PNG/JPG/JPEG/WEBP
- 一键构建索引（渲染 -> OCR -> 摘要 -> 索引）
- 提问并显示答案、证据、页码引用与页面预览
- 可设置 4bit 量化、OCR 后端和 Demo 最大处理页数
- `Visual Build Mode` 默认 `required`，用于把 ColPali/visual index 纳入主链路；资源或依赖不稳时可切到 `auto` 或 `skip`
- 构建日志会输出阶段耗时，并在 session 输出目录保存 `build_meta.json`
- 问答结果会尽量基于 OCR bbox 生成证据高亮页面预览
- 检索器可切换 `hybrid / visual / fusion`，默认使用 `fusion`

## 阶段 8：评测脚本

### 8.1 准备小规模测试集（建议先做 20 条）
参考 `eval/questions.example.jsonl` 扩展自己的 `eval/questions.jsonl`，每行一个样本：

```json
{
  "id": "q001",
  "doc_path": "data/samples/demo.pdf",
  "question": "这份文档的主要结论是什么？",
  "answer": "标准答案",
  "evidence_pages": [1, 2],
  "type": "summary"
}
```

字段说明：
- `id`：唯一问题编号
- `doc_path`：文档路径
- `question`：用户问题
- `answer`：标准答案
- `evidence_pages`：证据页码（1-based）
- `type`：`text/table/chart/summary/layout`

20 条建议配比：
- `text`：6
- `summary`：4
- `table`：4
- `chart`：4
- `layout`：2

### 8.2 检索评测（Recall@K + MRR）
```bash
python scripts/eval/eval_retrieval.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --retriever-type hybrid \
  --out data/outputs/eval_retrieval.json
```
输出包含：`Recall@1/3/5`、`MRR`、`NDCG@5`，并按问题类型给出分组指标。

### 8.3 问答评测（EM/F1/ANLS）
```bash
python scripts/eval/eval_qa.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --retriever-type hybrid \
  --out data/outputs/predictions.jsonl \
  --limit 10
```
输出包含：`EM`、`F1`、`ANLS`、`Recall@3`、`Citation Accuracy` 和平均响应时间。

### 8.4 错误分析
```bash
python scripts/eval/make_error_analysis.py \
  --predictions data/outputs/predictions.jsonl \
  --out data/outputs/error_analysis.md
```

错误分类模板包含：
- OCR 识别错误
- 检索未召回正确页面
- 检索排序错误
- 图表/表格读取错误
- 生成模型幻觉
- 引用页码错误
- 标注或标准答案不清楚

### 8.5 指标解释
- `Recall@K`：前 K 个检索页是否命中至少一个标注证据页（越高越好）
- `MRR`：第一个命中页的倒数排名均值（越高越好）
- `EM`：预测答案与标准答案规范化后完全一致比例
- `F1`：预测与标准答案 token 重叠平衡指标
- `ANLS`：基于编辑距离的近似匹配分数（低于阈值计 0）

建议报告同时给出：总体指标、分类型指标、典型错误案例与改进方向。

## benchmark：一键自动化评测

```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs
```

suite 输出结构：
```text
data/bench_runs/<suite_name>/<benchmark_name>/
  inputs/
    manifest.snapshot.json
    questions.jsonl
  intermediate/
    ocr.jsonl
    page_summaries.jsonl
    hybrid_index/
    visual_index/ (可选)
  results/
    eval_retrieval.json
    predictions.jsonl
    predictions.summary.json
  reports/
    error_analysis.md
  logs/
    pipeline.log
  run_meta.json
  summary.json
```

suite 根目录包含：
- `overview.json`
- `benchmarks.md`
- `README.md`
- 仅当传 `--archive` 时才打包压缩文件

缺输入时可自动准备数据；若要强制重下重转，使用 `--force-prepare`。

---

## 3. 阶段 9：可选增强（Visual / Fusion 检索）

支持三种检索模式：
- `hybrid`：主线模式，摘要 + OCR
- `visual`：Byaldi/ColPali 页面视觉检索
- `fusion`：hybrid + visual 的 RRF 融合排序，当前 Demo 默认使用

### 9.1 构建 visual index
```bash
python scripts/retrieve/build_visual_index.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --index-dir data/indexes/demo_visual
```

### 9.2 visual 问答
```bash
python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --visual-index-dir data/indexes/demo_visual \
  --retriever-type visual \
  --question "这个文档有多少张图片？" \
  --top-k 3
```

### 9.3 fusion 问答
```bash
python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --visual-index-dir data/indexes/demo_visual \
  --retriever-type fusion \
  --question "这个文档有多少张图片？" \
  --top-k 3
```

### 9.4 benchmark 切换检索器
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type fusion
```

### 9.5 纯文本基线 benchmark
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_small \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type text
```

---

## 4. 常见问题

### Q1：visual 路线报 peft/transformers/byaldi 兼容错误
这是可选增强依赖冲突。先保证 hybrid 主线可用。

建议：
1. 把依赖版本固定到 Dockerfile / requirements 后重建镜像
2. visual/fusion 仅用于对比实验时启用

### Q2：benchmark 数据集加载失败
优先检查：
- HF 镜像与网络
- `datasets` 是否安装
- split 是否正确（可尝试 `--split train`）

### Q3：为什么推荐虚拟环境而不是 Docker
当前项目主线已经支持直接使用 `python -m venv` 安装运行：

- 环境更轻，便于调试和增量安装
- 本地缓存和模型目录更容易复用
- 文档里的命令可以直接在仓库根目录执行

如果你仍然需要容器化部署，可以保留 Dockerfile 作为可选方案，但开发、调试和实验默认推荐虚拟环境。

---

## 5. 脚本分层

`scripts/` 已按功能组织：
- `scripts/env/`
- `scripts/ingest/`
- `scripts/retrieve/`
- `scripts/qa/`
- `scripts/eval/`
- `scripts/bench/`

每个子目录均包含对应 `README.md` 说明脚本作用与用法。
