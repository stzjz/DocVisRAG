# DocVisRAG

DocVisRAG 是一个面向多页 PDF、扫描文档、课件、表格、图表和版面密集材料的多模态文档 RAG 系统。

项目主线是：**先对文档集合离线建索引，根据问题检索证据页，再把候选页面作为视觉上下文交给 VLM 回答**。这和很多公开 VQA 数据集的“给定一张图再问答”不同；DocVisRAG 更关注真实文档助手场景中的完整流程：先建库、再检索、最后基于证据页回答。

## 主线

### 1. 为什么自建 HomeworkQA

DocVQA、ChartQA、TextVQA 等公开数据集适合做 sanity check，但它们大多是 question-image pair，不太适合评估“多页文档集合检索 + VLM 回答”的实际 RAG 流程。

因此本项目自建了两个核心评估集：

| 数据集 | 规模 | 来源 | 用途 |
|---|---:|---|---|
| HomeworkQA | 180 道简答题 | 16 个自收集 PDF | 多页文档问答，使用 LLM judge 评估 |
| HomeworkQA-MC | 180 道选择题 | 同一批 16 个 PDF | 控制答案空间，比较不同检索/融合方式 |

问题围绕多页证据、视觉版面、表格、表单和跨页推理构造，更贴近企业/个人文档助手的应用：先离线索引文档，再根据用户问题检索证据页并回答。

### 2. 系统流程

DocVisRAG 分为离线索引和在线问答两个阶段。

```text
离线索引
PDF / 图片
  -> 页面渲染
  -> OCR 与版面解析
  -> 页面摘要
  -> text / hybrid / visual 索引

在线问答
问题
  -> 检索 top-k 证据页
  -> 将页面和问题输入 VLM
  -> 输出答案和引用页
```

支持的检索模式：

| 模式 | 检索信号 | 输入给回答模型的证据 | 作用 |
|---|---|---|---|
| text | OCR 文本块/页面 | 检索页面与 OCR 上下文 | 强文本基线 |
| hybrid | OCR + 页面摘要 + 词法重排 | 检索页面 | 通用文档检索器 |
| visual | 页面图像 embedding | 检索页面图像 | 版面/表格/视觉证据 |
| fusion | 多路融合 | 检索页面或候选答案 | 主多模态设置 |

在 HomeworkQA 简答题上，当前主实验默认使用**答案级 hybrid/fusion**：

- `hybrid-answer`：分别用 `text` 和 `raw_hybrid` 生成答案，再由本地 LLM 选择/综合最终答案。
- `fusion-answer`：分别用 `text`、`raw_hybrid`、`visual` 和 raw page-fusion 生成答案，再选择/综合最终答案。
- 页面级、块级、visual-dominant、snippet 级融合保留为消融实验。

## 重点配置

下面是当前主实验和提交说明中需要突出的位置。

| 项目 | 默认设置 |
|---|---|
| 保留进 Git 的自建数据集 | `data/bench_full/homeworkqa`, `data/bench_full/homeworkqa_mc` |
| 排除出 Git 的生成产物 | `data/bench_runs`, `.work`, FAISS/Byaldi 索引、渲染页面、临时预测文件 |
| Text 索引 | block 级 OCR 文本索引 |
| Hybrid 索引 | block 级 OCR-only hybrid 索引，词法重排 `lexical_weight=0.2` |
| Visual 索引 | ColQwen/Byaldi 页面视觉索引 |
| 简答题主实验 hybrid | 答案级融合：`text + raw_hybrid_block` |
| 简答题主实验 fusion | 答案级融合：`text + raw_hybrid_block + visual + raw_page_fusion` |
| 页面级/块级/visual-dominant fusion | 作为消融实验保留 |

一键主实验：

```bash
cd /data1/home/zengjian/DocVisRAG

CUDA_VISIBLE_DEVICES=3 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
DOCVISRAG_LOCAL_FILES_ONLY=1 \
bash scripts/bench/run_homeworkqa_experiments.sh \
  --device 3 \
  --run-root data/bench_runs/homeworkqa_experiments_$(date +%Y%m%d_%H%M%S)
```

复用已有 visual 索引以节省时间：

```bash
bash scripts/bench/run_homeworkqa_experiments.sh \
  --device 3 \
  --run-root data/bench_runs/homeworkqa_experiments_$(date +%Y%m%d_%H%M%S) \
  --reuse-visual data/bench_runs/homeworkqa_experiments_20260618_101350/indexes/visual
```

## 当前主结果

完整结果见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

### HomeworkQA 简答题

| 方法 | Judge Acc | Judge Score | Relaxed Acc | F1 | Citation Acc |
|---|---:|---:|---:|---:|---:|
| text | 0.4389 | 0.5039 | 0.4333 | 0.3519 | 0.8667 |
| hybrid-answer | **0.4556** | 0.4950 | **0.4833** | **0.3696** | **0.9278** |
| visual | 0.4500 | 0.5289 | 0.4167 | 0.3415 | 0.8833 |
| fusion-answer | **0.5000** | **0.5578** | **0.5444** | **0.3928** | **0.9722** |

简答题主实验满足预期趋势：

```text
hybrid-answer > text
fusion-answer > visual
```

### HomeworkQA-MC 选择题

移除 9 道 text-only 偏易/偏置题并补入更难题后，最新 MC 主实验结果如下：

| 方法 | Accuracy | Parse Rate | R@1 | R@3 | MRR |
|---|---:|---:|---:|---:|---:|
| text | 0.8444 | 1.0000 | 0.8389 | 0.9833 | 0.9099 |
| hybrid | 0.8722 | 1.0000 | 0.8278 | 0.9722 | 0.8935 |
| visual | 0.8889 | 1.0000 | 0.7500 | 0.9222 | 0.8381 |
| fusion | **0.9056** | 1.0000 | 0.8111 | 0.9500 | 0.8876 |

重平衡后的 MC 评估也满足：

```text
hybrid > text
fusion > visual
```

## 目录结构

```text
DocVisRAG/
├── app.py                         # Gradio UI demo
├── src/docvisrag/                 # 核心包
│   ├── ingest/                    # 渲染、OCR、版面解析、页面摘要
│   ├── retrieve/                  # text / hybrid / visual / fusion 索引
│   ├── qa/                        # 多模态问答引擎
│   ├── eval/                      # 指标计算
│   └── vlm/                       # Qwen-VL 封装
├── scripts/
│   ├── ingest/                    # 文档预处理
│   ├── retrieve/                  # 建索引和检索脚本
│   ├── bench/                     # benchmark 与 HomeworkQA 实验脚本
│   └── eval/                      # 评估工具
├── docs/
│   ├── README.md                  # 文档地图
│   ├── EXPERIMENTS.md             # 合并后的实验报告
│   └── TEXT_AWARE_FUSION.md       # 融合方法说明
└── requirements-*.txt
```

## 快速开始

先安装基础依赖：

```bash
pip install -r requirements-base.txt
```

如果要使用 `visual` 或 `fusion`，再安装视觉检索依赖：

```bash
pip install -r requirements-visual.txt
```

### 启动 UI Demo

`app.py` 会启动 Gradio 页面，默认监听 `0.0.0.0:7860`。

本地启动：

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

服务器启动时，建议先做端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 zengjian@222.200.185.159
cd /data1/home/zengjian/DocVisRAG
python app.py
```

然后在本机浏览器访问 `http://127.0.0.1:7860`。

### 运行 HomeworkQA 实验

在服务器上一键运行 HomeworkQA / HomeworkQA-MC 主实验和消融实验：

```bash
cd /data1/home/zengjian/DocVisRAG

CUDA_VISIBLE_DEVICES=3 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
DOCVISRAG_LOCAL_FILES_ONLY=1 \
bash scripts/bench/run_homeworkqa_experiments.sh \
  --device 3 \
  --run-root data/bench_runs/homeworkqa_experiments_$(date +%Y%m%d_%H%M%S)
```

只用已有索引运行 HomeworkQA-MC：

```bash
cd /data1/home/zengjian/DocVisRAG

export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DOCVISRAG_LOCAL_FILES_ONLY=1
export PY=/data1/home/zengjian/miniconda3/envs/docvisrag/bin/python
export IDX=data/bench_runs/homeworkqa_experiments_20260618_101350/indexes
export RUN=data/bench_runs/homeworkqa_mc_rebalanced_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN"

$PY scripts/bench/eval_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa_mc/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --text-index-dir "$IDX/text_block" \
  --hybrid-index-dir "$IDX/hybrid_block_ocr_only_lex02" \
  --visual-index-dir "$IDX/visual" \
  --out "$RUN/predictions.jsonl" \
  --summary-out "$RUN/summary.json" \
  --modes text hybrid visual fusion \
  --top-k 5 \
  --retrieval-query-mode question_choices \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-new-tokens 16
```

## 文档

建议从这里开始：

- [docs/README.md](docs/README.md)：文档地图和推荐阅读顺序。
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：公开数据集、HomeworkQA、HomeworkQA-MC、消融实验与复现实验指令。
- [docs/TEXT_AWARE_FUSION.md](docs/TEXT_AWARE_FUSION.md)：检索与融合方法说明。
- [docs/archive/SERVER_DEPLOYMENT.md](docs/archive/SERVER_DEPLOYMENT.md)：服务器部署与远程运行记录。

## 应用场景

DocVisRAG 面向更新频率较低、可以接受离线建索引成本的文档集合，例如企业文档管理、内部知识库、课程资料、报告库和个人文档助手。一个具体目标场景是：用户上传或维护一批 PDF，系统离线建立索引，在线提问时先检索证据页，再由 VLM 根据页面内容回答。
