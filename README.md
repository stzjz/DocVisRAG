# DocVisRAG

DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态文档问答系统。

当前能力覆盖：
- PDF/图片摄入与页面渲染
- OCR（PaddleOCR 优先，失败自动回退 pytesseract）
- 页面级混合检索（VLM 页面摘要 + OCR 文本）
- 端到端文档问答（检索 + VLM 生成 + 页码引用）
- 实验评测（Recall@K / MRR / EM / F1 / ANLS + 错误分析）
- 阶段 9 可选增强：视觉检索（Byaldi/ColPali 风格）与 fusion 融合检索

说明：视觉检索是可选增强，不影响主线 hybrid。即使 visual 依赖不可用，阶段 6/7/8 的 hybrid 流程仍可运行。

---

## 1. 环境与 Docker

### 1.1 构建镜像
```bash
docker build -t docvisrag:cu124 .
```

默认镜像会安装 Byaldi/ColPali visual retrieval 依赖，让 `fusion` 成为 Demo 主线检索模式。
如果只想构建轻量 hybrid 镜像，可使用：
```bash
docker build --build-arg INSTALL_VISUAL=false -t docvisrag:cu124 .
```

### 1.2 启动容器（推荐）
```bash
docker run --gpus '"device=0,1"' --ipc=host --network=host -it --rm \
  -v /data3/zengjian/DocVisRAG:/workspace/DocVisRAG \
  -v /data3/zengjian/.cache:/root/.cache \
  -w /workspace/DocVisRAG \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
  -e HF_ENDPOINT=https://hf-mirror.com \
  docvisrag:cu124
```

### 1.3 启动后建议检查
```bash
export HF_ENDPOINT=https://hf-mirror.com
env | grep -E "HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|HF_ENDPOINT"
python scripts/env/check_env.py
```

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
- `--retriever-type`（阶段9新增：`hybrid|visual|fusion`）
- `--visual-index-dir`（visual/fusion 时）

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

### Q3：为什么每次进入容器都要重新安装包
`docker run --rm` 下，容器内临时 `pip install` 会在退出后丢失。

建议：
- 把依赖写入项目并 `docker build` 重建镜像。

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
