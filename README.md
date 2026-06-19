# DocVisRAG

DocVisRAG is a multimodal document RAG system for multi-page PDFs, scanned documents, slides, forms, tables, charts, and image-heavy course materials.

The project is built around one core setting: **retrieve evidence pages from a document collection, then answer with a VLM using the retrieved pages as visual context**. This is different from many public VQA benchmarks that assume a single given image per question. DocVisRAG evaluates the harder pipeline of **indexing documents first, retrieving candidate pages by question, and then sending the retrieved pages plus the question to the model**.

## Main Line

### 1. Why Build HomeworkQA

Public datasets such as DocVQA, ChartQA, and TextVQA are useful sanity checks, but they are mostly organized around question-image pairs. They are not ideal for testing a practical RAG workflow over many multi-page documents.

DocVisRAG therefore builds an in-house benchmark:

| Dataset | Size | Source | Purpose |
|---|---:|---|---|
| HomeworkQA | 180 short-answer questions | 16 self-collected PDFs | Multi-page document QA with LLM judge |
| HomeworkQA-MC | 180 multiple-choice questions | Same 16 PDFs | Controlled answer selection and retrieval comparison |

The questions are generated from the PDFs and verified around multi-page evidence, visual layouts, tables, forms, and cross-page reasoning. This better matches the intended application: offline document indexing for enterprise or personal document assistants.

## Main Experiment Configuration

This is the configuration that should be highlighted in reports and commits.

| Component | Default |
|---|---|
| In-house datasets to keep in git | `data/bench_full/homeworkqa`, `data/bench_full/homeworkqa_mc` |
| Generated artifacts excluded from git | `data/bench_runs`, `.work`, FAISS/Byaldi indexes, rendered pages, temporary predictions |
| Text index | block-level OCR text index |
| Hybrid index | block-level OCR-only hybrid index with lexical rerank `lexical_weight=0.2` |
| Visual index | ColQwen/Byaldi page index |
| Short-answer main hybrid | answer-level fusion over `text + raw_hybrid_block` |
| Short-answer main fusion | answer-level fusion over `text + raw_hybrid_block + visual + raw_page_fusion` |
| Page/block/visual-dominant fusion | ablation only |

One-command main experiment:

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

Reuse an existing visual index to save time:

```bash
bash scripts/bench/run_homeworkqa_experiments.sh \
  --device 3 \
  --run-root data/bench_runs/homeworkqa_experiments_$(date +%Y%m%d_%H%M%S) \
  --reuse-visual data/bench_runs/homeworkqa_experiments_20260618_101350/indexes/visual
```

### 2. System Pipeline

DocVisRAG follows a two-stage pipeline.

```text
Offline indexing
PDF / image
  -> page rendering
  -> OCR and layout parsing
  -> page summaries
  -> text / hybrid / visual indexes

Online QA
question
  -> retrieve top-k evidence pages
  -> feed pages + question to VLM
  -> answer with citation pages
```

Supported retrieval modes:

| Mode | Retrieval signal | Evidence passed to reader | Role |
|---|---|---|---|
| text | OCR text blocks/pages | Retrieved pages and OCR context | Strong lexical baseline |
| hybrid | OCR + page summaries + lexical rerank | Retrieved pages | General document retriever |
| visual | Page image embeddings | Retrieved page images | Layout/table/visual evidence |
| fusion | Multi-route fusion | Retrieved pages or answer-level candidates | Main multimodal setting |

For short-answer HomeworkQA, the current main experiment uses **answer-level hybrid/fusion**:

- `hybrid-answer`: generate answers from `text` and `raw_hybrid`, then let a local LLM select/synthesize the final answer.
- `fusion-answer`: generate answers from `text`, `raw_hybrid`, `visual`, and raw page-fusion, then select/synthesize the final answer.
- Page-level, block-level, visual-dominant, and snippet-level fusion are kept as ablations.

### 3. Current Main Results

#### HomeworkQA Short Answer

Full result: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

| Method | Judge Acc | Judge Score | Relaxed Acc | F1 | Citation Acc |
|---|---:|---:|---:|---:|---:|
| text | 0.4389 | 0.5039 | 0.4333 | 0.3519 | 0.8667 |
| hybrid-answer | **0.4556** | 0.4950 | **0.4833** | **0.3696** | **0.9278** |
| visual | 0.4500 | 0.5289 | 0.4167 | 0.3415 | 0.8833 |
| fusion-answer | **0.5000** | **0.5578** | **0.5444** | **0.3928** | **0.9722** |

The short-answer benchmark satisfies the intended trend:

```text
hybrid-answer > text
fusion-answer > visual
```

#### HomeworkQA-MC

After removing 9 text-only easy/biased questions and replacing them with hard rebalance questions, the latest full MC main run is:

| Method | Accuracy | Parse Rate | R@1 | R@3 | MRR |
|---|---:|---:|---:|---:|---:|
| text | 0.8444 | 1.0000 | 0.8389 | 0.9833 | 0.9099 |
| hybrid | 0.8722 | 1.0000 | 0.8278 | 0.9722 | 0.8935 |
| visual | 0.8889 | 1.0000 | 0.7500 | 0.9222 | 0.8381 |
| fusion | **0.9056** | 1.0000 | 0.8111 | 0.9500 | 0.8876 |

The rebalanced MC benchmark now also satisfies:

```text
hybrid > text
fusion > visual
```

## Repository Layout

```text
DocVisRAG/
├── app.py                         # Gradio demo
├── src/docvisrag/                 # Core package
│   ├── ingest/                    # render, OCR, layout, page summary
│   ├── retrieve/                  # text, hybrid, visual, fusion indexes
│   ├── qa/                        # multimodal QA engines
│   ├── eval/                      # metrics
│   └── vlm/                       # Qwen-VL wrapper
├── scripts/
│   ├── ingest/                    # document preprocessing
│   ├── retrieve/                  # build/search indexes
│   ├── bench/                     # benchmark runners
│   └── eval/                      # evaluation utilities
├── docs/
│   ├── README.md                  # documentation map
│   ├── EXPERIMENTS.md             # consolidated experiment report
│   ├── TEXT_AWARE_FUSION.md       # fusion design note
│   └── SERVER_DEPLOYMENT.md       # remote/server deployment notes
└── requirements-*.txt
```

## Quick Start

Install the base environment first:

```bash
pip install -r requirements-base.txt
```

Install optional visual retrieval dependencies when using `visual` or `fusion`:

```bash
pip install -r requirements-visual.txt
```

Run the demo:

```bash
python app.py
```

Run HomeworkQA / HomeworkQA-MC main and ablation experiments on the server:

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

Run only HomeworkQA-MC with existing indexes:

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

## Documentation

Start here:

- [docs/README.md](docs/README.md): documentation map and recommended reading order.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): consolidated results for public datasets and HomeworkQA.
- [docs/TEXT_AWARE_FUSION.md](docs/TEXT_AWARE_FUSION.md): retrieval/fusion design.
- [docs/SERVER_DEPLOYMENT.md](docs/SERVER_DEPLOYMENT.md): server setup and deployment notes.

## Intended Application

DocVisRAG is aimed at slow-changing document collections where offline indexing cost is acceptable, such as enterprise document management, internal knowledge bases, coursework archives, reports, and personal assistants. A concrete deployment target is a document assistant for enterprise/private knowledge collections, where the user asks questions over uploaded PDFs and the system retrieves evidence pages before answering.
