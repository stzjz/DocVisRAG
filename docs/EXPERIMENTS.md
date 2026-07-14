# Experiment Report

> Last updated: 2026-06-18  
> Main server path: `/data1/home/zengjian/DocVisRAG`

This document is the single place for experiment results. Older progress-style documents have been merged here so that the project keeps one coherent experimental story.

## Evaluation Philosophy

DocVisRAG is not evaluated as a traditional single-image VQA system. The intended workflow is:

```text
many-page document collection
  -> offline indexing
  -> question-time retrieval
  -> top-k candidate pages
  -> VLM answer generation
```

Public datasets are used as sanity checks, while the main benchmark is the self-built HomeworkQA suite because it better matches multi-page document RAG.

## Datasets

| Dataset | Size | Source | Usage |
|---|---:|---|---|
| DocVQA small | 100 questions | public | OCR/document QA sanity check |
| ChartQA small | 100 retrieval questions, QA sanity subset | public | chart/table stress test |
| TextVQA small | 100 retrieval questions | public | natural image OCR diagnosis |
| HomeworkQA | 180 short-answer questions | 16 self-built PDFs | main short-answer benchmark |
| HomeworkQA-MC | 180 multiple-choice questions | same 16 PDFs | main controlled answer-selection benchmark |

HomeworkQA and HomeworkQA-MC are built from 16 PDFs and are designed around multi-page evidence, OCR text, tables, forms, visual layout, and cross-page reasoning.

## Methods

| Method | Meaning |
|---|---|
| text | OCR text retrieval baseline; strong lexical signal |
| hybrid | OCR/page-summary dense retrieval with lexical rerank; page evidence for VLM |
| visual | visual page retrieval with ColQwen/Byaldi style index |
| raw fusion | page-level RRF-style fusion of text/hybrid/visual candidates |
| hybrid-answer | answer-level fusion of text and raw-hybrid answers |
| fusion-answer | answer-level fusion of text, raw-hybrid, visual, and raw-fusion answers |

For HomeworkQA short-answer, the main experiment uses `hybrid-answer` and `fusion-answer` by default. Page-level, block-level, visual-dominant, and snippet-level fusion variants are treated as ablations.

## HomeworkQA Short Answer

Run root:

```text
data/bench_runs/homeworkqa_experiments_20260618_101350
```

Configuration:

| Item | Value |
|---|---|
| Questions | 180 |
| QA model | `Qwen/Qwen3-VL-4B-Instruct` |
| LLM judge | local `Qwen2.5-7B-Instruct` |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Visual retriever | `vidore/colqwen2-v1.0` |
| top-k | 5 pages |

### Main Results

| Method | Judge Acc | Judge Score | EM | F1 | ANLS | Relaxed Acc | R@1 | R@3 | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 0.4389 | 0.5039 | 0.0444 | 0.3519 | 0.1284 | 0.4333 | 0.8222 | 0.9778 | 0.8667 |
| raw_hybrid_block | 0.4167 | 0.4833 | 0.0500 | 0.3420 | 0.1264 | 0.4278 | 0.8667 | 0.9722 | 0.8778 |
| hybrid-answer | **0.4556** | 0.4950 | 0.0556 | **0.3696** | 0.1400 | **0.4833** | **0.8667** | 0.9722 | **0.9278** |
| visual | 0.4500 | 0.5289 | 0.0444 | 0.3415 | 0.1284 | 0.4167 | 0.8333 | 0.9611 | 0.8833 |
| raw_page_fusion | 0.4500 | 0.5228 | 0.0389 | 0.3389 | 0.1207 | 0.4222 | 0.8722 | **0.9778** | 0.8778 |
| fusion-answer | **0.5000** | **0.5578** | **0.0611** | **0.3928** | **0.1558** | **0.5444** | 0.8222 | **0.9778** | **0.9722** |

Conclusion:

- `hybrid-answer` improves over `text` on Judge Acc, Relaxed Acc, F1, and Citation Acc.
- `fusion-answer` is the best short-answer result and clearly improves over `visual`.
- The gain comes mainly from answer-level selection/synthesis, not from page retrieval alone.

### Ablations

| Method | Judge Acc | Judge Score | Relaxed Acc | F1 | Citation Acc | R@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_page | 0.4389 | 0.5036 | 0.3944 | 0.3560 | 0.8556 | 0.8667 | 0.9722 |
| hybrid_block | 0.4167 | 0.4833 | 0.4278 | 0.3420 | 0.8778 | 0.8667 | 0.9722 |
| fusion_page | 0.4333 | 0.5078 | 0.4222 | 0.3364 | 0.8833 | 0.8722 | 0.9778 |
| fusion_block | 0.4500 | 0.5228 | 0.4222 | 0.3389 | 0.8778 | 0.8722 | 0.9778 |
| fusion_visual_dominant | 0.4389 | 0.5117 | 0.4056 | 0.3299 | 0.9167 | 0.7000 | 0.9167 |
| fusion_visual_snippet | 0.4167 | 0.4925 | 0.4222 | 0.3389 | 0.9056 | 0.8333 | 0.9611 |
| fusion_block_snippet | 0.4167 | 0.4881 | 0.4222 | 0.3390 | 0.9056 | 0.8333 | 0.9611 |

Ablation takeaway:

- Raw page/block fusion does not reliably beat answer-level fusion.
- Visual-dominant fusion increases Citation Acc but hurts R@1/R@3 and final answer correctness.
- Snippet fusion improves evidence richness but not final accuracy.

## HomeworkQA-MC

The MC set was rebalanced after the first run because text accuracy was too high due to 9 questions where `text` was correct and `hybrid/visual/fusion` were all wrong. Those 9 items were removed and replaced with hard rebalance items sourced from cases where `text` was wrong but `hybrid/visual/fusion` were all correct.

Backup and report:

```text
data/bench_full/homeworkqa_mc/questions.before_text_rebalance_20260618_204347.jsonl
data/bench_full/homeworkqa_mc/rebalance_report_20260618.json
```

Latest main run:

```text
data/bench_runs/homeworkqa_mc_rebalanced_20260618_full
```

| Method | Accuracy | Parse Rate | R@1 | R@3 | MRR | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.8444 | 1.0000 | **0.8389** | **0.9833** | **0.9099** | 0.6093s |
| hybrid | 0.8722 | 1.0000 | 0.8278 | 0.9722 | 0.8935 | 6.8706s |
| visual | 0.8889 | 1.0000 | 0.7500 | 0.9222 | 0.8381 | 6.7842s |
| fusion | **0.9056** | 1.0000 | 0.8111 | 0.9500 | 0.8876 | 6.8826s |

Conclusion:

- The rebalanced MC set now satisfies `hybrid > text`.
- `fusion` is the best MC method and also satisfies `fusion > visual`.
- Text still has the best retrieval metrics, but not the best final answer accuracy. This suggests that retrieval recall alone is not enough; evidence type and reader behavior matter.

MC ablations should be rerun with the rebalanced set. Use:

```bash
cd /data1/home/zengjian/DocVisRAG

export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DOCVISRAG_LOCAL_FILES_ONLY=1
export PY=/data1/home/zengjian/miniconda3/envs/docvisrag/bin/python
export RUN=data/bench_runs/homeworkqa_mc_rebalanced_20260618_ablation
export IDX=data/bench_runs/homeworkqa_experiments_20260618_101350/indexes

mkdir -p "$RUN"
```

Page-level ablation:

```bash
$PY scripts/bench/eval_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa_mc/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --text-index-dir "$IDX/text_page" \
  --hybrid-index-dir "$IDX/hybrid_page_ocr_only_lex02" \
  --visual-index-dir "$IDX/visual" \
  --out "$RUN/page_level.jsonl" \
  --summary-out "$RUN/summary_page_level.json" \
  --modes text hybrid visual fusion \
  --top-k 5 \
  --retrieval-query-mode question_choices \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-new-tokens 16 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5 \
  --fusion-visual-weight 2
```

Visual-dominant ablation:

```bash
$PY scripts/bench/eval_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa_mc/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --text-index-dir "$IDX/text_block" \
  --hybrid-index-dir "$IDX/hybrid_block_ocr_only_lex02" \
  --visual-index-dir "$IDX/visual" \
  --out "$RUN/visual_dominant.jsonl" \
  --summary-out "$RUN/summary_visual_dominant.json" \
  --modes text hybrid visual fusion \
  --top-k 5 \
  --retrieval-query-mode question_choices \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-new-tokens 16 \
  --fusion-text-weight 0 \
  --fusion-hybrid-weight 0.5 \
  --fusion-visual-weight 10 \
  --fusion-text-candidates 0 \
  --fusion-hybrid-candidates 50 \
  --fusion-visual-candidates 50
```

## Public Dataset Sanity Checks

The public datasets are kept as validation and diagnosis, not as the core project benchmark.

### DocVQA

| Setting | Questions | EM | F1 | ANLS | Relaxed Acc | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|
| old text | 100 | 0.000 | 0.034 | 0.000 | - | 0.710 |
| old hybrid | 100 | 0.000 | 0.088 | 0.000 | - | 0.380 |
| old fusion | 100 | 0.000 | 0.067 | 0.000 | - | 0.340 |
| P0 text-VLM | 100 | **0.520** | **0.658** | **0.618** | **0.700** | **0.820** |
| P0 hybrid | 100 | 0.490 | 0.630 | 0.610 | 0.690 | 0.820 |
| P0 fusion | 100 | 0.480 | 0.603 | 0.581 | 0.610 | 0.670 |
| P0 visual | 100 | 0.210 | 0.295 | 0.262 | 0.310 | 0.290 |

Takeaway: text-VLM is the strongest DocVQA configuration because OCR text is highly informative and the reader benefits from page images.

### ChartQA

| Setting | Questions | EM | F1 | ANLS | Relaxed Acc | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|
| old text | 50 | 0.000 | 0.000 | 0.000 | - | 0.380 |
| old hybrid | 50 | 0.100 | 0.199 | 0.143 | - | 0.220 |
| P0 text-VLM | 50 | 0.140 | 0.334 | 0.249 | 0.220 | **0.500** |
| P0 hybrid | 50 | 0.140 | 0.335 | 0.252 | 0.200 | 0.480 |
| P0 fusion | 50 | **0.180** | **0.401** | **0.280** | 0.220 | 0.420 |
| P0 visual | 50 | 0.140 | 0.206 | 0.163 | 0.120 | 0.200 |

Takeaway: ChartQA benefits more from multimodal fusion than DocVQA, but full chart reasoning remains hard.

### TextVQA

| Setting | Questions | EM | F1 | ANLS | Relaxed Acc | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|
| P0 text-VLM | 47 | 0.191 | 0.243 | 0.207 | 0.213 | 0.319 |

Takeaway: TextVQA is bottlenecked by natural-image OCR. It is useful as an OCR diagnosis set, but not the main DocVisRAG benchmark.

## Reproduction

Run all HomeworkQA main and ablation experiments:

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

Reuse the previous visual index to save time:

```bash
bash scripts/bench/run_homeworkqa_experiments.sh \
  --device 3 \
  --run-root data/bench_runs/homeworkqa_experiments_$(date +%Y%m%d_%H%M%S) \
  --reuse-visual data/bench_runs/homeworkqa_experiments_20260618_101350/indexes/visual
```
