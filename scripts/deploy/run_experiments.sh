#!/bin/bash
set -e
source ~/DocVisRAG/.venv/bin/activate
cd ~/DocVisRAG

export HF_HOME=/home/tangbaizhen-27/DocVisRAG/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

echo "============================================"
echo "  DocVisRAG Benchmark Suite"
echo "  Model: Qwen/Qwen2.5-VL-7B-Instruct"
echo "  Started: $(date)"
echo "============================================"

# ── 1/3: TEXT baseline ──
echo ""
echo "[$(date)] 1/3: TEXT (DocVQA 100q)"
python scripts/bench/run_benchmark_suite.py \
  --suite-name experiment_20260603 --name docvqa_text \
  --manifest data/bench/docvqa_100/manifest.json \
  --questions data/bench/docvqa_100/questions.jsonl \
  --out-root data/bench_runs --retriever-type text \
  --qa-model-id Qwen/Qwen2.5-VL-7B-Instruct --qa-top-k 3 --retrieval-top-k 5
echo "[$(date)] TEXT DONE"

# ── 2/3: HYBRID ──
echo ""
echo "[$(date)] 2/3: HYBRID (DocVQA 100q)"
python scripts/bench/run_benchmark_suite.py \
  --suite-name experiment_20260603 --name docvqa_hybrid \
  --manifest data/bench/docvqa_100/manifest.json \
  --questions data/bench/docvqa_100/questions.jsonl \
  --out-root data/bench_runs --retriever-type hybrid \
  --summary-model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --qa-model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --qa-top-k 3 --retrieval-top-k 5
echo "[$(date)] HYBRID DONE"

# ── 3/3: FUSION ──
echo ""
echo "[$(date)] 3/3: FUSION (DocVQA 100q)"
python scripts/bench/run_benchmark_suite.py \
  --suite-name experiment_20260603 --name docvqa_fusion \
  --manifest data/bench/docvqa_100/manifest.json \
  --questions data/bench/docvqa_100/questions.jsonl \
  --out-root data/bench_runs --retriever-type fusion \
  --summary-model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --qa-model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --qa-top-k 3 --retrieval-top-k 5
echo "[$(date)] FUSION DONE"

echo ""
echo "ALL COMPLETE: $(date)"
