#!/bin/bash
set -e
source ~/DocVisRAG/.venv/bin/activate
cd ~/DocVisRAG

export HF_HOME=/home/tangbaizhen-27/DocVisRAG/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

MODEL=Qwen/Qwen2.5-VL-7B-Instruct

echo "============================================"
echo "  Extra Datasets: ChartQA + TextVQA"
echo "  Model: $MODEL"
echo "  Started: $(date)"
echo "============================================"

# ═══ ChartQA: TEXT ═══
echo ""
echo "[$(date)] ChartQA TEXT (50q)"
python scripts/bench/run_benchmark_suite.py \
  --suite-name experiment_20260603 --name chartqa_text \
  --manifest data/bench/chartqa_50/manifest.json \
  --questions data/bench/chartqa_50/questions.jsonl \
  --out-root data/bench_runs --retriever-type text \
  --qa-model-id $MODEL --qa-top-k 3 --retrieval-top-k 5
echo "[$(date)] ChartQA TEXT DONE"

# ═══ ChartQA: HYBRID ═══
echo ""
echo "[$(date)] ChartQA HYBRID (50q)"
python scripts/bench/run_benchmark_suite.py \
  --suite-name experiment_20260603 --name chartqa_hybrid \
  --manifest data/bench/chartqa_50/manifest.json \
  --questions data/bench/chartqa_50/questions.jsonl \
  --out-root data/bench_runs --retriever-type hybrid \
  --summary-model-id $MODEL --qa-model-id $MODEL \
  --qa-top-k 3 --retrieval-top-k 5
echo "[$(date)] ChartQA HYBRID DONE"

echo ""
echo "ALL EXTRA EXPERIMENTS COMPLETE: $(date)"
