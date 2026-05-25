# eval scripts

## Purpose
- Retrieval evaluation
- QA evaluation
- Error analysis report generation
- Metrics aligned with the proposal: Recall@K, MRR, NDCG@5, citation accuracy, and latency

## Scripts
- `eval_retrieval.py`
- `eval_qa.py`
- `make_error_analysis.py`

## Usage
```bash
python scripts/eval/eval_retrieval.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --retriever-type hybrid \
  --out data/outputs/eval_retrieval.json

python scripts/eval/eval_qa.py \
  --questions eval/questions.example.jsonl \
  --index-dir data/indexes/demo_hybrid \
  --retriever-type hybrid \
  --out data/outputs/predictions.jsonl \
  --limit 10

python scripts/eval/make_error_analysis.py \
  --predictions data/outputs/predictions.jsonl \
  --out data/outputs/error_analysis.md
```

For `retriever-type visual/fusion`, pass `--visual-index-dir` when needed.
