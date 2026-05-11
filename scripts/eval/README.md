# eval scripts

## Purpose
Retrieval/QA evaluation and error analysis.

## Scripts
- eval_retrieval.py
- eval_qa.py
- make_error_analysis.py

## Usage
`ash
python scripts/eval/eval_retrieval.py --questions eval/questions.example.jsonl --index-dir data/indexes/demo_hybrid --out data/outputs/eval_retrieval.json
python scripts/eval/eval_qa.py --questions eval/questions.example.jsonl --index-dir data/indexes/demo_hybrid --out data/outputs/predictions.jsonl --limit 10
python scripts/eval/make_error_analysis.py --predictions data/outputs/predictions.jsonl --out data/outputs/error_analysis.md
`
