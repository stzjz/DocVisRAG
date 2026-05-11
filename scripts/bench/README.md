# bench scripts

## Purpose
- Prepare benchmark datasets in DocVisRAG format.
- Run end-to-end benchmark suite.
- In stage 9, optionally compare retriever types: `hybrid`, `visual`, `fusion`.

## Scripts
- `prepare_benchmark.py`
- `run_benchmark_suite.py`

## Usage
```bash
python scripts/bench/prepare_benchmark.py --benchmark docvqa --out-dir data/bench/docvqa

python scripts/bench/run_benchmark_suite.py \
  --name docvqa \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type hybrid
```

Optional visual/fusion run:
```bash
python scripts/bench/run_benchmark_suite.py \
  --name docvqa \
  --manifest data/bench/docvqa/manifest.json \
  --questions data/bench/docvqa/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type fusion \
  --visual-model-id vidore/colqwen2-v1.0
```
