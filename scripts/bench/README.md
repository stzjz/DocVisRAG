# bench scripts

## Purpose
Benchmark dataset preparation and one-command benchmark suite execution.

## Scripts
- prepare_benchmark.py
- run_benchmark_suite.py

## Usage
`ash
python scripts/bench/prepare_benchmark.py --benchmark docvqa --out-dir data/bench/docvqa
python scripts/bench/run_benchmark_suite.py --name docvqa --manifest data/bench/docvqa/manifest.json --questions data/bench/docvqa/questions.jsonl --out-root data/bench_runs
`
