# Scripts Organization

Scripts are organized by function:
- env
- ingest
- retrieve
- qa
- eval
- bench

Run from project root inside the configured conda environment:

```bash
cd /data1/home/zengjian/DocVisRAG
source /data1/home/zengjian/miniconda3/bin/activate docvisrag
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-base.txt
```

Optional stage-9 visual retrieval dependencies:

```bash
python -m pip install -r requirements-visual.txt
```

End-to-end loopback validation:

```bash
bash scripts/loopback_test.sh
```

Common overrides:

```bash
QUESTION_TEXT="请总结这页的核心结论" \
INPUT_REL=data/samples/DeepSeekMath.pdf \
RUN_TEXT_BASELINE=1 \
RUN_VISUAL=1 \
RUN_BENCHMARK=1 \
bash scripts/loopback_test.sh
```

The loopback script also writes a compact run summary to
`data/outputs/loopback_smoke/summary.json`.

Run from project root, e.g.:
```bash
python scripts/env/check_env.py
```
