# Scripts Organization

Scripts are organized by function:
- env
- ingest
- retrieve
- qa
- eval
- bench

Run from project root inside the virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-base.txt
```

Optional stage-9 visual retrieval dependencies:

```bash
python -m pip install -r requirements-visual.txt
```

Run from project root, e.g.:
```bash
python scripts/env/check_env.py
```
