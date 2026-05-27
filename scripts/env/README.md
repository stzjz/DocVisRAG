# env scripts

Environment checks for Python, Torch/CUDA, Transformers, OCR, hybrid retrieval,
and optional stage-9 visual retrieval dependencies.

## Usage

```bash
python scripts/env/check_env.py
python scripts/env/check_env.py --visual
```

Run the visual check after installing `requirements-visual.txt` and before
building a visual index.

With the pinned Transformers-v4 visual stack, this line may be `NO`:

```text
peft.tp_helper          : NO
```

That is expected for `colpali-engine==0.3.13` and `peft==0.17.1`; the project
continues in relaxed compatibility mode. Set `DOCVISRAG_STRICT_VISUAL_CHECK=1`
only when validating a newer stack that is expected to provide the helper.
