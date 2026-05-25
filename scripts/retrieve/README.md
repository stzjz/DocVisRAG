# retrieve scripts

## Purpose
- Build/search text and hybrid indexes.
- Build/search visual indexes for the default fusion retrieval path.

## Scripts
- `build_text_index.py`
- `text_search.py`
- `build_hybrid_index.py`
- `hybrid_search.py`
- `build_visual_index.py`
- `visual_search.py`

## Usage
```bash
python scripts/retrieve/build_text_index.py --ocr data/outputs/demo_pages/ocr.jsonl --index-dir data/indexes/demo_text
python scripts/retrieve/text_search.py --index-dir data/indexes/demo_text --question "本文档主要内容是什么？" --top-k 5

python scripts/retrieve/build_hybrid_index.py --manifest data/outputs/demo_pages/manifest.json --ocr data/outputs/demo_pages/ocr.jsonl --summaries data/outputs/demo_pages/page_summaries.jsonl --index-dir data/indexes/demo_hybrid
python scripts/retrieve/hybrid_search.py --index-dir data/indexes/demo_hybrid --question "图表展示了什么趋势？" --top-k 3

python scripts/retrieve/build_visual_index.py --manifest data/outputs/demo_pages/manifest.json --index-dir data/indexes/demo_visual
python scripts/retrieve/visual_search.py --index-dir data/indexes/demo_visual --question "图表展示了什么趋势？" --top-k 3
```

## Notes
- Visual retrieval depends on Byaldi/ColPali related libraries.
- The Dockerfile installs visual dependencies by default. For a lightweight hybrid-only image, build with `--build-arg INSTALL_VISUAL=false`.
- If visual dependencies are unavailable, hybrid retrieval still works, but `visual` and full `fusion` evaluation cannot run.
