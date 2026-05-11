# retrieve scripts

## Purpose
Build/search text and hybrid indexes.

## Scripts
- build_text_index.py
- text_search.py
- build_hybrid_index.py
- hybrid_search.py

## Usage
`ash
python scripts/retrieve/build_text_index.py --ocr data/outputs/demo_pages/ocr.jsonl --index-dir data/indexes/demo_text
python scripts/retrieve/text_search.py --index-dir data/indexes/demo_text --question  本文档主要内容是什么？ --top-k 5
python scripts/retrieve/build_hybrid_index.py --manifest data/outputs/demo_pages/manifest.json --ocr data/outputs/demo_pages/ocr.jsonl --summaries data/outputs/demo_pages/page_summaries.jsonl --index-dir data/indexes/demo_hybrid
python scripts/retrieve/hybrid_search.py --index-dir data/indexes/demo_hybrid --question 图表展示了什么趋势？ --top-k 3
`
