# ingest scripts

## Purpose
Document ingestion/rendering, OCR, and VLM page summary generation.

## Scripts
- ingest_render.py
- run_ocr.py
- build_page_summaries.py

## Usage
`ash
python scripts/ingest/ingest_render.py --input data/samples/demo.pdf --output data/outputs/demo_pages --dpi 180
python scripts/ingest/run_ocr.py --manifest data/outputs/demo_pages/manifest.json --out data/outputs/demo_pages/ocr.jsonl
python scripts/ingest/build_page_summaries.py --manifest data/outputs/demo_pages/manifest.json --out data/outputs/demo_pages/page_summaries.jsonl
`
