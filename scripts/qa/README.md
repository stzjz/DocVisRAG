# qa scripts

## Purpose
VLM single-image QA, page QA, and end-to-end doc QA entrypoints.

## Scripts
- vlm_qa.py
- page_qa.py
- doc_qa.py

## Usage
`ash
python scripts/qa/vlm_qa.py --image data/samples/test.png --question 请概括这张图片的主要内容。
python scripts/qa/page_qa.py --manifest data/outputs/demo_pages/manifest.json --page 1 --question 这一页主要讲了什么？
python scripts/qa/doc_qa.py --index-dir data/indexes/demo_hybrid --question 第2节的主要结论是什么？ --top-k 3
`
