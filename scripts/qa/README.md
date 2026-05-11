# qa scripts

## Purpose
- Single-image VLM QA
- Page QA
- End-to-end DocQA with selectable retriever

## Scripts
- `vlm_qa.py`
- `page_qa.py`
- `doc_qa.py`

## Usage
```bash
python scripts/qa/vlm_qa.py --image data/samples/test.png --question "请概括这张图片的主要内容。"

python scripts/qa/page_qa.py --manifest data/outputs/demo_pages/manifest.json --page 1 --question "这一页主要讲了什么？"

python scripts/qa/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --question "第2节的主要结论是什么？" \
  --retriever-type hybrid \
  --top-k 3
```

For `--retriever-type visual/fusion`, add `--visual-index-dir` if it cannot be inferred.
