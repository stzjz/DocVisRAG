# DocVisRAG

当前阶段：**阶段 5 - 轻量多模态页面索引**

## 项目简介
DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态文档问答系统。  
当前阶段实现“页面摘要 + OCR 文本”的页面级检索基线，不生成最终答案。

## Docker 运行方式
```bash
docker run --gpus all --ipc=host --network=host -it --rm \
  -v /data3/zengjian/DocVisRAG:/workspace/DocVisRAG \
  -v /data3/zengjian/.cache:/root/.cache \
  -w /workspace/DocVisRAG \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_ENDPOINT=https://hf-mirror.com \
  docvisrag:cu124
```

## 阶段 0：环境检查
```bash
python scripts/check_env.py
```

## 阶段 1：单图 VLM 问答
```bash
python scripts/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "请概括这张图片的主要内容。"
```

## 阶段 2：文档摄入与页面渲染
```bash
python scripts/ingest_render.py \
  --input data/samples/demo.pdf \
  --output data/outputs/demo_pages \
  --dpi 180
```

## 阶段 3：页面级 VLM 问答
```bash
python scripts/page_qa.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --page 1 \
  --question "这一页主要讲了什么？"
```

## 阶段 4：OCR 与纯文本检索基线
```bash
python scripts/run_ocr.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/ocr.jsonl

python scripts/build_text_index.py \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --index-dir data/indexes/demo_text

python scripts/text_search.py \
  --index-dir data/indexes/demo_text \
  --question "本文档的主要结论是什么？" \
  --top-k 5
```

## 阶段 5：轻量多模态页面索引

### 1) 生成每页 VLM 摘要
```bash
python scripts/build_page_summaries.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/page_summaries.jsonl
```

输出 `page_summaries.jsonl`，每行包含：
- `doc_id`
- `page_index`
- `image_path`
- `summary`

### 2) 建立页面级混合索引（摘要 + OCR）
```bash
python scripts/build_hybrid_index.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --summaries data/outputs/demo_pages/page_summaries.jsonl \
  --index-dir data/indexes/demo_hybrid
```

索引目录输出：
- `index.faiss`
- `metadata.jsonl`
- `config.json`

### 3) 页面级检索
```bash
python scripts/hybrid_search.py \
  --index-dir data/indexes/demo_hybrid \
  --question "图表展示了什么趋势？" \
  --top-k 3
```

返回字段包含：
- `page_index`
- `image_path`
- `summary`
- `ocr_text_preview`
- `score`

## 当前阶段不包含
- ColPali 接入
- Web UI
- 最终问答生成

## 注意
- 不要把模型权重提交进仓库。
