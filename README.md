# DocVisRAG

当前阶段：**阶段 4 - OCR 与纯文本检索基线**

## 项目简介
DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态文档问答系统。  
当前阶段聚焦文档管线基础能力：页面渲染、页级问答、OCR 与纯文本检索。

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

输出结构：
```text
data/outputs/demo_pages/
  pages/
    page_001.png
    page_002.png
  manifest.json
```

## 阶段 3：页面级 VLM 问答
```bash
python scripts/page_qa.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --page 1 \
  --question "这一页主要讲了什么？"
```

输出中固定包含：
- `答案：`
- `引用：第 X 页`

## 阶段 4：OCR 与纯文本检索基线

### 1) 基于 manifest 运行 OCR
```bash
python scripts/run_ocr.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --out data/outputs/demo_pages/ocr.jsonl
```

说明：
- 优先使用 PaddleOCR
- PaddleOCR 初始化失败时自动 fallback 到 pytesseract
- 输出 `ocr.jsonl`，每行一个 OCR block（含 `doc_id/page_index/text/bbox/confidence`）

### 2) 从 OCR 构建文本向量索引
```bash
python scripts/build_text_index.py \
  --ocr data/outputs/demo_pages/ocr.jsonl \
  --index-dir data/indexes/demo_text
```

索引目录输出：
- `index.faiss`
- `metadata.jsonl`
- `config.json`

### 3) 纯文本检索
```bash
python scripts/text_search.py \
  --index-dir data/indexes/demo_text \
  --question "本文档的主要结论是什么？" \
  --top-k 5
```

返回字段包括：
- `text`
- `page_index`
- `bbox`
- `score`

## 当前阶段不包含
- 多模态检索融合
- 页面摘要生成
- 最终答案生成
- Web UI

## 注意
- 不要把模型权重提交进仓库。
