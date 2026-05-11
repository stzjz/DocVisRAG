# DocVisRAG

当前阶段：**阶段 6 - 端到端多模态 RAG 问答**

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

## 阶段 6：端到端多模态 RAG 问答

输入用户问题后，系统会自动：
1. 从 hybrid index 检索 top-k 页面；
2. 读取候选页面图像、页面摘要、OCR 文本；
3. 交给 VLM 生成带引用页码的答案。

```bash
python scripts/doc_qa.py \
  --index-dir data/indexes/demo_hybrid \
  --question "第 2 节的主要结论是什么？" \
  --top-k 3
```

可选参数：
- `--model-id`
- `--load-in-4bit`
- `--max-new-tokens`

输出包含：
- `答案：`
- `依据：`
- `引用：`
- `不确定性：`

其中引用按“第 X 页”格式输出。

## 当前阶段不包含
- ColPali 接入
- Web UI

## 注意
- 不要把模型权重提交进仓库。

## 阶段 7：Gradio Demo

运行：

```bash
python app.py
```

启动后打开：

```text
http://localhost:7860
```

Demo 功能：
- 上传 PDF/PNG/JPG/JPEG/WEBP。
- 点击“构建索引”自动执行：文档渲染 -> OCR -> 页面摘要 -> Hybrid 索引。
- 输入问题并点击“提问”，返回答案、依据、引用页码和不确定性。
- 页面过多时，Demo 默认只处理前 10 页并在日志提示。
