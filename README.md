# DocVisRAG

当前阶段：**阶段 3 - 页面级 VLM 文档问答**

## 项目简介
DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态 RAG 问答系统。  
当前阶段实现文档摄入与页面渲染：将 PDF/图片输入转换为页面图像，并输出 `manifest.json`。

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

## 阶段 1：单图问答
```bash
python scripts/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "请概括这张图片的主要内容。"
```

切换 7B：
```bash
python scripts/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "请概括这张图片的主要内容。" \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct
```

4bit 加载：
```bash
python scripts/vlm_qa.py \
  --image data/samples/test_stage1.png \
  --question "请概括这张图片的主要内容。" \
  --load-in-4bit
```

## 阶段 2：文档摄入与页面渲染

### 输入支持
- PDF：`.pdf`
- 图片：`.png` `.jpg` `.jpeg` `.webp`

### 命令
```bash
python scripts/ingest_render.py \
  --input data/samples/demo.pdf \
  --output data/outputs/demo_pages \
  --dpi 180
```

### 输出结构
```text
data/outputs/demo_pages/
  pages/
    page_001.png
    page_002.png
  manifest.json
```

`manifest.json` 记录每页的：
- `doc_id`
- `source_path`
- `page_index`（从 1 开始）
- `image_path`
- `width`
- `height`

### 单图输入示例
```bash
python scripts/ingest_render.py \
  --input data/samples/1444.jpg \
  --output data/outputs/demo_image_pages
```

## 当前阶段不包含
- OCR
- RAG 检索与向量库
- Web UI

## 阶段 3：页面级问答

基于阶段 2 产出的 `manifest.json` 和页面图像，针对指定页提问。

### 命令
```bash
python scripts/page_qa.py \
  --manifest data/outputs/demo_pages/manifest.json \
  --page 1 \
  --question "这一页主要讲了什么？"
```

可选参数：
- `--model-id`
- `--load-in-4bit`
- `--max-new-tokens`

输出固定包含：
- `答案：`
- `引用：第 X 页`

说明：
- 本阶段不做自动检索页面，只回答你指定的页码。

## 注意
- 不要把模型权重提交进仓库。
