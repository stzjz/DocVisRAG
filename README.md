# DocVisRAG

当前阶段：**阶段 1 - 单图 VLM 问答流道**

## 项目简介
DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态 RAG 问答系统。  
当前阶段仅实现最小可用的单图问答链路：`图片 + 问题 -> Qwen2.5-VL 回答`。

## Docker 运行方式
```bash
docker run --gpus all --ipc=host --network=host -it --rm \
  -v /data3/zengjian/DocVisRAG:/workspace/DocVisRAG \
  -v /data3/zengjian/.cache:/root/.cache \
  -w /workspace/DocVisRAG \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  docvisrag:cu124
```

## 阶段 0：环境检查
```bash
python scripts/check_env.py
```

## 阶段 1：单图问答

### 1) 准备测试图片
把任意一张 PNG/JPG 图片放到：
```text
data/samples/test.png
```
或使用你自己的路径。

### 2) 运行单图问答
```bash
python scripts/vlm_qa.py \
  --image data/samples/test.png \
  --question "请概括这张图片的主要内容。"
```

### 3) 切换到 7B 模型
```bash
python scripts/vlm_qa.py \
  --image data/samples/test.png \
  --question "请概括这张图片的主要内容。" \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct
```

### 4) 使用 4bit 加载
```bash
python scripts/vlm_qa.py \
  --image data/samples/test.png \
  --question "请概括这张图片的主要内容。" \
  --load-in-4bit
```

### 5) 使用环境变量覆盖模型名
```bash
export DOCVISRAG_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
python scripts/vlm_qa.py \
  --image data/samples/test.png \
  --question "请概括这张图片的主要内容。"
```

## 当前阶段不包含
- PDF 渲染
- OCR
- RAG 检索与向量库
- Web UI
- 多轮对话

## 注意
- 不要把模型权重提交进仓库。
