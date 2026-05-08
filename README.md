# DocVisRAG

当前阶段：**阶段 0 - 环境检查与项目骨架**

## 项目简介

DocVisRAG 是一个面向复杂 PDF、扫描件、PPT 截图等文档的多模态 RAG 问答系统项目。  
本阶段仅完成工程目录搭建与环境可用性检查。

## Docker 运行方式

你提供的容器启动命令如下：

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

## 如何运行环境检查

在项目根目录执行：

```bash
python scripts/check_env.py
```

脚本会检查并输出：
- Python 版本
- PyTorch 与 CUDA 可用性（含 GPU 名称）
- transformers
- qwen_vl_utils
- fitz / pymupdf
- PIL
- faiss
- paddleocr

## 阶段 0 不包含内容

当前阶段不包含以下能力：
- 模型推理
- PDF 解析
- OCR 处理
- RAG 检索与问答流程
- Web UI
