import json
import logging
from pathlib import Path
from typing import Dict

from src.docvisrag.ingest.render import load_manifest
from src.docvisrag.vlm import QwenVLClient


LOGGER = logging.getLogger(__name__)


SUMMARY_PROMPT = (
    "请阅读这页文档图像，用中文概括页面内容。重点描述：\n"
    "1. 标题和主题\n"
    "2. 主要段落内容\n"
    "3. 表格、图表、公式或图片区域\n"
    "4. 可能适合回答的问题类型\n"
    "要求 100-200 字，不要编造看不见的信息。"
)


def _normalize_summary(text: str) -> str:
    summary = (text or "").strip()
    if len(summary) > 220:
        summary = summary[:200].rstrip()
    return summary


def _resolve_manifest_image(manifest_path: str, image_path: str) -> Path:
    manifest_file = Path(manifest_path).expanduser().resolve()
    img = Path(image_path)

    if img.is_absolute() and img.exists():
        return img

    candidate = (manifest_file.parent / img).resolve()
    if candidate.exists():
        return candidate

    candidate_cwd = (Path.cwd() / img).resolve()
    if candidate_cwd.exists():
        return candidate_cwd

    raise FileNotFoundError(
        f"Image in manifest cannot be resolved: {image_path} (manifest: {manifest_file})"
    )


def summarize_page_with_vlm(
    image_path: str,
    page_index: int,
    model_id: str | None = None,
    load_in_4bit: bool = False,
) -> str:
    model = model_id or "Qwen/Qwen2.5-VL-3B-Instruct"
    client = QwenVLClient(model_id=model, load_in_4bit=load_in_4bit)
    question = f"{SUMMARY_PROMPT}\n当前是第 {page_index} 页。"
    summary = client.answer_image(image_path=image_path, question=question, max_new_tokens=320)
    return _normalize_summary(summary)


def build_page_summaries(
    manifest_path: str,
    output_jsonl: str,
    model_id: str | None = None,
    load_in_4bit: bool = False,
) -> None:
    pages = load_manifest(manifest_path)
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = model_id or "Qwen/Qwen2.5-VL-3B-Instruct"
    client = QwenVLClient(model_id=model, load_in_4bit=load_in_4bit)

    with out_file.open("w", encoding="utf-8") as f:
        for page in pages:
            image_path = _resolve_manifest_image(manifest_path, page.image_path)
            question = f"{SUMMARY_PROMPT}\n当前是第 {page.page_index} 页。"
            summary = client.answer_image(
                image_path=str(image_path),
                question=question,
                max_new_tokens=320,
            )
            summary = _normalize_summary(summary)

            row: Dict[str, object] = {
                "doc_id": page.doc_id,
                "page_index": page.page_index,
                "image_path": str(image_path),
                "summary": summary,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            LOGGER.info("Page %s summary generated (%s chars).", page.page_index, len(summary))

    LOGGER.info("Page summaries saved to: %s", out_file)
