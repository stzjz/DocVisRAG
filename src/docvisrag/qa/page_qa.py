from pathlib import Path
from typing import Dict, Optional

from src.docvisrag.ingest import load_manifest
from src.docvisrag.vlm import QwenVLClient


class PageQAEngine:
    def __init__(self, model_id: Optional[str] = None, load_in_4bit: bool = False):
        self.model_id = model_id or "Qwen/Qwen2.5-VL-3B-Instruct"
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = 512
        self.vlm = QwenVLClient(
            model_id=self.model_id,
            load_in_4bit=self.load_in_4bit,
        )

    def _resolve_page_image(self, manifest_path: str, image_path: str) -> Path:
        img = Path(image_path)
        if img.is_absolute() and img.exists():
            return img

        manifest_file = Path(manifest_path).expanduser().resolve()
        candidate_from_manifest_parent = manifest_file.parent / img
        if candidate_from_manifest_parent.exists():
            return candidate_from_manifest_parent.resolve()

        candidate_from_cwd = (Path.cwd() / img).resolve()
        if candidate_from_cwd.exists():
            return candidate_from_cwd

        raise FileNotFoundError(
            "Image path from manifest does not exist.\n"
            f"manifest: {manifest_file}\n"
            f"image_path field: {image_path}"
        )

    def answer_page(self, manifest_path: str, page: int, question: str) -> Dict[str, str]:
        if page <= 0:
            raise ValueError(f"page must be >= 1, got {page}.")
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        try:
            pages = load_manifest(manifest_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Manifest format is invalid: {manifest_path}. Error: {exc}") from exc

        if not pages:
            raise ValueError(f"Manifest has no page records: {manifest_path}")

        selected = next((p for p in pages if p.page_index == page), None)
        if selected is None:
            available = sorted({p.page_index for p in pages})
            raise ValueError(
                f"Requested page {page} does not exist in manifest. "
                f"Available pages: {available}"
            )

        resolved_image = self._resolve_page_image(manifest_path, selected.image_path)
        citation = f"第 {page} 页"
        prompt = (
            "请仅基于当前页面图像回答问题，不要使用页面外信息。\n"
            f"问题：{question.strip()}"
        )
        answer_text = self.vlm.answer_image(
            image_path=str(resolved_image),
            question=prompt,
            max_new_tokens=self.max_new_tokens,
        )

        return {
            "answer": answer_text,
            "page": page,
            "image_path": str(resolved_image),
            "citation": citation,
        }
