import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.docvisrag.ingest.render import load_manifest


LOGGER = logging.getLogger(__name__)
VALID_OCR_BACKENDS = {"auto", "paddle", "tesseract"}


@dataclass
class OCRBlock:
    doc_id: str
    page_index: int
    text: str
    bbox: list[float]
    confidence: float

def _ocr_with_paddle(image_path: str, page_index: int, doc_id: str) -> List[OCRBlock]:
    # Run PaddleOCR in a subprocess to isolate hard crashes (e.g., SIGILL).
    code = r"""
import json, sys
from paddleocr import PaddleOCR

image_path = sys.argv[1]
ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
raw = ocr.ocr(image_path, cls=True)
lines = raw[0] if raw else []
out = []
for line in lines:
    if not line or len(line) < 2:
        continue
    points = line[0]
    text_info = line[1]
    if not text_info or len(text_info) < 2:
        continue
    text = str(text_info[0]).strip()
    if not text:
        continue
    conf = float(text_info[1])
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    out.append({
        "text": text,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "confidence": conf,
    })
print(json.dumps(out, ensure_ascii=False))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code, image_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"PaddleOCR subprocess failed (code={proc.returncode}). stderr: {proc.stderr.strip()}"
        )

    try:
        lines = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"PaddleOCR subprocess returned invalid JSON: {proc.stdout}") from exc

    blocks: List[OCRBlock] = []
    for line in lines:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        conf = float(line.get("confidence", 0.0))
        blocks.append(
            OCRBlock(
                doc_id=doc_id,
                page_index=page_index,
                text=text,
                bbox=[float(x) for x in line.get("bbox", [0.0, 0.0, 0.0, 0.0])],
                confidence=conf,
            )
        )
    return blocks


def _ocr_with_tesseract(image_path: str, page_index: int, doc_id: str) -> List[OCRBlock]:
    try:
        import pytesseract
        from pytesseract import Output
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Fallback OCR requires pytesseract and Pillow, but they are not available."
        ) from exc

    with Image.open(image_path) as img:
        data = pytesseract.image_to_data(img.convert("RGB"), output_type=Output.DICT)

    blocks: List[OCRBlock] = []
    total = len(data.get("text", []))
    for i in range(total):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        conf_raw = str(data["conf"][i]).strip()
        try:
            conf = float(conf_raw)
        except Exception:  # noqa: BLE001
            conf = 0.0
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        blocks.append(
            OCRBlock(
                doc_id=doc_id,
                page_index=page_index,
                text=text,
                bbox=[x, y, x + w, y + h],
                confidence=conf / 100.0 if conf > 1 else conf,
            )
        )
    return blocks


def resolve_ocr_backend(backend: str | None = None) -> str:
    value = (backend or os.getenv("DOCVISRAG_OCR_BACKEND", "auto")).strip().lower()
    if value not in VALID_OCR_BACKENDS:
        allowed = ", ".join(sorted(VALID_OCR_BACKENDS))
        raise ValueError(f"Invalid OCR backend: {value}. Supported: {allowed}")
    return value


def run_ocr_on_image(
    image_path: str,
    page_index: int,
    doc_id: str,
    backend: str | None = None,
) -> List[OCRBlock]:
    image_file = Path(image_path).expanduser()
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found for OCR: {image_file}")

    resolved_backend = resolve_ocr_backend(backend)

    if resolved_backend == "tesseract":
        blocks = _ocr_with_tesseract(str(image_file), page_index=page_index, doc_id=doc_id)
        LOGGER.info("Page %s OCR by pytesseract: %s blocks", page_index, len(blocks))
        return blocks

    if resolved_backend == "paddle":
        blocks = _ocr_with_paddle(str(image_file), page_index=page_index, doc_id=doc_id)
        LOGGER.info("Page %s OCR by PaddleOCR: %s blocks", page_index, len(blocks))
        return blocks

    try:
        blocks = _ocr_with_paddle(str(image_file), page_index=page_index, doc_id=doc_id)
        LOGGER.info("Page %s OCR by PaddleOCR: %s blocks", page_index, len(blocks))
        return blocks
    except Exception as paddle_exc:  # noqa: BLE001
        LOGGER.warning("PaddleOCR failed on page %s, fallback to pytesseract. Error: %s", page_index, paddle_exc)
        blocks = _ocr_with_tesseract(str(image_file), page_index=page_index, doc_id=doc_id)
        LOGGER.info("Page %s OCR by pytesseract: %s blocks", page_index, len(blocks))
        return blocks


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


def run_ocr_on_manifest(
    manifest_path: str,
    output_jsonl: str,
    backend: str | None = None,
) -> Dict[str, object]:
    pages = load_manifest(manifest_path)
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_backend = resolve_ocr_backend(backend)

    total_blocks = 0
    failed_pages: List[int] = []
    with out_file.open("w", encoding="utf-8") as f:
        for page in pages:
            image_path = _resolve_manifest_image(manifest_path, page.image_path)
            try:
                blocks = run_ocr_on_image(
                    image_path=str(image_path),
                    page_index=page.page_index,
                    doc_id=page.doc_id,
                    backend=resolved_backend,
                )
            except Exception:
                failed_pages.append(int(page.page_index))
                raise
            kept = 0
            for block in blocks:
                if not block.text.strip():
                    continue
                f.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")
                kept += 1
            total_blocks += kept
            LOGGER.info("Page %s saved OCR blocks: %s", page.page_index, kept)

    LOGGER.info("OCR completed. Total saved blocks: %s. Output: %s", total_blocks, out_file)
    return {
        "backend": resolved_backend,
        "num_pages": len(pages),
        "total_blocks": total_blocks,
        "failed_pages": failed_pages,
        "output_path": str(out_file),
    }
