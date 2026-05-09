import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import fitz
from PIL import Image


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class PageImage:
    doc_id: str
    source_path: str
    page_index: int
    image_path: str
    width: int
    height: int


def _ensure_output_layout(output_dir: str) -> Path:
    out = Path(output_dir)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    return pages_dir


def make_doc_id(input_path: str) -> str:
    src = Path(input_path).expanduser().resolve()
    stem = src.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def render_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 180) -> List[PageImage]:
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}.")

    src = Path(pdf_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"PDF file does not exist: {src}")
    if src.suffix.lower() != ".pdf":
        raise ValueError(f"Input is not a PDF file: {src}")

    pages_dir = _ensure_output_layout(output_dir)
    doc_id = make_doc_id(str(src))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        doc = fitz.open(str(src))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to open PDF (possibly corrupted): {src}. Error: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise RuntimeError(f"PDF has no pages: {src}")

    pages: List[PageImage] = []
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_name = f"page_{i:03d}.png"
            image_file = pages_dir / image_name
            pix.save(str(image_file))

            pages.append(
                PageImage(
                    doc_id=doc_id,
                    source_path=str(src),
                    page_index=i,
                    image_path=str(image_file),
                    width=pix.width,
                    height=pix.height,
                )
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed while rendering PDF pages: {src}. Error: {exc}") from exc
    finally:
        doc.close()

    return pages


def copy_image_as_page(image_path: str, output_dir: str) -> List[PageImage]:
    src = Path(image_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Image file does not exist: {src}")
    if src.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTS))
        raise ValueError(f"Unsupported image format: {src.suffix}. Supported: {allowed}")

    pages_dir = _ensure_output_layout(output_dir)
    doc_id = make_doc_id(str(src))
    out_file = pages_dir / "page_001.png"

    try:
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            rgb.save(out_file, format="PNG")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to process image: {src}. Error: {exc}") from exc

    return [
        PageImage(
            doc_id=doc_id,
            source_path=str(src),
            page_index=1,
            image_path=str(out_file),
            width=width,
            height=height,
        )
    ]


def ingest_document(input_path: str, output_dir: str, dpi: int = 180) -> List[PageImage]:
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")

    suffix = src.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_to_images(str(src), output_dir, dpi=dpi)
    if suffix in SUPPORTED_IMAGE_EXTS:
        return copy_image_as_page(str(src), output_dir)

    allowed = [".pdf", *sorted(SUPPORTED_IMAGE_EXTS)]
    raise ValueError(
        f"Unsupported input format: {suffix}. "
        f"Please provide one of: {', '.join(allowed)}"
    )


def save_manifest(pages: List[PageImage], manifest_path: str) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(p) for p in pages]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: str) -> List[PageImage]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list of page items: {path}")

    pages: List[PageImage] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item #{idx} is not an object.")
        try:
            pages.append(PageImage(**item))
        except TypeError as exc:
            raise ValueError(f"Manifest item #{idx} has invalid schema: {item}") from exc
    return pages
