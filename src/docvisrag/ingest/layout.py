import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.docvisrag.ingest.render import load_manifest

LOGGER = logging.getLogger(__name__)

VALID_LAYOUT_BACKENDS = {"auto", "ppstructure", "opencv"}

REGION_TYPES = {
    "text", "title", "table", "chart", "figure",
    "formula", "header", "footer", "page_number", "list", "unknown",
}

# PP-Structure label -> our region_type
_PPSTRUCTURE_LABEL_MAP = {
    "text": "text",
    "title": "title",
    "figure": "figure",
    "table": "table",
    "list": "text",
    "reference": "text",
    "equation": "formula",
    "formula": "formula",
}


@dataclass
class LayoutRegion:
    region_id: int
    doc_id: str
    page_index: int
    region_type: str
    bbox: list[float]
    confidence: float
    text: str = ""
    image_crop_path: str = ""

    def area(self) -> float:
        w = max(0.0, self.bbox[2] - self.bbox[0])
        h = max(0.0, self.bbox[3] - self.bbox[1])
        return w * h


def _resolve_layout_backend(backend: str | None = None) -> str:
    value = (backend or os.getenv("DOCVISRAG_LAYOUT_BACKEND", "auto")).strip().lower()
    if value not in VALID_LAYOUT_BACKENDS:
        raise ValueError(f"Invalid layout backend: {value}. Supported: {', '.join(sorted(VALID_LAYOUT_BACKENDS))}")
    return value


def _run_ppstructure(image_path: str, page_index: int, doc_id: str) -> List[LayoutRegion]:
    """Run PP-Structure in a subprocess for layout detection."""
    code = r"""
import json, sys
image_path = sys.argv[1]

try:
    from paddleocr import PPStructure
    engine = PPStructure(
        table=True, layout=True, show_log=False,
        image_orientation=False, lang="ch",
    )
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"cv2.imread returned None for: {image_path}")
    h, w = img.shape[:2]
    results = engine(img)
except Exception as exc:
    print(json.dumps({"error": str(exc)}, ensure_ascii=False))
    sys.exit(1)

out = []
for i, region in enumerate(results):
    raw_type = str(region.get("type", "text"))
    bbox_raw = region.get("bbox", [0, 0, 0, 0])
    if len(bbox_raw) != 4:
        continue
    # normalize bbox to 0-1
    x0, y0, x1, y1 = [float(v) for v in bbox_raw]
    x0_n = max(0.0, min(1.0, x0 / w))
    y0_n = max(0.0, min(1.0, y0 / h))
    x1_n = max(0.0, min(1.0, x1 / w))
    y1_n = max(0.0, min(1.0, y1 / h))
    bbox_norm = [round(v, 6) for v in (x0_n, y0_n, x1_n, y1_n)]

    # table-specific: extract HTML or cell text
    cell_text = ""
    if raw_type == "table" and "res" in region:
        res = region["res"]
        if isinstance(res, dict):
            cell_text = str(res.get("html", ""))
        elif isinstance(res, str):
            cell_text = res

    out.append({
        "region_type": raw_type,
        "bbox": bbox_norm,
        "confidence": float(region.get("confidence", 0.9)),
        "text": cell_text,
        "raw_label": raw_type,
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
            f"PP-Structure subprocess failed (code={proc.returncode}). stderr: {proc.stderr.strip()}"
        )

    stdout = proc.stdout.strip()
    raw_regions: Any = None
    if stdout:
        # Paddle may print model-download progress before the JSON payload on first run.
        for line in reversed(stdout.splitlines()):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                raw_regions = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if raw_regions is None:
        raise RuntimeError(f"PP-Structure returned invalid JSON: {proc.stdout}")

    if isinstance(raw_regions, dict) and "error" in raw_regions:
        raise RuntimeError(f"PP-Structure error: {raw_regions['error']}")

    regions: List[LayoutRegion] = []
    for i, rr in enumerate(raw_regions):
        raw_type = str(rr.get("region_type", rr.get("raw_label", "text")))
        mapped = _PPSTRUCTURE_LABEL_MAP.get(raw_type, "text")
        if mapped == "text":
            mapped = _refine_text_region_type(
                rr.get("bbox", [0, 0, 0, 0]),
                raw_type,
            )
        bbox = [float(v) for v in rr.get("bbox", [0, 0, 0, 0])]
        regions.append(LayoutRegion(
            region_id=i,
            doc_id=doc_id,
            page_index=page_index,
            region_type=mapped,
            bbox=bbox,
            confidence=float(rr.get("confidence", 0.9)),
            text=str(rr.get("text", "")),
        ))
    return regions


def _refine_text_region_type(bbox: list[float], raw_type: str) -> str:
    """Apply heuristics to refine region type for text-like blocks."""
    y0, y1 = bbox[1], bbox[3]
    h = y1 - y0

    # top 5% of page → header
    if y1 < 0.05:
        return "header"
    # bottom 5% → footer
    if y0 > 0.95:
        return "footer"
    # very small at bottom → page number
    if y0 > 0.92 and h < 0.04:
        return "page_number"
    # title is large and near top
    if raw_type == "title":
        return "title"
    return "text"


def _run_opencv_layout(image_path: str, page_index: int, doc_id: str) -> List[LayoutRegion]:
    """Simple OpenCV-based layout detection as fallback.

    Uses morphological operations to find text blocks and separate
    figures/tables from text regions by contour analysis.
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV fallback requires opencv-python-headless.") from exc

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8,
    )

    # Dilate to merge nearby text into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[LayoutRegion] = []
    region_id = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        img_area = w * h

        # filter tiny noise
        if area < img_area * 0.001 or cw < 10 or ch < 10:
            continue

        # normalize bbox
        bbox = [
            round(x / w, 6),
            round(y / h, 6),
            round((x + cw) / w, 6),
            round((y + ch) / h, 6),
        ]

        # classify region
        aspect = cw / max(ch, 1)
        density = float(np.sum(binary[y:y+ch, x:x+cw]) / 255.0) / max(area, 1)

        region_type = _classify_opencv_region(
            bbox=bbox, aspect=aspect, density=density,
            img_area=img_area, area=area, w=w, h=h,
        )

        if region_type == "unknown":
            continue

        regions.append(LayoutRegion(
            region_id=region_id,
            doc_id=doc_id,
            page_index=page_index,
            region_type=region_type,
            bbox=bbox,
            confidence=0.7,
        ))
        region_id += 1

    return regions


def _classify_opencv_region(
    bbox: list[float],
    aspect: float,
    density: float,
    img_area: float,
    area: float,
    w: int,
    h: int,
) -> str:
    """Classify an OpenCV-detected region by heuristic rules."""
    y0, y1 = bbox[1], bbox[3]
    bh = y1 - y0

    # header region: top 6%
    if y1 < 0.06:
        return "header"
    # footer: bottom 6%
    if y0 > 0.94:
        return "footer"
    # page number: bottom 5%, small height
    if y0 > 0.90 and bh < 0.05:
        return "page_number"

    # large region with low text density → figure or chart
    area_ratio = area / img_area
    if area_ratio > 0.1:
        if density < 0.05:
            return "figure"
        if density < 0.12:
            return "chart"

    # wide and short → could be a table
    if aspect > 4 and bh > 0.03:
        return "table"

    # very wide, very short → likely a separator line, skip
    if aspect > 20 and bh < 0.015:
        return "unknown"

    return "text"


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
    raise FileNotFoundError(f"Image cannot be resolved: {image_path}")


class LayoutAnalyzer:
    """Analyze document page images to detect semantic layout regions."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = _resolve_layout_backend(backend)
        self._available_backend: Optional[str] = None

    def _detect_backend(self) -> str:
        if self._available_backend is not None:
            return self._available_backend

        if self.backend == "ppstructure":
            self._available_backend = "ppstructure"
        elif self.backend == "opencv":
            self._available_backend = "opencv"
        else:
            if self._ppstructure_available():
                self._available_backend = "ppstructure"
            else:
                LOGGER.warning("PP-Structure not available, falling back to OpenCV layout detection.")
                self._available_backend = "opencv"

        LOGGER.info("Layout backend resolved: %s", self._available_backend)
        return self._available_backend

    @staticmethod
    def _ppstructure_available() -> bool:
        try:
            code = "from paddleocr import PPStructure; print('ok')"
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, check=False,
            )
            return proc.returncode == 0 and "ok" in proc.stdout
        except Exception:
            return False

    def analyze_page(
        self, image_path: str, page_index: int, doc_id: str,
    ) -> List[LayoutRegion]:
        image_file = Path(image_path).expanduser()
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        backend = self._detect_backend()
        if backend == "ppstructure":
            return _run_ppstructure(str(image_file), page_index, doc_id)
        return _run_opencv_layout(str(image_file), page_index, doc_id)

    def analyze_manifest(
        self, manifest_path: str, output_jsonl: str,
    ) -> Dict[str, Any]:
        pages = load_manifest(manifest_path)
        out_file = Path(output_jsonl)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        total_regions = 0
        failed_pages: List[int] = []

        with out_file.open("w", encoding="utf-8") as f:
            for page in pages:
                image_path = _resolve_manifest_image(manifest_path, page.image_path)
                try:
                    regions = self.analyze_page(
                        image_path=str(image_path),
                        page_index=page.page_index,
                        doc_id=page.doc_id,
                    )
                except Exception:
                    failed_pages.append(int(page.page_index))
                    LOGGER.exception("Layout analysis failed on page %s", page.page_index)
                    continue

                for region in regions:
                    f.write(json.dumps(asdict(region), ensure_ascii=False) + "\n")
                total_regions += len(regions)
                LOGGER.info("Page %s layout regions: %s", page.page_index, len(regions))

        LOGGER.info("Layout analysis completed. Total regions: %s. Output: %s", total_regions, out_file)
        return {
            "backend": self._available_backend or self.backend,
            "num_pages": len(pages),
            "total_regions": total_regions,
            "failed_pages": failed_pages,
            "output_path": str(out_file),
        }

    def assign_ocr_to_regions(
        self,
        ocr_jsonl: str,
        layout_jsonl: str,
        output_jsonl: str,
    ) -> Dict[str, Any]:
        """Assign OCR text blocks to layout regions by bbox overlap.

        Produces enriched layout regions with aggregated OCR text.
        """
        ocr_blocks = LayoutAnalyzer._load_jsonl(ocr_jsonl)
        layout_regions = LayoutAnalyzer._load_jsonl(layout_jsonl)

        # index OCR blocks by (page_index)
        ocr_by_page: Dict[int, List[dict]] = {}
        for block in ocr_blocks:
            pi = int(block.get("page_index", -1))
            ocr_by_page.setdefault(pi, []).append(block)

        out_file = Path(output_jsonl)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        enriched_count = 0
        with out_file.open("w", encoding="utf-8") as f:
            for region in layout_regions:
                pi = int(region.get("page_index", -1))
                r_bbox = region.get("bbox", [0, 0, 0, 0])
                page_ocr = ocr_by_page.get(pi, [])

                page_width = max(
                    (
                        max(float(v) for v in block.get("bbox", [0.0, 0.0, 0.0, 0.0])[0::2])
                        for block in page_ocr
                    ),
                    default=1.0,
                )
                page_height = max(
                    (
                        max(float(v) for v in block.get("bbox", [0.0, 0.0, 0.0, 0.0])[1::2])
                        for block in page_ocr
                    ),
                    default=1.0,
                )

                # collect OCR blocks that overlap with this region
                assigned_texts: List[str] = []
                for block in page_ocr:
                    o_bbox = _normalize_bbox_to_unit(
                        block.get("bbox", [0, 0, 0, 0]),
                        width=page_width,
                        height=page_height,
                    )
                    if _iou(r_bbox, o_bbox) > 0.1:
                        assigned_texts.append(str(block.get("text", "")))

                region["text"] = " ".join(assigned_texts)
                f.write(json.dumps(region, ensure_ascii=False) + "\n")
                enriched_count += 1

        return {
            "total_regions": enriched_count,
            "output_path": str(out_file),
        }

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        items: List[dict] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {line_no} is not an object.")
                items.append(obj)
        return items


def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Calculate IoU between two bounding boxes [x0, y0, x1, y1]."""
    x_left = max(bbox_a[0], bbox_b[0])
    y_top = max(bbox_a[1], bbox_b[1])
    x_right = min(bbox_a[2], bbox_b[2])
    y_bottom = min(bbox_a[3], bbox_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(0.0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalize_bbox_to_unit(bbox: list[float], width: float, height: float) -> list[float]:
    if len(bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    try:
        coords = [float(v) for v in bbox]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]

    # Layout regions are already normalized to 0-1. OCR boxes are pixel-space.
    if max(coords) <= 1.0 and min(coords) >= 0.0:
        return coords

    x0, y0, x1, y1 = coords
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return [
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    ]


def load_layout_jsonl(path: str) -> List[LayoutRegion]:
    """Load layout regions from a JSONL file."""
    items = LayoutAnalyzer._load_jsonl(path)
    regions: List[LayoutRegion] = []
    for item in items:
        try:
            regions.append(LayoutRegion(**item))
        except TypeError as exc:
            LOGGER.warning("Skipping invalid layout region: %s — %s", item, exc)
    return regions
