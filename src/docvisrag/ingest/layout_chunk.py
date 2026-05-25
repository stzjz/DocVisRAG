import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.docvisrag.ingest.layout import LayoutAnalyzer, load_layout_jsonl

LOGGER = logging.getLogger(__name__)

CHUNKABLE_TYPES = {"text", "title", "table", "formula", "list", "header", "footer"}


@dataclass
class LayoutChunk:
    chunk_id: str
    doc_id: str
    page_index: int
    chunk_type: str
    text: str
    bbox: list[float]
    confidence: float


class LayoutChunker:
    """Build layout-aware chunks from enriched layout regions."""

    def build_chunks_from_enriched(
        self,
        enriched_jsonl: str,
        output_jsonl: str,
        min_text_length: int = 10,
    ) -> Dict[str, Any]:
        regions = LayoutAnalyzer._load_jsonl(enriched_jsonl)
        out_file = Path(output_jsonl)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        chunks: List[dict] = []
        for i, region in enumerate(regions):
            rtype = str(region.get("region_type", "text"))
            if rtype not in CHUNKABLE_TYPES:
                continue
            text = str(region.get("text", "")).strip()
            if len(text) < min_text_length:
                continue

            chunk_id = f"{region.get('doc_id', 'doc')}-p{region.get('page_index', 0)}-{i}"
            chunks.append(asdict(LayoutChunk(
                chunk_id=chunk_id,
                doc_id=str(region.get("doc_id", "")),
                page_index=int(region.get("page_index", -1)),
                chunk_type=rtype,
                text=text,
                bbox=[float(v) for v in region.get("bbox", [0, 0, 0, 0])],
                confidence=float(region.get("confidence", 0.0)),
            )))

        with out_file.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        LOGGER.info("Layout chunks built: %s chunks → %s", len(chunks), out_file)
        return {"total_chunks": len(chunks), "output_path": str(out_file)}

    def build_chunks_from_layout(
        self,
        layout_jsonl: str,
        ocr_jsonl: str,
        output_jsonl: str,
        min_text_length: int = 10,
    ) -> Dict[str, Any]:
        """Build layout-aware chunks by assigning OCR text to layout regions.

        This is a convenience method that runs OCR assignment + chunking in one call.
        """
        analyzer = LayoutAnalyzer()
        enriched_path = Path(output_jsonl).with_suffix(".enriched.jsonl")
        analyzer.assign_ocr_to_regions(
            ocr_jsonl=ocr_jsonl,
            layout_jsonl=layout_jsonl,
            output_jsonl=str(enriched_path),
        )
        return self.build_chunks_from_enriched(
            enriched_jsonl=str(enriched_path),
            output_jsonl=output_jsonl,
            min_text_length=min_text_length,
        )
