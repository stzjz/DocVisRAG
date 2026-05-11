import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.docvisrag.ingest.render import load_manifest
from src.docvisrag.retrieve.base import BaseRetriever


@dataclass
class _HybridConfig:
    model_name: str
    dimension: int
    metric: str
    num_pages: int


class HybridPageIndex(BaseRetriever):
    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.index: Any = None
        self.metadata: List[Dict[str, Any]] = []
        self.model: Any = None

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object.")
                rows.append(row)
        return rows

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for HybridPageIndex embedding but is not available."
            ) from exc
        return SentenceTransformer(model_name)

    @staticmethod
    def _key(doc_id: str, page_index: int) -> Tuple[str, int]:
        return (str(doc_id), int(page_index))

    def build(
        self,
        manifest_path: str,
        ocr_jsonl: str | None,
        summary_jsonl: str,
        index_dir: str,
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        try:
            import faiss
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faiss is required for HybridPageIndex but is not available.") from exc

        manifest_file = Path(manifest_path).expanduser().resolve()
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        summary_file = Path(summary_jsonl).expanduser().resolve()
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary jsonl not found: {summary_file}")

        pages = load_manifest(str(manifest_file))
        if not pages:
            raise ValueError(f"Manifest has no pages: {manifest_file}")

        summary_rows = self._load_jsonl(str(summary_file))
        summary_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for row in summary_rows:
            key = self._key(row.get("doc_id", ""), int(row.get("page_index", -1)))
            summary_map[key] = row

        ocr_by_page: Dict[Tuple[str, int], List[str]] = {}
        if ocr_jsonl:
            ocr_file = Path(ocr_jsonl).expanduser().resolve()
            if not ocr_file.exists():
                raise FileNotFoundError(f"OCR jsonl not found: {ocr_file}")
            for row in self._load_jsonl(str(ocr_file)):
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                key = self._key(row.get("doc_id", ""), int(row.get("page_index", -1)))
                ocr_by_page.setdefault(key, []).append(text)

        page_texts: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for i, page in enumerate(pages):
            key = self._key(page.doc_id, page.page_index)
            summary = str(summary_map.get(key, {}).get("summary", "")).strip()
            ocr_text = " ".join(ocr_by_page.get(key, []))
            ocr_preview = ocr_text[:300]

            combined = (
                f"页面摘要：{summary}\n"
                f"页面OCR：{ocr_text}"
            ).strip()
            if not combined:
                combined = f"第 {page.page_index} 页"

            page_texts.append(combined)
            metadata.append(
                {
                    "id": i,
                    "doc_id": page.doc_id,
                    "page_index": page.page_index,
                    "image_path": page.image_path,
                    "summary": summary,
                    "ocr_text_preview": ocr_preview,
                    "score": 0.0,
                }
            )

        self.model_name = model_name
        self.model = self._load_model(model_name)
        embeddings = self.model.encode(page_texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        embeddings = self._normalize(embeddings)

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(out_dir / "index.faiss"))
        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        cfg = _HybridConfig(
            model_name=model_name,
            dimension=dim,
            metric="cosine_ip",
            num_pages=len(metadata),
        )
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = metadata

    @classmethod
    def load(cls, index_dir: str) -> "HybridPageIndex":
        try:
            import faiss
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faiss is required for HybridPageIndex but is not available.") from exc

        root = Path(index_dir).expanduser().resolve()
        index_file = root / "index.faiss"
        meta_file = root / "metadata.jsonl"
        cfg_file = root / "config.json"

        missing = [p for p in [index_file, meta_file, cfg_file] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Index directory is incomplete, missing: {missing}")

        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls()
        instance.model_name = str(cfg.get("model_name", "BAAI/bge-small-zh-v1.5"))
        instance.index = faiss.read_index(str(index_file))
        instance.metadata = instance._load_jsonl(str(meta_file))
        instance.model = instance._load_model(instance.model_name)
        return instance

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self.index is None or self.model is None:
            raise RuntimeError("HybridPageIndex is not loaded. Call load() or build() first.")

        q_emb = self.model.encode([query.strip()], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        q_emb = self._normalize(q_emb)

        k = min(top_k, len(self.metadata))
        scores, ids = self.index.search(q_emb, k)

        results: List[Dict[str, Any]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            row = dict(self.metadata[int(idx)])
            row["score"] = float(score)
            results.append(row)
        return results
