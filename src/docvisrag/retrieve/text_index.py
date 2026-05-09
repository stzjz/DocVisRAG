import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class _IndexConfig:
    model_name: str
    dimension: int
    metric: str
    num_vectors: int


class TextIndex:
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
        items: List[Dict[str, Any]] = []
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
                    raise ValueError(f"JSONL line {line_no} is not an object.")
                text = str(obj.get("text", "")).strip()
                if not text:
                    continue
                items.append(obj)
        return items

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required for TextIndex embedding but is not available."
            ) from exc
        return SentenceTransformer(model_name)

    def build_from_ocr_jsonl(
        self,
        ocr_jsonl: str,
        index_dir: str,
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        try:
            import faiss
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faiss is required for TextIndex but is not available.") from exc

        src = Path(ocr_jsonl).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"OCR jsonl not found: {src}")

        items = self._load_jsonl(str(src))
        if not items:
            raise ValueError(f"No valid OCR text blocks found in: {src}")

        self.model_name = model_name
        self.model = self._load_model(model_name)

        texts = [str(x["text"]).strip() for x in items]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        embeddings = self._normalize(embeddings)

        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        metadata: List[Dict[str, Any]] = []
        for i, item in enumerate(items):
            metadata.append(
                {
                    "id": i,
                    "doc_id": item.get("doc_id", ""),
                    "page_index": int(item.get("page_index", -1)),
                    "text": str(item.get("text", "")),
                    "bbox": item.get("bbox", []),
                    "confidence": float(item.get("confidence", 0.0)),
                    "score": 0.0,
                }
            )

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(out_dir / "index.faiss"))
        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        cfg = _IndexConfig(
            model_name=model_name,
            dimension=dim,
            metric="cosine_ip",
            num_vectors=len(metadata),
        )
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = metadata

    @classmethod
    def load(cls, index_dir: str) -> "TextIndex":
        try:
            import faiss
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faiss is required for TextIndex but is not available.") from exc

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

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self.index is None or self.model is None:
            raise RuntimeError("TextIndex is not loaded. Call load() or build_from_ocr_jsonl() first.")

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
