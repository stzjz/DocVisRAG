"""BM25 稀疏检索索引。

在 OCR 文本块上构建 BM25 倒排索引，作为 Dense (FAISS) 检索的补充通道。
BM25 对精确关键词匹配（数字、专有名词、日期）有天然优势，
弥补 Dense Embedding 的语义漂移问题。
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.docvisrag.retrieve.base import BaseRetriever


@dataclass
class _BM25Config:
    index_type: str = "bm25"
    num_docs: int = 0


class BM25Index(BaseRetriever):
    """基于 rank_bm25 的 BM25 稀疏检索索引。"""

    def __init__(self) -> None:
        self.index: Any = None
        self.metadata: List[Dict[str, Any]] = []
        self._texts: List[str] = []

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
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = str(obj.get("text", "")).strip()
                if not text:
                    continue
                items.append(obj)
        return items

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        tokens: List[str] = []
        for part in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]", text.lower()):
            if part:
                tokens.append(part)
        return tokens

    def build_from_ocr_jsonl(self, ocr_jsonl: str, index_dir: str) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RuntimeError("BM25Index requires rank_bm25. Install with: pip install rank-bm25")

        src = Path(ocr_jsonl).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"OCR jsonl not found: {src}")

        items = self._load_jsonl(str(src))
        if not items:
            raise ValueError(f"No valid OCR text blocks in: {src}")

        self._texts = [str(x["text"]).strip() for x in items]
        tokenized = [self._tokenize(t) for t in self._texts]
        self.index = BM25Okapi(tokenized)

        self.metadata = []
        for i, item in enumerate(items):
            self.metadata.append({
                "id": i, "doc_id": item.get("doc_id", ""),
                "page_index": int(item.get("page_index", -1)),
                "text": str(item.get("text", "")),
                "bbox": item.get("bbox", []),
                "confidence": float(item.get("confidence", 0.0)),
                "score": 0.0,
            })

        out_dir = Path(index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "bm25_index.pkl").open("wb") as f:
            pickle.dump({"index": self.index, "texts": self._texts}, f)
        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in self.metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        cfg = _BM25Config(index_type="bm25", num_docs=len(self.metadata))
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "BM25Index":
        root = Path(index_dir).expanduser().resolve()
        index_file = root / "bm25_index.pkl"
        meta_file = root / "metadata.jsonl"
        cfg_file = root / "config.json"
        missing = [p for p in [index_file, meta_file, cfg_file] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"BM25 index directory incomplete, missing: {missing}")
        instance = cls()
        with index_file.open("rb") as f:
            data = pickle.load(f)
        instance.index = data["index"]
        instance._texts = data["texts"]
        instance.metadata = instance._load_jsonl(str(meta_file))
        return instance

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self.index is None:
            raise RuntimeError("BM25Index not loaded.")
        tokenized_query = self._tokenize(query.strip())
        scores = self.index.get_scores(tokenized_query)
        k = min(top_k, len(scores))
        if k == 0:
            return []
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in indexed_scores[:k]]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = dict(self.metadata[idx])
            row["score"] = float(scores[idx])
            results.append(row)
        return results
