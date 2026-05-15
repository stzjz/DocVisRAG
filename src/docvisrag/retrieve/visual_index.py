import json
import shutil
import traceback
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.docvisrag.ingest.render import load_manifest
from src.docvisrag.retrieve.base import BaseRetriever


@dataclass
class _VisualConfig:
    backend: str
    model_id: str
    index_name: str
    index_root: str
    num_pages: int


class VisualPageIndex(BaseRetriever):
    def __init__(self) -> None:
        self.backend: Optional[str] = None
        self.model_id: Optional[str] = None
        self.index_name: Optional[str] = None
        self.index_root: Optional[str] = None
        self.rag_model: Any = None
        self.metadata: List[Dict[str, Any]] = []
        self.doc_id_to_meta: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def dependency_report() -> Dict[str, str]:
        report: Dict[str, str] = {}
        for pkg in ["byaldi", "colpali-engine", "peft", "transformers", "accelerate", "torch"]:
            try:
                report[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                report[pkg] = "MISSING"
        return report

    @classmethod
    def diagnostics(cls) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "backend": "byaldi",
            "dependencies": cls.dependency_report(),
            "ready": False,
            "error": "",
        }
        try:
            cls._require_byaldi()
            report["ready"] = True
        except Exception as exc:  # noqa: BLE001
            report["error"] = str(exc)
        return report

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
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

    @staticmethod
    def _require_byaldi() -> Any:
        deps = VisualPageIndex.dependency_report()
        try:
            from peft.utils import save_and_load as peft_save_and_load
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "VisualPageIndex requires `peft` but import failed. "
                f"Detected dependency versions: {deps}. "
                "Rebuild with visual dependencies, e.g. "
                "`docker build --build-arg INSTALL_VISUAL=true -t docvisrag:cu124 .`"
            ) from exc

        if not hasattr(peft_save_and_load, "_maybe_shard_state_dict_for_tp"):
            raise RuntimeError(
                "Detected incompatible peft version for Byaldi/ColPali path: "
                "missing `peft.utils.save_and_load._maybe_shard_state_dict_for_tp`.\n"
                f"Detected dependency versions: {deps}\n"
                "Recommended fix:\n"
                "  docker build --build-arg INSTALL_VISUAL=true -t docvisrag:cu124 .\n"
                "or adjust requirements-visual.txt to a compatible Byaldi/ColPali stack."
            )

        try:
            from byaldi import RAGMultiModalModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "VisualPageIndex requires optional visual retrieval dependency `byaldi`.\n"
                f"Detected dependency versions: {deps}\n"
                "Install suggestions:\n"
                "1) rebuild with `docker build --build-arg INSTALL_VISUAL=true -t docvisrag:cu124 .`\n"
                "2) verify model compatibility for ColPali/Byaldi."
            ) from exc
        return RAGMultiModalModel

    @staticmethod
    def _write_failure(index_dir: Path, exc: Exception) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "byaldi",
            "ready": False,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=3),
            "dependencies": VisualPageIndex.dependency_report(),
        }
        with (index_dir / "visual_status.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_success(index_dir: Path, cfg: _VisualConfig) -> None:
        payload = {
            "backend": cfg.backend,
            "ready": True,
            "model_id": cfg.model_id,
            "index_name": cfg.index_name,
            "index_root": cfg.index_root,
            "num_pages": cfg.num_pages,
            "dependencies": VisualPageIndex.dependency_report(),
        }
        with (index_dir / "visual_status.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_metadata(self, index_dir: Path, metadata: List[Dict[str, Any]]) -> None:
        with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_metadata(self, index_dir: Path) -> None:
        rows = self._load_jsonl(index_dir / "metadata.jsonl")
        self.metadata = rows
        self.doc_id_to_meta = {int(r["visual_doc_id"]): r for r in rows}

    def build(
        self,
        manifest_path: str,
        index_dir: str,
        model_id: str = "vidore/colqwen2-v1.0",
    ) -> None:
        manifest_file = Path(manifest_path).expanduser().resolve()
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        pages = load_manifest(str(manifest_file))
        if not pages:
            raise ValueError(f"Manifest has no pages: {manifest_file}")

        out_dir = Path(index_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._build_checked(
                pages=pages,
                manifest_file=manifest_file,
                out_dir=out_dir,
                model_id=model_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._write_failure(out_dir, exc)
            raise

    def _build_checked(
        self,
        pages: List[Any],
        manifest_file: Path,
        out_dir: Path,
        model_id: str,
    ) -> None:
        RAGMultiModalModel = self._require_byaldi()

        pages_dir = out_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        metadata: List[Dict[str, Any]] = []
        for i, page in enumerate(pages):
            src = Path(page.image_path).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(f"Page image from manifest not found: {src}")
            dst = pages_dir / f"page_{i + 1:06d}{src.suffix.lower() or '.png'}"
            if src != dst:
                shutil.copy2(src, dst)
            metadata.append(
                {
                    "visual_doc_id": i,
                    "doc_id": page.doc_id,
                    "page_index": int(page.page_index),
                    "image_path": str(Path(page.image_path)),
                    "local_index_image_path": str(dst),
                    "score": 0.0,
                }
            )

        index_root = out_dir / "byaldi_index"
        index_root.mkdir(parents=True, exist_ok=True)
        index_name = "docvisrag_visual"

        rag = RAGMultiModalModel.from_pretrained(
            model_id,
            index_root=str(index_root),
            verbose=1,
        )

        # byaldi versions vary in index(...) kwargs. Try rich call first, then fallback.
        try:
            rag.index(
                input_path=str(pages_dir),
                index_name=index_name,
                store_collection_with_index=False,
                overwrite=True,
                doc_ids=[int(x["visual_doc_id"]) for x in metadata],
                metadata=[
                    {
                        "page_index": int(x["page_index"]),
                        "image_path": str(x["image_path"]),
                    }
                    for x in metadata
                ],
            )
        except TypeError:
            rag.index(
                input_path=str(pages_dir),
                index_name=index_name,
                store_collection_with_index=False,
                overwrite=True,
            )

        cfg = _VisualConfig(
            backend="byaldi",
            model_id=model_id,
            index_name=index_name,
            index_root=str(index_root),
            num_pages=len(metadata),
        )
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)
        self._save_metadata(out_dir, metadata)
        self._write_success(out_dir, cfg)

        self.backend = cfg.backend
        self.model_id = cfg.model_id
        self.index_name = cfg.index_name
        self.index_root = cfg.index_root
        self.rag_model = rag
        self.metadata = metadata
        self.doc_id_to_meta = {int(x["visual_doc_id"]): x for x in metadata}

    @classmethod
    def load(cls, index_dir: str) -> "VisualPageIndex":
        RAGMultiModalModel = cls._require_byaldi()

        root = Path(index_dir).expanduser().resolve()
        cfg_file = root / "config.json"
        meta_file = root / "metadata.jsonl"
        missing = [p for p in [cfg_file, meta_file] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Visual index directory is incomplete, missing: {missing}")

        with cfg_file.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        instance = cls()
        instance.backend = str(cfg.get("backend", "byaldi"))
        instance.model_id = str(cfg.get("model_id", "vidore/colqwen2-v1.0"))
        instance.index_name = str(cfg.get("index_name", "docvisrag_visual"))
        instance.index_root = str(cfg.get("index_root", str(root / "byaldi_index")))
        instance._load_metadata(root)

        # byaldi versions differ: sometimes `from_index(index_name, index_root=...)`,
        # sometimes accept path-like index identifier.
        try:
            instance.rag_model = RAGMultiModalModel.from_index(
                instance.index_name,
                index_root=instance.index_root,
            )
        except Exception:
            index_path = str(Path(instance.index_root) / instance.index_name)
            instance.rag_model = RAGMultiModalModel.from_index(index_path)
        return instance

    def _coerce_result(self, raw: Any, rank: int) -> Dict[str, Any]:
        if isinstance(raw, dict):
            data = dict(raw)
        else:
            data = {}
            for attr in ["score", "doc_id", "page_num", "metadata"]:
                if hasattr(raw, attr):
                    data[attr] = getattr(raw, attr)

        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = {}

        doc_id_val = data.get("doc_id")
        try:
            doc_id = int(doc_id_val)
        except Exception:
            doc_id = None

        row: Dict[str, Any] = {}
        if doc_id is not None and doc_id in self.doc_id_to_meta:
            row.update(self.doc_id_to_meta[doc_id])

        page_index = meta.get("page_index", row.get("page_index", -1))
        image_path = meta.get("image_path", row.get("image_path", ""))
        score = data.get("score", row.get("score", 0.0))

        row.update(
            {
                "page_index": int(page_index) if str(page_index).strip() else -1,
                "image_path": str(image_path),
                "score": float(score) if score is not None else float(max(0.0, 1.0 - rank * 0.01)),
                "visual_backend": self.backend or "byaldi",
            }
        )
        return row

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self.rag_model is None:
            raise RuntimeError("VisualPageIndex is not loaded. Call load() or build() first.")

        try:
            raw_results = self.rag_model.search(query.strip(), k=top_k)
        except TypeError:
            raw_results = self.rag_model.search(query.strip(), top_k=top_k)

        if raw_results is None:
            return []

        results: List[Dict[str, Any]] = []
        for rank, raw in enumerate(list(raw_results), start=1):
            row = self._coerce_result(raw, rank=rank)
            if not row.get("image_path"):
                continue
            results.append(row)
        return results[:top_k]
