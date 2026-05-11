import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BENCHMARK_ALIASES = {
    "docvqa": "docvqa",
    "chartqa": "chartqa",
    "chartvqa": "chartqa",
    "textvqa": "textvqa",
}

DEFAULT_DATASETS = {
    "docvqa": {
        "dataset_id": "pixparse/docvqa-single-page-questions",
        "split": "validation",
        "type": "text",
    },
    "chartqa": {
        "dataset_id": "HuggingFaceM4/ChartQA",
        "split": "val",
        "type": "chart",
    },
    "textvqa": {
        "dataset_id": "lmms-lab/textvqa",
        "split": "validation",
        "type": "text",
    },
}

SPLIT_ALIASES = {
    "validation": ["validation", "val"],
    "val": ["val", "validation"],
    "test": ["test"],
    "train": ["train"],
}


def canonical_benchmark(name: str) -> str:
    value = (name or "").strip().lower()
    if value not in BENCHMARK_ALIASES:
        raise ValueError(f"Unsupported benchmark: {name}")
    return BENCHMARK_ALIASES[value]


def _load_dataset(dataset_id: str, split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "datasets library is required. Please install it with: pip install -U datasets"
        ) from exc

    tried = []
    for s in SPLIT_ALIASES.get(split, [split]):
        tried.append(s)
        try:
            return load_dataset(dataset_id, split=s, streaming=streaming), s
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError(
        f"Failed to load dataset split. dataset={dataset_id}, split={split}, tried={tried}. "
        "You can override with --dataset-id or --split."
    )


def _image_to_pil(image_obj: Any):
    from PIL import Image

    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, dict):
        img_bytes = image_obj.get("bytes")
        img_path = image_obj.get("path")
        if img_bytes is not None:
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        if img_path:
            return Image.open(img_path).convert("RGB")

    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")

    raise ValueError(f"Unsupported image object type: {type(image_obj)}")


def _pick_answer(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            ans = str(item).strip()
            if ans:
                return ans
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pick_question(row: Dict[str, Any]) -> str:
    for key in ["question", "query", "problem"]:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _image_key(row: Dict[str, Any], benchmark: str, row_idx: int, image_obj: Any) -> str:
    if benchmark == "textvqa" and row.get("image_id"):
        return f"image_id:{row['image_id']}"

    if benchmark == "docvqa":
        meta = row.get("other_metadata")
        if isinstance(meta, dict):
            if meta.get("image"):
                return f"meta_image:{meta['image']}"
            if meta.get("doc_id") is not None:
                return f"doc_id:{meta['doc_id']}"

    if row.get("imgname"):
        return f"imgname:{row['imgname']}"

    if hasattr(image_obj, "filename") and getattr(image_obj, "filename", ""):
        return f"filename:{Path(image_obj.filename).name}"

    from PIL import Image

    if isinstance(image_obj, Image.Image):
        with BytesIO() as buffer:
            image_obj.save(buffer, format="PNG")
            return "md5:" + hashlib.md5(buffer.getvalue()).hexdigest()

    return f"row:{row_idx}"


def _iter_rows(ds: Any, limit: int | None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if limit is None:
        for idx, row in enumerate(ds):
            yield idx, row
        return

    for idx, row in enumerate(ds):
        if idx >= limit:
            break
        yield idx, row


def _build_outputs(
    benchmark: str,
    dataset_id: str,
    split: str,
    ds: Any,
    out_dir: Path,
    limit: int | None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    questions_path = out_dir / "questions.jsonl"

    doc_id = f"{benchmark}-{split}"
    q_type = DEFAULT_DATASETS.get(benchmark, {}).get("type", "text")

    image_to_page: Dict[str, int] = {}
    pages: List[Dict[str, Any]] = []

    stats = {
        "rows_seen": 0,
        "questions_written": 0,
        "pages_written": 0,
        "benchmark": benchmark,
        "dataset_id": dataset_id,
        "split": split,
    }

    with questions_path.open("w", encoding="utf-8") as qf:
        for row_idx, row in _iter_rows(ds, limit):
            stats["rows_seen"] += 1

            question = _pick_question(row)
            if not question:
                continue

            answer_raw = row.get("answers", row.get("label", row.get("answer", "")))
            answer = _pick_answer(answer_raw)

            image_obj = row.get("image")
            if image_obj is None:
                continue
            pil = _image_to_pil(image_obj)
            key = _image_key(row, benchmark=benchmark, row_idx=row_idx, image_obj=pil)

            if key not in image_to_page:
                page_index = len(pages) + 1
                image_rel = f"images/page_{page_index:06d}.png"
                image_path = out_dir / image_rel
                pil.save(image_path, format="PNG")
                width, height = pil.size

                pages.append(
                    {
                        "doc_id": doc_id,
                        "source_path": f"huggingface://{dataset_id}/{split}/{key}",
                        "page_index": page_index,
                        "image_path": str(image_path.resolve()),
                        "width": int(width),
                        "height": int(height),
                    }
                )
                image_to_page[key] = page_index

            page_index = image_to_page[key]

            qid = row.get("question_id", row.get("id", f"{benchmark}_{split}_{row_idx:06d}"))
            payload = {
                "id": str(qid),
                "doc_path": str(manifest_path),
                "question": question,
                "answer": answer,
                "evidence_pages": [int(page_index)],
                "type": str(q_type),
            }
            qf.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stats["questions_written"] += 1

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    stats["pages_written"] = len(pages)
    return manifest_path, questions_path, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert benchmark datasets to DocVisRAG bench format.")
    parser.add_argument("--benchmark", required=True, help="docvqa/chartqa/chartvqa/textvqa")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-id", default=None, help="Optional Hugging Face dataset id override")
    parser.add_argument("--split", default=None, help="Dataset split override")
    parser.add_argument("--limit", type=int, default=None, help="Only prepare first N samples")
    parser.add_argument("--streaming", action="store_true", help="Force streaming mode")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        benchmark = canonical_benchmark(args.benchmark)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] invalid benchmark: {exc}")
        return 1

    defaults = DEFAULT_DATASETS[benchmark]
    dataset_id = args.dataset_id or defaults["dataset_id"]
    split = args.split or defaults["split"]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_streaming = args.streaming or (args.limit is not None)

    try:
        ds, used_split = _load_dataset(dataset_id=dataset_id, split=split, streaming=use_streaming)
        manifest, questions, stats = _build_outputs(
            benchmark=benchmark,
            dataset_id=dataset_id,
            split=used_split,
            ds=ds,
            out_dir=out_dir,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] prepare benchmark failed: {exc}")
        return 1

    meta = {
        "benchmark": benchmark,
        "requested_benchmark": args.benchmark,
        "dataset_id": dataset_id,
        "split": used_split,
        "limit": args.limit,
        "streaming": use_streaming,
        "manifest": str(manifest),
        "questions": str(questions),
        "stats": stats,
    }
    with (out_dir / "prepare_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] benchmark prepared")
    print(f"- benchmark: {benchmark}")
    print(f"- dataset_id: {dataset_id}")
    print(f"- split: {used_split}")
    print(f"- rows_seen: {stats['rows_seen']}")
    print(f"- pages(unique images): {stats['pages_written']}")
    print(f"- questions: {stats['questions_written']}")
    print(f"- manifest: {manifest}")
    print(f"- questions_jsonl: {questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
