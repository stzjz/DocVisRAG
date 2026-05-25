"""构建版面感知 FAISS 文本索引。

与 build_text_index.py 的区别：
- 输入是 layout chunks JSONL（版面感知的语义块），而非 OCR 行级 JSONL
- 每个 chunk 带有类型标签（text/table/title 等），元数据中保留类型和 bbox
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import TextIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build FAISS text index from layout-aware chunks."
    )
    parser.add_argument("--chunks", required=True, help="Layout chunks JSONL 路径。")
    parser.add_argument("--index-dir", required=True, help="索引输出目录。")
    parser.add_argument(
        "--model-name", default="BAAI/bge-small-zh-v1.5",
        help="Embedding 模型名称。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    chunks_path = Path(args.chunks)

    # Read chunks and convert to OCR-like format for TextIndex
    items = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            items.append(obj)

    if not items:
        print("[ERROR] No valid chunks with text found.")
        return 1

    # Write as temporary OCR-style JSONL and build index
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as tmp:
        for item in items:
            tmp.write(json.dumps({
                "doc_id": item.get("doc_id", ""),
                "page_index": item.get("page_index", -1),
                "text": item.get("text", ""),
                "bbox": item.get("bbox", []),
                "confidence": item.get("confidence", 0.0),
                "chunk_type": item.get("chunk_type", "text"),
            }, ensure_ascii=False) + "\n")
        tmp_path = tmp.name

    try:
        idx = TextIndex()
        idx.build_from_ocr_jsonl(
            ocr_jsonl=tmp_path,
            index_dir=args.index_dir,
            model_name=args.model_name,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    type_counts = {}
    for item in items:
        t = item.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("[OK] Layout-aware index built.")
    print(f"- index_dir: {args.index_dir}")
    print(f"- model_name: {args.model_name}")
    print(f"- total vectors: {len(items)}")
    print("- chunk type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
