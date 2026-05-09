import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import TextIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search OCR text blocks from FAISS index.")
    parser.add_argument("--index-dir", required=True, help="Index directory path.")
    parser.add_argument("--question", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        idx = TextIndex.load(args.index_dir)
        results = idx.search(args.question, top_k=args.top_k)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Text search failed: {exc}")
        return 1

    print(f"[OK] Retrieved {len(results)} results (top_k={args.top_k})")
    for i, row in enumerate(results, start=1):
        print(f"\n[{i}] score={row.get('score', 0.0):.4f}")
        print(f"page={row.get('page_index')} bbox={row.get('bbox')}")
        print(f"text={row.get('text', '')}")
        print("json=", json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
