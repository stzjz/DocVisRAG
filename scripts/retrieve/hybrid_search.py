import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import HybridPageIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search page-level hybrid index.")
    parser.add_argument("--index-dir", required=True, help="Hybrid index directory")
    parser.add_argument("--question", required=True, help="Query question")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k pages")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        idx = HybridPageIndex.load(args.index_dir)
        results = idx.search(args.question, top_k=args.top_k)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Hybrid search failed: {exc}")
        return 1

    print(f"[OK] Retrieved {len(results)} pages (top_k={args.top_k})")
    for i, row in enumerate(results, start=1):
        print(f"\n[{i}] score={row.get('score', 0.0):.4f}")
        print(f"page={row.get('page_index')} image_path={row.get('image_path')}")
        print(f"summary={row.get('summary', '')}")
        print(f"ocr_text_preview={row.get('ocr_text_preview', '')}")
        print("json=", json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
