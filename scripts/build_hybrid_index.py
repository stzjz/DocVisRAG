import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import HybridPageIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hybrid page index from summaries + OCR.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--ocr", default=None, help="Optional OCR jsonl path")
    parser.add_argument("--summaries", required=True, help="Path to page summaries jsonl")
    parser.add_argument("--index-dir", required=True, help="Output index directory")
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding model name for sentence-transformers",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        idx = HybridPageIndex()
        idx.build(
            manifest_path=args.manifest,
            ocr_jsonl=args.ocr,
            summary_jsonl=args.summaries,
            index_dir=args.index_dir,
            model_name=args.model_name,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Build hybrid index failed: {exc}")
        return 1
    print("[OK] Hybrid page index built.")
    print(f"- index_dir: {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
