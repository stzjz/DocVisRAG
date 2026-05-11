import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import TextIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS text index from OCR jsonl.")
    parser.add_argument("--ocr", required=True, help="Path to OCR jsonl.")
    parser.add_argument("--index-dir", required=True, help="Directory to save index files.")
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding model name for sentence-transformers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        idx = TextIndex()
        idx.build_from_ocr_jsonl(args.ocr, args.index_dir, model_name=args.model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Build text index failed: {exc}")
        return 1

    print("[OK] Text index built.")
    print(f"- index_dir: {args.index_dir}")
    print(f"- model_name: {args.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
