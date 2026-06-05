#!/usr/bin/env python3
"""构建 BM25 稀疏检索索引（在 OCR 文本块上）。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import BM25Index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BM25 sparse index from OCR JSONL.")
    parser.add_argument("--ocr", required=True, help="Path to ocr.jsonl")
    parser.add_argument("--index-dir", required=True, help="Output BM25 index directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        index = BM25Index()
        index.build_from_ocr_jsonl(ocr_jsonl=args.ocr, index_dir=args.index_dir)
        print(f"[OK] BM25 index built: {args.index_dir}")
        print(f"     documents: {len(index.metadata)}")
    except Exception as exc:
        print(f"[ERROR] BM25 index build failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
