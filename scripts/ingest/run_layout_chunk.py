import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.ingest import LayoutChunker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="版面感知分块：将布局区域与 OCR 文本归并，生成类型感知的文档块。"
    )
    parser.add_argument("--layout", required=True, help="Layout JSONL 文件（由 run_layout.py 生成）。")
    parser.add_argument("--ocr", required=True, help="OCR JSONL 文件（由 run_ocr.py 生成）。")
    parser.add_argument("--out", required=True, help="输出 chunks JSONL 文件路径。")
    parser.add_argument(
        "--min-text-length", type=int, default=10,
        help="最小文本长度，短于此长度的块将被跳过（默认 10）。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        chunker = LayoutChunker()
        result = chunker.build_chunks_from_layout(
            layout_jsonl=args.layout,
            ocr_jsonl=args.ocr,
            output_jsonl=args.out,
            min_text_length=args.min_text_length,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Layout chunking failed: {exc}")
        return 1

    print("[OK] Layout-aware chunks built.")
    print(f"- chunks: {result['total_chunks']}")
    print(f"- output: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
