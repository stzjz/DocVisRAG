import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.ingest import run_ocr_on_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR on rendered page images from manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--out", required=True, help="Output OCR jsonl path")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args()
    try:
        run_ocr_on_manifest(args.manifest, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] OCR failed: {exc}")
        return 1
    print(f"[OK] OCR output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
