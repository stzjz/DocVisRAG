import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.ingest import build_page_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build VLM page summaries from manifest pages.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--out", required=True, help="Output summary jsonl path")
    parser.add_argument("--model-id", default=None, help="Optional VLM model id override")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args()
    try:
        build_page_summaries(args.manifest, args.out, model_id=args.model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Build page summaries failed: {exc}")
        return 1
    print(f"[OK] Page summaries output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
