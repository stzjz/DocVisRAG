import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.retrieve import VisualPageIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build page-level visual index (Byaldi/ColPali style).")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--index-dir", required=True, help="Output visual index directory")
    parser.add_argument(
        "--model-id",
        default="vidore/colqwen2-v1.0",
        help="Visual retriever model id for byaldi backend.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        idx = VisualPageIndex()
        idx.build(
            manifest_path=args.manifest,
            index_dir=args.index_dir,
            model_id=args.model_id,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Build visual index failed: {exc}")
        return 1
    print("[OK] Visual page index built.")
    print(f"- index_dir: {args.index_dir}")
    print(f"- model_id: {args.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
