import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.ingest import ingest_document, save_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PDF/image document into page images.")
    parser.add_argument("--input", required=True, help="Input path (.pdf/.png/.jpg/.jpeg/.webp).")
    parser.add_argument("--output", required=True, help="Output directory for pages and manifest.")
    parser.add_argument("--dpi", type=int, default=180, help="Rendering DPI for PDF input.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    manifest_path = output_dir / "manifest.json"

    try:
        pages = ingest_document(
            input_path=args.input,
            output_dir=str(output_dir),
            dpi=args.dpi,
        )
        save_manifest(pages, str(manifest_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Ingest/render failed: {exc}")
        return 1

    print(f"[OK] Rendered pages: {len(pages)}")
    print(f"[OK] Manifest: {manifest_path}")
    for p in pages:
        print(f"- page {p.page_index}: {p.image_path} ({p.width}x{p.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
