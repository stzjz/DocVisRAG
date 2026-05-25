import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.ingest import LayoutAnalyzer, load_layout_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="版面分析：检测页面中的文本、表格、图表、公式等语义区域。"
    )
    parser.add_argument("--manifest", required=True, help="Manifest JSON 文件路径。")
    parser.add_argument("--out", required=True, help="输出 layout JSONL 文件路径。")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "ppstructure", "opencv"],
        help="版面检测后端：auto（优选PP-Structure）、ppstructure、opencv。",
    )
    parser.add_argument("--ocr", default=None, help="可选 OCR JSONL，用于将 OCR 文本归并到版面区域。")
    parser.add_argument(
        "--enriched-out", default=None,
        help="可选输出路径，保存带有 OCR 文本的富化后版面区域 JSONL。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        analyzer = LayoutAnalyzer(backend=args.backend)
        result = analyzer.analyze_manifest(
            manifest_path=args.manifest,
            output_jsonl=args.out,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Layout analysis failed: {exc}")
        return 1

    print("[OK] Layout analysis completed.")
    print(f"- backend: {result['backend']}")
    print(f"- pages: {result['num_pages']}")
    print(f"- regions: {result['total_regions']}")
    if result["failed_pages"]:
        print(f"- failed_pages: {result['failed_pages']}")
    print(f"- output: {result['output_path']}")

    if args.ocr and args.enriched_out:
        print("\nAssigning OCR text to layout regions...")
        try:
            enrich_result = analyzer.assign_ocr_to_regions(
                ocr_jsonl=args.ocr,
                layout_jsonl=args.out,
                output_jsonl=args.enriched_out,
            )
            print(f"[OK] Enriched regions: {enrich_result['total_regions']}")
            print(f"- output: {enrich_result['output_path']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] OCR-to-region assignment failed: {exc}")
            return 1

    # Print type distribution
    try:
        regions = load_layout_jsonl(args.out)
        type_counts = {}
        for r in regions:
            t = r.region_type
            type_counts[t] = type_counts.get(t, 0) + 1
        print("\n区域类型分布：")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
    except Exception:  # noqa: BLE001
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
