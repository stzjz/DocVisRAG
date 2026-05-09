import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.qa import PageQAEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Page-level VLM QA for rendered document pages.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--page", required=True, type=int, help="1-based page index")
    parser.add_argument("--question", required=True, help="Question for the selected page")
    parser.add_argument("--model-id", default=None, help="Optional model id override")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit loading")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        engine = PageQAEngine(model_id=args.model_id, load_in_4bit=args.load_in_4bit)
        engine.max_new_tokens = args.max_new_tokens
        result = engine.answer_page(
            manifest_path=args.manifest,
            page=args.page,
            question=args.question,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Page QA failed: {exc}")
        return 1

    print("答案：")
    print(result["answer"])
    print(f"引用：{result['citation']}")
    print(f"页面图像：{result['image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
