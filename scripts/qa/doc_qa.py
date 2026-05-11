import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.qa import DocQAEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end multimodal DocQA from hybrid index.")
    parser.add_argument("--index-dir", required=True, help="Hybrid index directory.")
    parser.add_argument(
        "--retriever-type",
        default="hybrid",
        choices=["hybrid", "visual", "fusion"],
        help="Retriever type: hybrid (default), visual, or fusion.",
    )
    parser.add_argument(
        "--visual-index-dir",
        default=None,
        help="Optional visual index directory. Required for visual/fusion when it cannot be inferred.",
    )
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k pages for retrieval.")
    parser.add_argument("--model-id", default=None, help="Optional VLM model id override.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit model loading.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        engine = DocQAEngine(
            index_dir=args.index_dir,
            model_id=args.model_id,
            top_k=args.top_k,
            load_in_4bit=args.load_in_4bit,
            retriever_type=args.retriever_type,
            visual_index_dir=args.visual_index_dir,
        )
        engine.max_new_tokens = args.max_new_tokens
        result = engine.answer(args.question)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Doc QA failed: {exc}")
        return 1

    print("答案：")
    print(result.answer)

    if result.evidence:
        print("依据：")
        for i, ev in enumerate(result.evidence, start=1):
            print(
                f"[{i}] 第 {ev['page_index']} 页 "
                f"score={ev['score']:.4f} image={ev['image_path']}"
            )
            print(f"summary={ev['summary']}")
            print(f"ocr_text_preview={ev['ocr_text_preview']}")
    else:
        print("依据：无")

    print("引用：")
    print("; ".join(result.citations) if result.citations else "无")
    print("不确定性：")
    print(result.uncertainty or "无")
    print("结构化结果：")
    print(json.dumps(result.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

