import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.qa import TextDocQAEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="纯文本 RAG 文档问答（仅基于 OCR 文本，不使用图像）。")
    parser.add_argument("--index-dir", required=True, help="Text index 目录（由 build_text_index.py 构建）。")
    parser.add_argument("--question", required=True, help="用户问题。")
    parser.add_argument("--top-k", type=int, default=5, help="检索文本块数量（默认 5）。")
    parser.add_argument("--model-id", default=None, help="可选 LLM model id 覆盖。")
    parser.add_argument("--load-in-4bit", action="store_true", help="启用 4-bit 模型加载。")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成 token 数。")
    parser.add_argument(
        "--layout", default=None,
        help="可选 layout JSONL 路径（用于图/表编号引用）。",
    )
    parser.add_argument("--manifest", default=None, help="可选 manifest.json 路径。")
    parser.add_argument("--summaries", default=None, help="可选 page_summaries.jsonl 路径。")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        engine = TextDocQAEngine(
            index_dir=args.index_dir,
            model_id=args.model_id,
            top_k=args.top_k,
            load_in_4bit=args.load_in_4bit,
            layout_jsonl=args.layout,
            manifest_path=args.manifest,
            summary_jsonl=args.summaries,
        )
        engine.max_new_tokens = args.max_new_tokens
        result = engine.answer(args.question)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Text QA failed: {exc}")
        return 1

    print("答案：")
    print(result.answer)

    if result.evidence:
        print("依据：")
        for i, ev in enumerate(result.evidence, start=1):
            print(
                f"[{i}] 第 {ev['page_index']} 页 "
                f"score={ev['score']:.4f}"
            )
            text_preview = str(ev.get("text", ""))[:200]
            print(f"text={text_preview}")
    else:
        print("依据：无")

    print("引用：")
    print("; ".join(result.citations) if result.citations else "无")
    print("不确定性：")
    print(result.uncertainty or "无")
    print("结构化结果：")
    print(json.dumps(result.__dict__, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
