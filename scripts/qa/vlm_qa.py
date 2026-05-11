import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.vlm import QwenVLClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image VLM QA with Qwen2.5-VL.")
    parser.add_argument("--image", required=True, help="Local image path (.png/.jpg).")
    parser.add_argument("--question", required=True, help="Question for the image.")
    parser.add_argument("--model-id", default=None, help="Optional model id override.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens for generation.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()

    model_id = args.model_id or "Qwen/Qwen2.5-VL-3B-Instruct"

    try:
        client = QwenVLClient(
            model_id=model_id,
            load_in_4bit=args.load_in_4bit,
        )
        answer = client.answer_image(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("VLM QA failed: %s", exc)
        return 1

    print("\n=== Answer ===")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
