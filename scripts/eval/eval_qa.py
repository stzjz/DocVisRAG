import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.eval import exact_match, recall_at_k, simple_anls, token_f1
from src.docvisrag.qa import DocQAEngine


def _load_questions(path: str) -> List[Dict[str, Any]]:
    q_path = Path(path).expanduser().resolve()
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    rows: List[Dict[str, Any]] = []
    with q_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            obj.setdefault("type", "text")
            obj.setdefault("evidence_pages", [])
            rows.append(obj)
    return rows


def _extract_pages_from_citations(citations: List[str]) -> List[int]:
    pages: List[int] = []
    for c in citations:
        for num in re.findall(r"\d+", str(c)):
            pages.append(int(num))
    # preserve order while dedup
    uniq: List[int] = []
    seen = set()
    for p in pages:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DocQA answer quality (EM/F1/ANLS).")
    parser.add_argument("--questions", required=True, help="Path to questions jsonl")
    parser.add_argument("--index-dir", required=True, help="Hybrid index directory")
    parser.add_argument(
        "--retriever-type",
        default="hybrid",
        choices=["hybrid", "visual", "fusion"],
        help="Retriever type for DocQA.",
    )
    parser.add_argument(
        "--visual-index-dir",
        default=None,
        help="Optional visual index directory for visual/fusion mode.",
    )
    parser.add_argument("--out", required=True, help="Output predictions jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k pages for DocQA retrieval")
    parser.add_argument("--model-id", default=None, help="Optional model id override")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit loading")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens for generation")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        questions = _load_questions(args.questions)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load questions: {exc}")
        return 1

    if args.limit is not None and args.limit > 0:
        questions = questions[: args.limit]

    if not questions:
        print("[ERROR] No valid questions to evaluate.")
        return 1

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
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to initialize DocQAEngine: {exc}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ems: List[float] = []
    f1s: List[float] = []
    anls_scores: List[float] = []
    recall3: List[float] = []

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample in questions:
            qid = str(sample.get("id", ""))
            question = str(sample.get("question", "")).strip()
            gold_answer = str(sample.get("answer", "")).strip()
            gold_pages = [int(x) for x in sample.get("evidence_pages", [])]

            if not question:
                continue

            try:
                result = engine.answer(question)
                pred_answer = result.answer
                evidence_pages = [int(x.get("page_index", -1)) for x in result.evidence if x.get("page_index") is not None]
                citation_pages = _extract_pages_from_citations(result.citations)
                row_error = None
            except Exception as exc:  # noqa: BLE001
                pred_answer = ""
                evidence_pages = []
                citation_pages = []
                row_error = str(exc)

            em = exact_match(pred_answer, gold_answer)
            f1 = token_f1(pred_answer, gold_answer)
            anls = simple_anls(pred_answer, gold_answer)
            r3 = recall_at_k(evidence_pages, gold_pages, 3) if gold_pages else 0.0

            ems.append(em)
            f1s.append(f1)
            anls_scores.append(anls)
            recall3.append(r3)

            payload = {
                "id": qid,
                "type": sample.get("type", "text"),
                "doc_path": sample.get("doc_path", ""),
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "gold_pages": gold_pages,
                "pred_citations": result.citations if row_error is None else [],
                "pred_citation_pages": citation_pages,
                "pred_evidence_pages": evidence_pages,
                "uncertainty": result.uncertainty if row_error is None else "error",
                "em": em,
                "f1": f1,
                "anls": anls,
                "recall@3": r3,
                "error": row_error,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    summary = {
        "num_questions": count,
        "retriever_type": args.retriever_type,
        "em": _avg(ems),
        "f1": _avg(f1s),
        "anls": _avg(anls_scores),
        "recall@3": _avg(recall3),
        "questions": str(Path(args.questions).as_posix()),
        "index_dir": str(Path(args.index_dir).as_posix()),
        "predictions": str(out_path.as_posix()),
    }

    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] QA evaluation completed.")
    print(f"- questions: {count}")
    print(f"- EM={summary['em']:.4f} F1={summary['f1']:.4f} ANLS={summary['anls']:.4f}")
    print(f"- predictions: {out_path}")
    print(f"- summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

