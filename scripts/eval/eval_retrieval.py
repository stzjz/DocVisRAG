import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.eval import mrr, recall_at_k
from src.docvisrag.retrieve import HybridPageIndex


def _load_questions(path: str) -> List[Dict[str, Any]]:
    q_path = Path(path).expanduser().resolve()
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    questions: List[Dict[str, Any]] = []
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
            questions.append(obj)
    return questions


def _avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on questions JSONL.")
    parser.add_argument("--questions", required=True, help="Path to questions jsonl")
    parser.add_argument("--index-dir", required=True, help="Hybrid index directory")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k (default: 5)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        questions = _load_questions(args.questions)
        index = HybridPageIndex.load(args.index_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Init retrieval evaluation failed: {exc}")
        return 1

    if not questions:
        print("[ERROR] No valid questions found.")
        return 1

    details: List[Dict[str, Any]] = []
    group_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"r1": [], "r3": [], "r5": [], "mrr": []}
    )

    for q in questions:
        qid = str(q.get("id", ""))
        question = str(q.get("question", "")).strip()
        qtype = str(q.get("type", "text"))
        gold_pages = [int(x) for x in q.get("evidence_pages", [])]

        if not question:
            continue

        try:
            results = index.search(question, top_k=max(5, args.top_k))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Retrieval failed for {qid}: {exc}")
            continue

        retrieved_pages = [int(r.get("page_index", -1)) for r in results]
        r1 = recall_at_k(retrieved_pages, gold_pages, 1)
        r3 = recall_at_k(retrieved_pages, gold_pages, 3)
        r5 = recall_at_k(retrieved_pages, gold_pages, 5)
        rr = mrr(retrieved_pages, gold_pages)

        row = {
            "id": qid,
            "type": qtype,
            "question": question,
            "gold_pages": gold_pages,
            "retrieved_pages": retrieved_pages,
            "recall@1": r1,
            "recall@3": r3,
            "recall@5": r5,
            "mrr": rr,
        }
        details.append(row)

        group_metrics[qtype]["r1"].append(r1)
        group_metrics[qtype]["r3"].append(r3)
        group_metrics[qtype]["r5"].append(r5)
        group_metrics[qtype]["mrr"].append(rr)

    if not details:
        print("[ERROR] No question was evaluated.")
        return 1

    overall = {
        "num_questions": len(details),
        "recall@1": _avg([x["recall@1"] for x in details]),
        "recall@3": _avg([x["recall@3"] for x in details]),
        "recall@5": _avg([x["recall@5"] for x in details]),
        "mrr": _avg([x["mrr"] for x in details]),
    }

    by_type: Dict[str, Dict[str, float]] = {}
    for qtype, m in group_metrics.items():
        by_type[qtype] = {
            "count": len(m["r1"]),
            "recall@1": _avg(m["r1"]),
            "recall@3": _avg(m["r3"]),
            "recall@5": _avg(m["r5"]),
            "mrr": _avg(m["mrr"]),
        }

    payload = {
        "questions_file": str(Path(args.questions).as_posix()),
        "index_dir": str(Path(args.index_dir).as_posix()),
        "overall": overall,
        "by_type": by_type,
        "details": details,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] Retrieval evaluation completed.")
    print(f"- questions: {len(details)}")
    print(
        "- overall: "
        f"R@1={overall['recall@1']:.4f}, "
        f"R@3={overall['recall@3']:.4f}, "
        f"R@5={overall['recall@5']:.4f}, "
        f"MRR={overall['mrr']:.4f}"
    )
    print(f"- output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
