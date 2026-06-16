"""Create a multiple-choice variant of HomeworkQA.

The source HomeworkQA questions are short-answer PDF QA examples. This script
turns each row into a 4-6 option question by sampling plausible distractors from
answers in the same benchmark, preferring the same PDF/type/answer shape.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LETTERS = "ABCDEF"


def _target_benchmark(rows: List[Dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    source = str(rows[0].get("benchmark", "homeworkqa")).strip() if rows else "homeworkqa"
    return source if source.endswith("_mc") else f"{source}_mc"


def _retarget_id(qid: str, target_benchmark: str) -> str:
    if qid.startswith("homeworkqa_hard_"):
        return qid.replace("homeworkqa_hard_", f"{target_benchmark}_", 1)
    if qid.startswith("homeworkqa_"):
        return qid.replace("homeworkqa_", f"{target_benchmark}_", 1)
    return f"{target_benchmark}_{qid}"


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_rng(seed: int, text: str) -> random.Random:
    digest = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:12], 16))


def _answer_shape(answer: str) -> str:
    text = str(answer).strip()
    if not text:
        return "empty"
    if any(ch.isdigit() for ch in text):
        if any(ch in text for ch in ["年", "月", "日", "-", "/", "."]):
            return "date_or_numeric"
        return "numeric"
    if len(text) <= 6:
        return "short"
    if len(text) <= 20:
        return "medium"
    return "long"


def _unique_answers(rows: Iterable[Dict[str, Any]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for row in rows:
        ans = str(row.get("answer", "")).strip()
        if ans and ans not in seen:
            seen.add(ans)
            out.append(ans)
    return out


def _collect_candidates(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[str]:
    answer = str(row.get("answer", "")).strip()
    doc_id = str(row.get("doc_id", "")).strip()
    qtype = str(row.get("type", "")).strip()
    shape = _answer_shape(answer)

    buckets: List[List[Dict[str, Any]]] = []
    buckets.append([r for r in rows if str(r.get("doc_id", "")).strip() == doc_id and str(r.get("type", "")).strip() == qtype])
    buckets.append([r for r in rows if str(r.get("doc_id", "")).strip() == doc_id])
    buckets.append([r for r in rows if str(r.get("type", "")).strip() == qtype and _answer_shape(str(r.get("answer", ""))) == shape])
    buckets.append([r for r in rows if _answer_shape(str(r.get("answer", ""))) == shape])
    buckets.append(rows)

    candidates: List[str] = []
    seen = {answer}
    for bucket in buckets:
        for ans in _unique_answers(bucket):
            if ans not in seen:
                seen.add(ans)
                candidates.append(ans)
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HomeworkQA-MC from HomeworkQA short-answer questions.")
    parser.add_argument("--questions", default="data/bench_full/homeworkqa/questions.jsonl")
    parser.add_argument("--batch-config", default="data/bench_full/homeworkqa/batch_config.json")
    parser.add_argument("--out-dir", default="data/bench_full/homeworkqa_mc")
    parser.add_argument("--min-choices", type=int, default=4)
    parser.add_argument("--max-choices", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--benchmark", default=None, help="Output benchmark id; defaults to <source_benchmark>_mc.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    min_choices = max(2, min(6, args.min_choices))
    max_choices = max(min_choices, min(6, args.max_choices))
    rows = _load_jsonl(args.questions)
    if not rows:
        print("[ERROR] No source questions found.")
        return 1

    target_benchmark = _target_benchmark(rows, args.benchmark)
    out_rows: List[Dict[str, Any]] = []
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("id", ""))
        answer = str(row.get("answer", "")).strip()
        if not qid or not answer:
            continue
        rng = _stable_rng(args.seed, qid)
        num_choices = rng.randint(min_choices, max_choices)
        distractors = _collect_candidates(rows, row)
        options = [answer] + distractors[: max(0, num_choices - 1)]
        if len(options) < min_choices:
            print(f"[WARN] Skip {qid}: not enough unique options.")
            continue
        rng.shuffle(options)
        answer_idx = options.index(answer)
        choices = {LETTERS[i]: opt for i, opt in enumerate(options)}
        out = dict(row)
        out["id"] = _retarget_id(qid, target_benchmark)
        out["benchmark"] = target_benchmark
        out["source_question_id"] = qid
        out["question"] = str(row.get("question", "")).strip()
        out["choices"] = choices
        out["answer_key"] = LETTERS[answer_idx]
        out["answer"] = answer
        out["choice_generation"] = {
            "method": "deterministic_answer_pool_distractors",
            "min_choices": min_choices,
            "max_choices": max_choices,
            "seed": args.seed,
        }
        out_rows.append(out)
        by_doc[str(out.get("doc_id", "")).strip()].append(out)

    out_dir = Path(args.out_dir).expanduser().resolve()
    _write_jsonl(out_dir / "questions.jsonl", out_rows)
    for doc_id, doc_rows in by_doc.items():
        if doc_id:
            _write_jsonl(out_dir / "questions_by_pdf" / f"{doc_id}.jsonl", doc_rows)

    batch_src = Path(args.batch_config).expanduser().resolve()
    if batch_src.exists():
        batch = json.loads(batch_src.read_text(encoding="utf-8"))
        for item in batch.get("benchmarks", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            doc_name = name.removeprefix("homeworkqa_")
            item["name"] = name.replace("homeworkqa_hard_", f"{target_benchmark}_", 1) if name.startswith("homeworkqa_hard_") else name.replace("homeworkqa_", f"{target_benchmark}_", 1)
            item["benchmark"] = target_benchmark
            item["questions"] = str(out_dir / "questions_by_pdf" / f"{doc_name}.jsonl")
            item["dataset_id"] = f"local_{target_benchmark}"
        (out_dir / "batch_config.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "benchmark": target_benchmark,
        "source_questions": str(Path(args.questions)),
        "num_questions": len(out_rows),
        "min_choices": min_choices,
        "max_choices": max_choices,
        "seed": args.seed,
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
