"""Fuse HomeworkQA short-answer predictions at answer level without gold labels."""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bench.judge_homeworkqa_qa import TextJudge  # noqa: E402
from src.docvisrag.eval import citation_accuracy, exact_match, mrr, recall_at_k, relaxed_accuracy, simple_anls, token_f1  # noqa: E402


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
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id", "")): row for row in rows}


def _truncate(text: str, limit: int = 900) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    candidates = [raw]
    candidates.extend(re.findall(r"\{.*?\}", raw, flags=re.S))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {"answer": raw, "source_mode": "", "reason": "failed_to_parse_json"}


def _extract_pages_from_text(text: str) -> List[int]:
    pages: List[int] = []
    for match in re.findall(r"第\s*(\d+)\s*页", str(text or "")):
        try:
            pages.append(int(match))
        except ValueError:
            continue
    return pages


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "num_questions": len(rows),
        "em": _avg([float(r.get("em", 0.0)) for r in rows]),
        "f1": _avg([float(r.get("f1", 0.0)) for r in rows]),
        "anls": _avg([float(r.get("anls", 0.0)) for r in rows]),
        "relaxed_accuracy": _avg([float(r.get("relaxed_accuracy", 0.0)) for r in rows]),
        "retrieval_recall@1": _avg([float(r.get("retrieval_recall@1", 0.0)) for r in rows]),
        "retrieval_recall@3": _avg([float(r.get("retrieval_recall@3", 0.0)) for r in rows]),
        "retrieval_mrr": _avg([float(r.get("retrieval_mrr", 0.0)) for r in rows]),
        "citation_accuracy": _avg([float(r.get("citation_accuracy", 0.0)) for r in rows]),
    }


def _summarize_by(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field, "")), []).append(row)
    return {key: _summarize(group) for key, group in sorted(groups.items())}


def _candidate_block(mode: str, row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"[{mode}]",
            f"答案：{_truncate(str(row.get('pred_answer', '')), 360)}",
            f"完整输出：{_truncate(str(row.get('raw_pred', '')), 700)}",
            f"引用页/检索页：{row.get('retrieved_pages', [])}",
        ]
    )


def _build_prompt(sample_id: str, candidates: Mapping[str, Mapping[str, Any]]) -> str:
    first = next(iter(candidates.values()))
    question = str(first.get("question", "")).strip()
    cand_text = "\n\n".join(_candidate_block(mode, row) for mode, row in candidates.items())
    return "\n".join(
        [
            "你是 HomeworkQA 简答题的答案融合器。",
            "你只能根据题目和多个模型候选答案进行选择或合并，不要使用标准答案，也不要引入外部知识。",
            "目标：输出一个更完整、更准确、且尽量简短的最终答案。",
            "规则：",
            "1. 如果题目包含“第一/第二/第三/第四”，最终答案必须逐项回答，缺项要从其他候选中补齐。",
            "2. 优先相信包含明确依据和引用页的候选；如果 visual 候选完整且无明显错误，优先保留 visual。",
            "3. 如果某个候选只回答部分问题，不要直接照抄；可以与其他候选合并。",
            "4. 如果候选之间冲突，选择证据页更具体、答案更贴近题目问法的说法。",
            "5. 输出严格 JSON，不要 Markdown，不要额外解释。",
            "",
            "JSON 格式：",
            '{"answer": "最终简短答案", "source_mode": "visual/text/hybrid/fusion/mixed", "reason": "一句话说明为什么这样融合", "citation_pages": [1, 2]}',
            "",
            f"样本ID：{sample_id}",
            f"题目：{question}",
            "",
            "候选答案：",
            cand_text,
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer-level fusion for HomeworkQA QA predictions.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--visual", required=True)
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mode_rows = {
        "text": _by_id(_load_jsonl(args.text)),
        "hybrid": _by_id(_load_jsonl(args.hybrid)),
        "visual": _by_id(_load_jsonl(args.visual)),
        "fusion": _by_id(_load_jsonl(args.fusion)),
    }
    ids = [sample_id for sample_id in mode_rows["visual"] if all(sample_id in rows for rows in mode_rows.values())]
    if not ids:
        print("[ERROR] No overlapping prediction ids.")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        existing = _by_id(_load_jsonl(out_path))

    model = TextJudge(args.judge_model)
    rows_out: List[Dict[str, Any]] = list(existing.values())
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        iterator = tqdm(ids, desc="Answer fusion", unit="q", dynamic_ncols=True, disable=args.no_progress)
        for sample_id in iterator:
            if sample_id in existing:
                continue
            candidates = {mode_name: rows[sample_id] for mode_name, rows in mode_rows.items()}
            prompt = _build_prompt(sample_id, candidates)
            raw = model.judge(prompt, max_new_tokens=args.max_new_tokens)
            parsed = _extract_json(raw)
            answer = str(parsed.get("answer", "")).strip()
            source_mode = str(parsed.get("source_mode", "mixed")).strip() or "mixed"
            citation_pages = parsed.get("citation_pages", [])
            if not isinstance(citation_pages, list):
                citation_pages = []
            try:
                citation_pages = [int(x) for x in citation_pages]
            except Exception:
                citation_pages = []

            base = dict(candidates["visual"])
            gold_answer = str(base.get("gold_answer", "")).strip()
            gold_pages = [int(x) for x in base.get("gold_pages", [])]
            retrieved_pages: List[int] = []
            for row in candidates.values():
                for page in row.get("retrieved_pages", []) or []:
                    try:
                        page_i = int(page)
                    except Exception:
                        continue
                    if page_i not in retrieved_pages:
                        retrieved_pages.append(page_i)
            raw_pred = "\n".join(
                [
                    f"答案：{answer}",
                    f"依据：{parsed.get('reason', '')}",
                    "引用：" + ("、".join(f"第 {p} 页" for p in citation_pages) if citation_pages else "候选答案融合"),
                    f"source_mode={source_mode}",
                    f"fusion_raw={raw}",
                ]
            )
            if not citation_pages:
                citation_pages = _extract_pages_from_text(raw_pred)

            row_out = {
                **base,
                "mode": "answer_fusion",
                "pred_answer": answer,
                "raw_pred": raw_pred,
                "retrieved_pages": retrieved_pages,
                "em": exact_match(answer, gold_answer),
                "f1": token_f1(answer, gold_answer),
                "anls": simple_anls(answer, gold_answer),
                "relaxed_accuracy": relaxed_accuracy(answer, gold_answer),
                "retrieval_recall@1": recall_at_k(retrieved_pages, gold_pages, 1) if gold_pages else 0.0,
                "retrieval_recall@3": recall_at_k(retrieved_pages, gold_pages, 3) if gold_pages else 0.0,
                "retrieval_mrr": mrr(retrieved_pages, gold_pages) if gold_pages else 0.0,
                "citation_accuracy": citation_accuracy(citation_pages, gold_pages) if gold_pages else 0.0,
                "answer_fusion_source_mode": source_mode,
                "answer_fusion_reason": str(parsed.get("reason", "")).strip(),
                "answer_fusion_error": None,
            }
            rows_out.append(row_out)
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "benchmark": "homeworkqa",
        "task": "short_answer_answer_fusion",
        "summary": _summarize(rows_out),
        "by_doc": _summarize_by(rows_out, "doc_id"),
        "by_type": _summarize_by(rows_out, "type"),
        "predictions": str(out_path),
    }
    summary_path = Path(args.summary_out).expanduser().resolve() if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
