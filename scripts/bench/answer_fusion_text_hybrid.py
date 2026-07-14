"""Fuse text and hybrid HomeworkQA predictions without gold labels."""

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
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id", "")): row for row in rows}


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    for cand in [raw] + re.findall(r"\{.*?\}", raw, flags=re.S):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {"answer": raw, "source_mode": "mixed", "reason": "failed_to_parse_json", "citation_pages": []}


def _extract_pages(text: str) -> List[int]:
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


def _prompt(sample_id: str, text_row: Mapping[str, Any], hybrid_row: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "你是 HomeworkQA 简答题的 hybrid 答案融合器。",
            "你只能根据题目、text 候选答案和 hybrid 候选答案进行选择或合并，不要使用标准答案，不要引入外部知识。",
            "目标：生成一个比单独 text 或单独 hybrid 更完整、更准确的简短答案。",
            "规则：",
            "1. 如果题目包含“第一/第二/第三/第四”，最终答案必须逐项回答。",
            "2. hybrid 候选通常有更强的检索排序；text 候选可能有更完整表述。两者一致时合并为更简短答案。",
            "3. 如果 hybrid 只回答部分问题，而 text 补充了缺项，应合并；反之亦然。",
            "4. 如果二者冲突，选择引用页和题目更一致、字段更具体的一方。",
            "5. 输出严格 JSON。",
            "",
            '{"answer": "最终简短答案", "source_mode": "text/hybrid/mixed", "reason": "一句话理由", "citation_pages": [1, 2]}',
            "",
            f"样本ID：{sample_id}",
            f"题目：{text_row.get('question', '')}",
            "",
            "[text 候选]",
            f"答案：{_truncate(str(text_row.get('pred_answer', '')), 450)}",
            f"完整输出：{_truncate(str(text_row.get('raw_pred', '')), 900)}",
            f"检索页：{text_row.get('retrieved_pages', [])}",
            "",
            "[hybrid 候选]",
            f"答案：{_truncate(str(hybrid_row.get('pred_answer', '')), 450)}",
            f"完整输出：{_truncate(str(hybrid_row.get('raw_pred', '')), 900)}",
            f"检索页：{hybrid_row.get('retrieved_pages', [])}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    text_rows = _by_id(_load_jsonl(args.text))
    hybrid_rows = _by_id(_load_jsonl(args.hybrid))
    ids = [sample_id for sample_id in text_rows if sample_id in hybrid_rows]
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.exists():
        existing = _by_id(_load_jsonl(out_path))

    judge = TextJudge(args.judge_model)
    outputs: List[Dict[str, Any]] = list(existing.values())
    mode = "a" if args.resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for sample_id in tqdm(ids, desc="Hybrid answer fusion", unit="q", dynamic_ncols=True, disable=args.no_progress):
            if sample_id in existing:
                continue
            text_row = text_rows[sample_id]
            hybrid_row = hybrid_rows[sample_id]
            raw = judge.judge(_prompt(sample_id, text_row, hybrid_row), max_new_tokens=args.max_new_tokens)
            parsed = _extract_json(raw)
            answer = str(parsed.get("answer", "")).strip()
            citation_pages = parsed.get("citation_pages", [])
            if not isinstance(citation_pages, list):
                citation_pages = []
            try:
                citation_pages = [int(x) for x in citation_pages]
            except Exception:
                citation_pages = []

            base = dict(hybrid_row)
            gold = str(base.get("gold_answer", "")).strip()
            gold_pages = [int(x) for x in base.get("gold_pages", [])]
            retrieved_pages: List[int] = []
            for row in [hybrid_row, text_row]:
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
                    "引用：" + ("、".join(f"第 {p} 页" for p in citation_pages) if citation_pages else "text/hybrid候选融合"),
                    f"fusion_raw={raw}",
                ]
            )
            if not citation_pages:
                citation_pages = _extract_pages(raw_pred)

            out = {
                **base,
                "mode": "hybrid_answer_fusion",
                "pred_answer": answer,
                "raw_pred": raw_pred,
                "retrieved_pages": retrieved_pages,
                "em": exact_match(answer, gold),
                "f1": token_f1(answer, gold),
                "anls": simple_anls(answer, gold),
                "relaxed_accuracy": relaxed_accuracy(answer, gold),
                "retrieval_recall@1": recall_at_k(retrieved_pages, gold_pages, 1) if gold_pages else 0.0,
                "retrieval_recall@3": recall_at_k(retrieved_pages, gold_pages, 3) if gold_pages else 0.0,
                "retrieval_mrr": mrr(retrieved_pages, gold_pages) if gold_pages else 0.0,
                "citation_accuracy": citation_accuracy(citation_pages, gold_pages) if gold_pages else 0.0,
                "hybrid_answer_fusion_source": str(parsed.get("source_mode", "mixed")),
            }
            outputs.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    summary = {"benchmark": "homeworkqa", "task": "hybrid_answer_fusion", "summary": _summarize(outputs), "predictions": str(out_path)}
    summary_path = Path(args.summary_out).expanduser().resolve() if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
