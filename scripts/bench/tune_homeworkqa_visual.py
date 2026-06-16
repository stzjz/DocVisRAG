"""Try lightweight query rewrites for HomeworkQA visual retrieval."""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.eval import mrr, ndcg_at_k, recall_at_k
from src.docvisrag.retrieve import VisualPageIndex


PageKey = Tuple[str, int]


STOPWORDS = [
    "是什么",
    "哪些",
    "哪个",
    "多少",
    "如何",
    "是否",
    "这份",
    "这个",
    "中的",
    "中",
    "的",
    "请问",
    "包含",
    "展示",
    "主要",
]

TYPE_HINTS = {
    "table": "表格 列 行",
    "chart": "图表 坐标 颜色 趋势",
    "formula": "公式 符号 推导 手写",
    "calculation": "计算 数值 公式",
    "visual": "图形 图示 手写 电路 结构",
    "procedure": "步骤 流程 算法",
    "comparison": "对比 两页 差异",
    "summary": "标题 小节 页面布局",
}


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _page_key(row: Mapping[str, Any]) -> PageKey:
    return str(row.get("doc_id", "")).strip(), int(row.get("page_index", -1))


def _question_doc_name(question: Mapping[str, Any]) -> str:
    doc_id = str(question.get("doc_id", "")).strip()
    if doc_id:
        return doc_id
    doc_path = str(question.get("doc_path", "")).strip()
    if doc_path:
        return Path(doc_path).parent.name
    source_pdf = str(question.get("source_pdf", "")).strip()
    if source_pdf:
        return Path(source_pdf).stem
    return ""


def _load_doc_scopes(batch_config: str | Path) -> Dict[str, set[PageKey]]:
    cfg = _load_json(batch_config)
    scopes: Dict[str, set[PageKey]] = {}
    for item in cfg.get("benchmarks", []):
        if not isinstance(item, dict):
            continue
        manifest = str(item.get("manifest", "")).strip()
        if not manifest:
            continue
        name = str(item.get("name", "")).strip()
        doc_name = name.removeprefix("homeworkqa_") if name else Path(manifest).parent.name
        pages = _load_json(manifest)
        allowed: set[PageKey] = set()
        for page in pages if isinstance(pages, list) else []:
            if isinstance(page, dict):
                allowed.add((str(page.get("doc_id", "")).strip(), int(page.get("page_index", -1))))
        if allowed:
            scopes[doc_name] = allowed
    return scopes


def _filter_rows(rows: Iterable[Dict[str, Any]], allowed_pages: set[PageKey]) -> List[Dict[str, Any]]:
    return [row for row in rows if _page_key(row) in allowed_pages]


def _dedup_pages(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[PageKey] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = _page_key(row)
        if key[1] <= 0 or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _clean_question(question: str) -> str:
    text = str(question).strip()
    for word in STOPWORDS:
        text = text.replace(word, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rewrite_query(row: Mapping[str, Any], mode: str) -> str:
    question = str(row.get("question", "")).strip()
    doc_name = _question_doc_name(row)
    qtype = str(row.get("type", "")).strip()
    hint = TYPE_HINTS.get(qtype, "")
    cleaned = _clean_question(question)
    if mode == "raw":
        return question
    if mode == "clean":
        return cleaned or question
    if mode == "doc_clean":
        return " ".join(x for x in [doc_name, cleaned] if x)
    if mode == "visual_hint":
        return " ".join(x for x in [hint, cleaned] if x) or question
    if mode == "doc_visual_hint":
        return " ".join(x for x in [doc_name, hint, cleaned] if x) or question
    raise ValueError(f"unknown mode: {mode}")


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune HomeworkQA visual retrieval query rewrites.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--batch-config", required=True)
    parser.add_argument("--visual-index-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=0, help="0 means all visual pages.")
    parser.add_argument(
        "--rewrite-modes",
        nargs="+",
        default=["raw", "clean", "doc_clean", "visual_hint", "doc_visual_hint"],
        choices=["raw", "clean", "doc_clean", "visual_hint", "doc_visual_hint"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    questions = _load_jsonl(args.questions)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    scopes = _load_doc_scopes(args.batch_config)
    index = VisualPageIndex.load(args.visual_index_dir)
    candidate_k = len(index.metadata) if args.candidate_k <= 0 else args.candidate_k

    details: List[Dict[str, Any]] = []
    metrics = {
        mode: {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []}
        for mode in args.rewrite_modes
    }

    iterator = tqdm(questions, desc="Tune HomeworkQA visual", unit="q", disable=args.no_progress, dynamic_ncols=True)
    for row in iterator:
        doc_name = _question_doc_name(row)
        allowed = scopes.get(doc_name)
        if not allowed:
            continue
        gold_pages = [int(x) for x in row.get("evidence_pages", [])]
        item = {
            "id": row.get("id", ""),
            "doc_id": doc_name,
            "type": row.get("type", ""),
            "question": row.get("question", ""),
            "gold_pages": gold_pages,
            "modes": {},
        }
        for mode in args.rewrite_modes:
            query = rewrite_query(row, mode)
            results = _dedup_pages(_filter_rows(index.search(query, top_k=candidate_k), allowed))
            pages = [int(x.get("page_index", -1)) for x in results[: args.top_k]]
            metrics[mode]["r1"].append(recall_at_k(pages, gold_pages, 1))
            metrics[mode]["r3"].append(recall_at_k(pages, gold_pages, 3))
            metrics[mode]["r5"].append(recall_at_k(pages, gold_pages, 5))
            metrics[mode]["mrr"].append(mrr(pages, gold_pages))
            metrics[mode]["ndcg5"].append(ndcg_at_k(pages, gold_pages, 5))
            item["modes"][mode] = {"query": query, "pages": pages}
        details.append(item)

    summary = {
        mode: {
            "recall@1": _avg(vals["r1"]),
            "recall@3": _avg(vals["r3"]),
            "recall@5": _avg(vals["r5"]),
            "mrr": _avg(vals["mrr"]),
            "ndcg@5": _avg(vals["ndcg5"]),
        }
        for mode, vals in metrics.items()
    }
    payload = {
        "config": {
            "visual_index_dir": str(Path(args.visual_index_dir)),
            "top_k": args.top_k,
            "candidate_k": candidate_k,
            "rewrite_modes": args.rewrite_modes,
        },
        "summary": summary,
        "details": details,
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
