"""Tune HomeworkQA visual retrieval with metadata-aware reranking."""

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
from src.docvisrag.retrieve import HybridPageIndex, VisualPageIndex


PageKey = Tuple[str, int]


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


def _char_terms(text: str) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text).lower())
    merged = "".join(chars)
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", merged):
        if len(token) >= 2:
            terms.add(token)
    chinese = re.findall(r"[\u4e00-\u9fff]", merged)
    terms.update(chinese)
    for i in range(len(chinese) - 1):
        terms.add("".join(chinese[i : i + 2]))
    for i in range(len(chinese) - 2):
        terms.add("".join(chinese[i : i + 3]))
    return terms


def _lexical_score(query: str, page_text: str) -> float:
    q_terms = _char_terms(query)
    if not q_terms:
        return 0.0
    p_terms = _char_terms(page_text)
    if not p_terms:
        return 0.0
    overlap = len(q_terms & p_terms) / max(1, len(q_terms))
    q_nums = set(re.findall(r"\d+(?:\.\d+)?", str(query)))
    if q_nums:
        p_nums = set(re.findall(r"\d+(?:\.\d+)?", str(page_text)))
        overlap += 0.5 * (len(q_nums & p_nums) / max(1, len(q_nums)))
    return float(overlap)


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _norm_scores(rows: Sequence[Dict[str, Any]]) -> Dict[PageKey, float]:
    vals = [float(r.get("score", 0.0)) for r in rows]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    out: Dict[PageKey, float] = {}
    for r in rows:
        raw = float(r.get("score", 0.0))
        score = 1.0 if hi <= lo else (raw - lo) / (hi - lo)
        out[_page_key(r)] = score
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune visual retrieval with OCR/summary metadata reranking.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--batch-config", required=True)
    parser.add_argument("--visual-index-dir", required=True)
    parser.add_argument("--hybrid-index-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=0, help="0 means all visual pages.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    questions = _load_jsonl(args.questions)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    scopes = _load_doc_scopes(args.batch_config)
    visual = VisualPageIndex.load(args.visual_index_dir)
    hybrid = HybridPageIndex.load(args.hybrid_index_dir)
    hybrid_by_page = {_page_key(row): row for row in hybrid.metadata}
    candidate_k = len(visual.metadata) if args.candidate_k <= 0 else args.candidate_k

    metrics = {a: {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []} for a in args.alphas}
    details: List[Dict[str, Any]] = []

    iterator = tqdm(questions, desc="Tune visual rerank", unit="q", disable=args.no_progress, dynamic_ncols=True)
    for row in iterator:
        allowed = scopes.get(_question_doc_name(row))
        if not allowed:
            continue
        question = str(row.get("question", "")).strip()
        gold_pages = [int(x) for x in row.get("evidence_pages", [])]
        candidates = _dedup_pages(_filter_rows(visual.search(question, top_k=candidate_k), allowed))
        visual_norm = _norm_scores(candidates)
        candidate_rows: List[Dict[str, Any]] = []
        for rank, cand in enumerate(candidates, start=1):
            key = _page_key(cand)
            hrow = hybrid_by_page.get(key, {})
            page_text = "\n".join(
                [
                    str(hrow.get("summary", "")),
                    str(hrow.get("ocr_text_preview", "")),
                    str(hrow.get("search_text", "")),
                ]
            )
            candidate_rows.append(
                {
                    **cand,
                    "visual_rank": rank,
                    "visual_norm": visual_norm.get(key, 0.0),
                    "lexical_score": _lexical_score(question, page_text),
                }
            )
        item = {
            "id": row.get("id", ""),
            "doc_id": _question_doc_name(row),
            "type": row.get("type", ""),
            "question": question,
            "gold_pages": gold_pages,
            "alphas": {},
        }
        for alpha in args.alphas:
            a = float(alpha)
            ranked = sorted(
                candidate_rows,
                key=lambda x: (
                    -((a * float(x.get("visual_norm", 0.0))) + ((1.0 - a) * float(x.get("lexical_score", 0.0)))),
                    int(x.get("visual_rank", 999999)),
                ),
            )
            pages = [int(x.get("page_index", -1)) for x in ranked[: args.top_k]]
            metrics[alpha]["r1"].append(recall_at_k(pages, gold_pages, 1))
            metrics[alpha]["r3"].append(recall_at_k(pages, gold_pages, 3))
            metrics[alpha]["r5"].append(recall_at_k(pages, gold_pages, 5))
            metrics[alpha]["mrr"].append(mrr(pages, gold_pages))
            metrics[alpha]["ndcg5"].append(ndcg_at_k(pages, gold_pages, 5))
            item["alphas"][str(alpha)] = {"pages": pages}
        details.append(item)

    summary = {
        str(alpha): {
            "visual_weight_alpha": float(alpha),
            "metadata_weight": float(1.0 - alpha),
            "recall@1": _avg(vals["r1"]),
            "recall@3": _avg(vals["r3"]),
            "recall@5": _avg(vals["r5"]),
            "mrr": _avg(vals["mrr"]),
            "ndcg@5": _avg(vals["ndcg5"]),
        }
        for alpha, vals in metrics.items()
    }
    payload = {
        "config": {
            "visual_index_dir": str(Path(args.visual_index_dir)),
            "hybrid_index_dir": str(Path(args.hybrid_index_dir)),
            "top_k": args.top_k,
            "candidate_k": candidate_k,
            "alphas": args.alphas,
        },
        "summary": summary,
        "details": details,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
