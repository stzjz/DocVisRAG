"""Evaluate HomeworkQA on a combined index while preserving per-PDF scope.

HomeworkQA questions target a full PDF, not a single rendered page and not the
whole 10-PDF collection. This script loads the combined text/hybrid/visual
indexes once, then filters retrieval candidates to the PDF referenced by each
question before computing metrics.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.eval import mrr, ndcg_at_k, recall_at_k
from homeworkqa_visual_utils import metadata_rerank_visual_pages
from src.docvisrag.retrieve import (
    HybridPageIndex,
    TextIndex,
    VisualPageIndex,
    text_chunks_to_page_results,
    weighted_reciprocal_rank_fusion,
)


PageKey = Tuple[str, int]


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_no} in {path} is not an object.")
            rows.append(row)
    return rows


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _page_key(row: Mapping[str, Any]) -> PageKey:
    return str(row.get("doc_id", "")).strip(), int(row.get("page_index", -1))


def _filter_rows(rows: Iterable[Dict[str, Any]], allowed_pages: set[PageKey]) -> List[Dict[str, Any]]:
    return [row for row in rows if _page_key(row) in allowed_pages]


def _dedup_pages(rows: Iterable[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
    seen: set[PageKey] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = _page_key(row)
        if key[1] <= 0 or key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
        if top_k is not None and len(out) >= top_k:
            break
    return out


def _candidate_depth(requested: int | None, total: int, fallback: int) -> int:
    if requested is None or requested <= 0:
        return total
    return max(1, min(total, max(requested, fallback)))


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



def _strip_homeworkqa_prefix(name: str) -> str:
    for prefix in ["homeworkqa_hard_mc_", "homeworkqa_hard_", "homeworkqa_mc_", "homeworkqa_"]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name

def _load_doc_scopes(batch_config: str | Path) -> Dict[str, set[PageKey]]:
    cfg = _load_json(batch_config)
    benchmarks = cfg.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        raise ValueError("batch_config.json must contain a benchmarks list.")

    scopes: Dict[str, set[PageKey]] = {}
    for item in benchmarks:
        if not isinstance(item, dict):
            continue
        manifest = str(item.get("manifest", "")).strip()
        if not manifest:
            continue
        name = str(item.get("name", "")).strip()
        doc_name = _strip_homeworkqa_prefix(name) if name else Path(manifest).parent.name
        pages = _load_json(manifest)
        if not isinstance(pages, list):
            raise ValueError(f"Manifest must be a list: {manifest}")
        allowed: set[PageKey] = set()
        for page in pages:
            if not isinstance(page, dict):
                continue
            allowed.add((str(page.get("doc_id", "")).strip(), int(page.get("page_index", -1))))
        if allowed:
            scopes[doc_name] = allowed
    return scopes


def _coverage_at_k(retrieved: Sequence[int], gold_pages: Sequence[int], k: int) -> float:
    gold = {int(x) for x in gold_pages}
    if not gold:
        return 0.0
    got = {int(x) for x in list(retrieved)[:k]}
    return len(gold & got) / len(gold)


def _all_evidence_at_k(retrieved: Sequence[int], gold_pages: Sequence[int], k: int) -> float:
    gold = {int(x) for x in gold_pages}
    if not gold:
        return 0.0
    got = {int(x) for x in list(retrieved)[:k]}
    return 1.0 if gold.issubset(got) else 0.0


def _metric_row(retrieved: Sequence[int], gold_pages: Sequence[int]) -> Dict[str, float]:
    return {
        "recall@1": recall_at_k(list(retrieved), list(gold_pages), 1),
        "recall@3": recall_at_k(list(retrieved), list(gold_pages), 3),
        "recall@5": recall_at_k(list(retrieved), list(gold_pages), 5),
        "evidence_coverage@3": _coverage_at_k(retrieved, gold_pages, 3),
        "evidence_coverage@5": _coverage_at_k(retrieved, gold_pages, 5),
        "all_evidence@3": _all_evidence_at_k(retrieved, gold_pages, 3),
        "all_evidence@5": _all_evidence_at_k(retrieved, gold_pages, 5),
        "mrr": mrr(list(retrieved), list(gold_pages)),
        "ndcg@5": ndcg_at_k(list(retrieved), list(gold_pages), 5),
    }


def _summarize(details: Sequence[Dict[str, Any]], mode: str, group_field: str | None = None) -> Dict[str, Any]:
    rows = [row for row in details if mode in row.get("metrics", {})]
    if group_field is None:
        return {
            "num_questions": len(rows),
            "recall@1": _avg([row["metrics"][mode]["recall@1"] for row in rows]),
            "recall@3": _avg([row["metrics"][mode]["recall@3"] for row in rows]),
            "recall@5": _avg([row["metrics"][mode]["recall@5"] for row in rows]),
            "evidence_coverage@3": _avg([row["metrics"][mode].get("evidence_coverage@3", 0.0) for row in rows]),
            "evidence_coverage@5": _avg([row["metrics"][mode].get("evidence_coverage@5", 0.0) for row in rows]),
            "all_evidence@3": _avg([row["metrics"][mode].get("all_evidence@3", 0.0) for row in rows]),
            "all_evidence@5": _avg([row["metrics"][mode].get("all_evidence@5", 0.0) for row in rows]),
            "mrr": _avg([row["metrics"][mode]["mrr"] for row in rows]),
            "ndcg@5": _avg([row["metrics"][mode]["ndcg@5"] for row in rows]),
        }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)

    out: Dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        out[group] = {
            "num_questions": len(group_rows),
            "recall@1": _avg([row["metrics"][mode]["recall@1"] for row in group_rows]),
            "recall@3": _avg([row["metrics"][mode]["recall@3"] for row in group_rows]),
            "recall@5": _avg([row["metrics"][mode]["recall@5"] for row in group_rows]),
            "evidence_coverage@3": _avg([row["metrics"][mode].get("evidence_coverage@3", 0.0) for row in group_rows]),
            "evidence_coverage@5": _avg([row["metrics"][mode].get("evidence_coverage@5", 0.0) for row in group_rows]),
            "all_evidence@3": _avg([row["metrics"][mode].get("all_evidence@3", 0.0) for row in group_rows]),
            "all_evidence@5": _avg([row["metrics"][mode].get("all_evidence@5", 0.0) for row in group_rows]),
            "mrr": _avg([row["metrics"][mode]["mrr"] for row in group_rows]),
            "ndcg@5": _avg([row["metrics"][mode]["ndcg@5"] for row in group_rows]),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HomeworkQA retrieval with combined indexes and per-PDF candidate filtering."
    )
    parser.add_argument("--questions", required=True, help="HomeworkQA questions.jsonl")
    parser.add_argument("--batch-config", required=True, help="HomeworkQA batch_config.json")
    parser.add_argument("--text-index-dir", required=True, help="Combined text index directory")
    parser.add_argument("--hybrid-index-dir", required=True, help="Combined hybrid index directory")
    parser.add_argument("--visual-index-dir", required=True, help="Combined visual index directory")
    parser.add_argument("--out", required=True, help="Output JSON report")
    parser.add_argument("--top-k", type=int, default=5, help="Metric top-k; default: 5")
    parser.add_argument("--modes", nargs="+", default=["text", "hybrid", "visual", "fusion"], choices=["text", "hybrid", "visual", "fusion"])
    parser.add_argument("--fusion-text-weight", type=float, default=0.2)
    parser.add_argument("--fusion-hybrid-weight", type=float, default=5.0)
    parser.add_argument("--fusion-visual-weight", type=float, default=5.0)
    parser.add_argument("--fusion-text-candidates", type=int, default=0, help="0 means all indexed text chunks.")
    parser.add_argument("--fusion-hybrid-candidates", type=int, default=0, help="0 means all indexed pages.")
    parser.add_argument("--fusion-visual-candidates", type=int, default=0, help="0 means all indexed pages.")
    parser.add_argument("--visual-rerank-mode", default="metadata", choices=["none", "metadata"])
    parser.add_argument("--visual-rerank-alpha", type=float, default=0.1, help="Visual score weight for metadata rerank; 0 means metadata-only.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    modes = set(args.modes)
    if args.top_k <= 0:
        print("[ERROR] --top-k must be positive.")
        return 1

    try:
        questions = _load_jsonl(args.questions)
        scopes = _load_doc_scopes(args.batch_config)
        text_index = TextIndex.load(args.text_index_dir) if modes & {"text", "fusion"} else None
        hybrid_index = HybridPageIndex.load(args.hybrid_index_dir) if (modes & {"hybrid", "fusion"} or args.visual_rerank_mode == "metadata") else None
        visual_index = VisualPageIndex.load(args.visual_index_dir) if modes & {"visual", "fusion"} else None
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Init HomeworkQA evaluator failed: {exc}")
        return 1

    if not questions:
        print("[ERROR] No questions found.")
        return 1

    text_total = len(text_index.metadata) if text_index is not None else 0
    hybrid_total = len(hybrid_index.metadata) if hybrid_index is not None else 0
    visual_total = len(visual_index.metadata) if visual_index is not None else 0
    hybrid_meta_by_page = {_page_key(r): r for r in hybrid_index.metadata} if hybrid_index is not None else {}
    text_candidate_k = _candidate_depth(args.fusion_text_candidates, text_total, args.top_k * 20)
    hybrid_candidate_k = _candidate_depth(args.fusion_hybrid_candidates, hybrid_total, args.top_k * 10)
    visual_candidate_k = _candidate_depth(args.fusion_visual_candidates, visual_total, args.top_k * 10)

    details: List[Dict[str, Any]] = []
    iterator = tqdm(
        questions,
        desc="HomeworkQA combined retrieval",
        unit="q",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    for q in iterator:
        qid = str(q.get("id", ""))
        question = str(q.get("question", "")).strip()
        doc_name = _question_doc_name(q)
        allowed_pages = scopes.get(doc_name)
        if not question or not allowed_pages:
            continue

        gold_pages = [int(x) for x in q.get("evidence_pages", [])]
        row: Dict[str, Any] = {
            "id": qid,
            "doc_id": doc_name,
            "type": str(q.get("type", "text")),
            "question": question,
            "gold_pages": gold_pages,
            "retrieved": {},
            "metrics": {},
        }

        try:
            text_page_results: List[Dict[str, Any]] = []
            hybrid_results: List[Dict[str, Any]] = []
            visual_results: List[Dict[str, Any]] = []

            if text_index is not None:
                text_chunks = text_index.search(question, top_k=text_candidate_k)
                text_chunks = _filter_rows(text_chunks, allowed_pages)
                text_page_results = text_chunks_to_page_results(text_chunks, top_k=None)
                text_page_results = _dedup_pages(text_page_results)
                if "text" in modes:
                    pages = [int(x.get("page_index", -1)) for x in text_page_results[: args.top_k]]
                    row["retrieved"]["text"] = pages
                    row["metrics"]["text"] = _metric_row(pages, gold_pages)

            if hybrid_index is not None:
                hybrid_raw = hybrid_index.search(question, top_k=hybrid_candidate_k)
                hybrid_results = _dedup_pages(_filter_rows(hybrid_raw, allowed_pages))
                if "hybrid" in modes:
                    pages = [int(x.get("page_index", -1)) for x in hybrid_results[: args.top_k]]
                    row["retrieved"]["hybrid"] = pages
                    row["metrics"]["hybrid"] = _metric_row(pages, gold_pages)

            if visual_index is not None:
                visual_raw = visual_index.search(question, top_k=visual_candidate_k)
                visual_results = _dedup_pages(_filter_rows(visual_raw, allowed_pages))
                if args.visual_rerank_mode == "metadata":
                    visual_results = metadata_rerank_visual_pages(
                        visual_results,
                        query=question,
                        hybrid_meta_by_page=hybrid_meta_by_page,
                        alpha=args.visual_rerank_alpha,
                    )
                if "visual" in modes:
                    pages = [int(x.get("page_index", -1)) for x in visual_results[: args.top_k]]
                    row["retrieved"]["visual"] = pages
                    row["metrics"]["visual"] = _metric_row(pages, gold_pages)

            if "fusion" in modes:
                if text_index is None or hybrid_index is None or visual_index is None:
                    raise RuntimeError("fusion mode requires text, hybrid, and visual indexes.")
                hybrid_by_page = {_page_key(r): r for r in hybrid_results}
                for page_row in text_page_results:
                    hrow = hybrid_by_page.get(_page_key(page_row))
                    if hrow:
                        for field in ["image_path", "summary", "ocr_text_preview"]:
                            if not page_row.get(field) and hrow.get(field):
                                page_row[field] = hrow.get(field)
                fusion_results = weighted_reciprocal_rank_fusion(
                    {
                        "text": text_page_results,
                        "hybrid": hybrid_results,
                        "visual": visual_results,
                    },
                    weights={
                        "text": args.fusion_text_weight,
                        "hybrid": args.fusion_hybrid_weight,
                        "visual": args.fusion_visual_weight,
                    },
                    top_k=args.top_k,
                )
                pages = [int(x.get("page_index", -1)) for x in fusion_results]
                row["retrieved"]["fusion"] = pages
                row["metrics"]["fusion"] = _metric_row(pages, gold_pages)
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)

        details.append(row)

    evaluated = [row for row in details if row.get("metrics")]
    if not evaluated:
        print("[ERROR] No question was evaluated.")
        return 1

    summary: Dict[str, Any] = {}
    for mode in ["text", "hybrid", "visual", "fusion"]:
        if mode in modes:
            summary[mode] = _summarize(evaluated, mode)

    by_doc: Dict[str, Any] = {}
    by_type: Dict[str, Any] = {}
    for mode in ["text", "hybrid", "visual", "fusion"]:
        if mode in modes:
            by_doc[mode] = _summarize(evaluated, mode, group_field="doc_id")
            by_type[mode] = _summarize(evaluated, mode, group_field="type")

    report = {
        "benchmark": "homeworkqa",
        "scope": "combined_index_with_per_pdf_filter",
        "num_questions": len(evaluated),
        "config": {
            "questions": str(Path(args.questions)),
            "batch_config": str(Path(args.batch_config)),
            "text_index_dir": str(Path(args.text_index_dir)),
            "hybrid_index_dir": str(Path(args.hybrid_index_dir)),
            "visual_index_dir": str(Path(args.visual_index_dir)),
            "top_k": args.top_k,
            "modes": sorted(modes),
            "fusion_weights": {
                "text": args.fusion_text_weight,
                "hybrid": args.fusion_hybrid_weight,
                "visual": args.fusion_visual_weight,
            },
            "candidate_depth": {
                "text": text_candidate_k,
                "hybrid": hybrid_candidate_k,
                "visual": visual_candidate_k,
            },
            "visual_rerank": {
                "mode": args.visual_rerank_mode,
                "alpha": args.visual_rerank_alpha,
            },
        },
        "summary": summary,
        "by_doc": by_doc,
        "by_type": by_type,
        "details": details,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"num_questions": len(evaluated), "summary": summary}, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
