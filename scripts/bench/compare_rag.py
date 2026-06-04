"""纯文本 RAG vs 多模态 RAG 对比评测脚本。

运行纯文本（仅 OCR）和多模态（hybrid/fusion）两组管线，
输出对比报告（Recall@K / MRR / NDCG / EM / F1 / ANLS）。
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.docvisrag.eval import citation_accuracy, exact_match, mrr, ndcg_at_k, recall_at_k, simple_anls, token_f1
from src.docvisrag.qa import DocQAEngine, TextDocQAEngine
from src.docvisrag.retrieve import (
    HybridPageIndex,
    TextIndex,
    VisualPageIndex,
    reciprocal_rank_fusion,
    text_chunks_to_page_results,
    weighted_reciprocal_rank_fusion,
)


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


def _avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _dedup_pages(chunks: List[Dict]) -> List[Dict]:
    seen: set = set()
    results: List[Dict] = []
    for c in chunks:
        page = int(c.get("page_index", -1))
        if page > 0 and page not in seen:
            seen.add(page)
            results.append({"page_index": page, "score": c.get("score", 0.0)})
    return results


def _extract_pages_from_citations(citations: List[str]) -> List[int]:
    import re
    pages: List[int] = []
    for c in citations:
        for num in re.findall(r"\d+", str(c)):
            pages.append(int(num))
    uniq: List[int] = []
    seen = set()
    for p in pages:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def run_retrieval_comparison(
    questions: List[Dict],
    text_index: TextIndex,
    hybrid_index: HybridPageIndex,
    visual_index: VisualPageIndex | None,
    top_k: int = 5,
) -> Dict[str, Any]:
    modes = {
        "text": {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []},
        "hybrid": {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []},
    }
    if visual_index is not None:
        modes["visual"] = {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []}
        modes["fusion"] = {"r1": [], "r3": [], "r5": [], "mrr": [], "ndcg5": []}

    details: List[Dict] = []

    for q in questions:
        question = str(q.get("question", "")).strip()
        gold_pages = [int(x) for x in q.get("evidence_pages", [])]
        if not question:
            continue

        row: Dict[str, Any] = {"id": q.get("id", ""), "question": question, "gold_pages": gold_pages}

        # --- text retrieval ---
        text_chunks = text_index.search(question, top_k=max(50, top_k * 6))
        text_results = _dedup_pages(text_chunks)
        text_pages = [r["page_index"] for r in text_results[:top_k]]
        row["text_pages"] = text_pages
        for k, key in [(1, "r1"), (3, "r3"), (5, "r5")]:
            modes["text"][key].append(recall_at_k(text_pages, gold_pages, k))
        modes["text"]["mrr"].append(mrr(text_pages, gold_pages))
        modes["text"]["ndcg5"].append(ndcg_at_k(text_pages, gold_pages, 5))

        # --- hybrid retrieval ---
        hybrid_results = hybrid_index.search(question, top_k=top_k)
        hybrid_pages = [int(r.get("page_index", -1)) for r in hybrid_results]
        row["hybrid_pages"] = hybrid_pages
        for k, key in [(1, "r1"), (3, "r3"), (5, "r5")]:
            modes["hybrid"][key].append(recall_at_k(hybrid_pages, gold_pages, k))
        modes["hybrid"]["mrr"].append(mrr(hybrid_pages, gold_pages))
        modes["hybrid"]["ndcg5"].append(ndcg_at_k(hybrid_pages, gold_pages, 5))

        # --- visual / fusion ---
        if visual_index is not None:
            visual_raw = visual_index.search(question, top_k=max(10, top_k * 2))
            visual_pages_raw = [int(r.get("page_index", -1)) for r in visual_raw]
            row["visual_pages"] = visual_pages_raw[:top_k]
            for k, key in [(1, "r1"), (3, "r3"), (5, "r5")]:
                modes["visual"][key].append(recall_at_k(visual_pages_raw[:top_k], gold_pages, k))
            modes["visual"]["mrr"].append(mrr(visual_pages_raw[:top_k], gold_pages))
            modes["visual"]["ndcg5"].append(ndcg_at_k(visual_pages_raw[:top_k], gold_pages, 5))

            text_page_results = text_chunks_to_page_results(text_chunks, top_k=max(10, top_k * 2))
            hybrid_by_page = {
                (str(r.get("doc_id", "")), int(r.get("page_index", -1))): r
                for r in hybrid_results
            }
            for page_row in text_page_results:
                hrow = hybrid_by_page.get((str(page_row.get("doc_id", "")), int(page_row.get("page_index", -1))))
                if hrow:
                    for field in ["image_path", "summary", "ocr_text_preview"]:
                        if not page_row.get(field) and hrow.get(field):
                            page_row[field] = hrow.get(field)
            fusion_results = weighted_reciprocal_rank_fusion(
                {"text": text_page_results, "hybrid": hybrid_results, "visual": visual_raw},
                weights={"text": 2.0, "hybrid": 0.3, "visual": 0.1},
                top_k=top_k,
            )
            fusion_pages = [int(r.get("page_index", -1)) for r in fusion_results]
            row["fusion_pages"] = fusion_pages
            for k, key in [(1, "r1"), (3, "r3"), (5, "r5")]:
                modes["fusion"][key].append(recall_at_k(fusion_pages, gold_pages, k))
            modes["fusion"]["mrr"].append(mrr(fusion_pages, gold_pages))
            modes["fusion"]["ndcg5"].append(ndcg_at_k(fusion_pages, gold_pages, 5))

        details.append(row)

    summary: Dict[str, Any] = {}
    for mode_name, metrics in modes.items():
        summary[mode_name] = {
            "recall@1": _avg(metrics["r1"]),
            "recall@3": _avg(metrics["r3"]),
            "recall@5": _avg(metrics["r5"]),
            "mrr": _avg(metrics["mrr"]),
            "ndcg@5": _avg(metrics["ndcg5"]),
        }

    return {"summary": summary, "details": details}


def run_qa_comparison(
    questions: List[Dict],
    text_engine: TextDocQAEngine,
    multimodal_engine: DocQAEngine,
) -> Dict[str, Any]:
    modes = {
        "text": {"em": [], "f1": [], "anls": [], "r3": [], "cite_acc": [], "latency": []},
        "multimodal": {"em": [], "f1": [], "anls": [], "r3": [], "cite_acc": [], "latency": []},
    }
    predictions: List[Dict] = []

    for q in questions:
        question = str(q.get("question", "")).strip()
        gold_answer = str(q.get("answer", "")).strip()
        gold_pages = [int(x) for x in q.get("evidence_pages", [])]
        if not question:
            continue

        row: Dict[str, Any] = {
            "id": q.get("id", ""),
            "type": q.get("type", "text"),
            "question": question,
            "gold_answer": gold_answer,
            "gold_pages": gold_pages,
        }

        # --- text QA ---
        t_start = time.perf_counter()
        try:
            t_result = text_engine.answer(question)
            t_answer = t_result.answer
            t_evidence_pages = [int(x.get("page_index", -1)) for x in t_result.evidence if x.get("page_index") is not None]
            t_cite_pages = _extract_pages_from_citations(t_result.citations)
        except Exception as exc:
            t_answer = ""
            t_evidence_pages = []
            t_cite_pages = []
        t_latency = round(time.perf_counter() - t_start, 3)

        row["text_answer"] = t_answer
        row["text_cite_pages"] = t_cite_pages
        modes["text"]["em"].append(exact_match(t_answer, gold_answer))
        modes["text"]["f1"].append(token_f1(t_answer, gold_answer))
        modes["text"]["anls"].append(simple_anls(t_answer, gold_answer))
        modes["text"]["r3"].append(recall_at_k(t_evidence_pages, gold_pages, 3) if gold_pages else 0.0)
        modes["text"]["cite_acc"].append(citation_accuracy(t_cite_pages, gold_pages) if gold_pages else 0.0)
        modes["text"]["latency"].append(t_latency)

        # --- multimodal QA ---
        m_start = time.perf_counter()
        try:
            m_result = multimodal_engine.answer(question)
            m_answer = m_result.answer
            m_evidence_pages = [int(x.get("page_index", -1)) for x in m_result.evidence if x.get("page_index") is not None]
            m_cite_pages = _extract_pages_from_citations(m_result.citations)
        except Exception as exc:
            m_answer = ""
            m_evidence_pages = []
            m_cite_pages = []
        m_latency = round(time.perf_counter() - m_start, 3)

        row["multimodal_answer"] = m_answer
        row["multimodal_cite_pages"] = m_cite_pages
        modes["multimodal"]["em"].append(exact_match(m_answer, gold_answer))
        modes["multimodal"]["f1"].append(token_f1(m_answer, gold_answer))
        modes["multimodal"]["anls"].append(simple_anls(m_answer, gold_answer))
        modes["multimodal"]["r3"].append(recall_at_k(m_evidence_pages, gold_pages, 3) if gold_pages else 0.0)
        modes["multimodal"]["cite_acc"].append(citation_accuracy(m_cite_pages, gold_pages) if gold_pages else 0.0)
        modes["multimodal"]["latency"].append(m_latency)

        predictions.append(row)

    summary: Dict[str, Any] = {}
    for mode_name, metrics in modes.items():
        summary[mode_name] = {
            "em": _avg(metrics["em"]),
            "f1": _avg(metrics["f1"]),
            "anls": _avg(metrics["anls"]),
            "recall@3": _avg(metrics["r3"]),
            "citation_accuracy": _avg(metrics["cite_acc"]),
            "avg_latency_seconds": _avg(metrics["latency"]),
        }

    return {"summary": summary, "predictions": predictions}


def _print_retrieval_table(retrieval_summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("检索对比结果 (Recall@K / MRR / NDCG@5)")
    print("=" * 80)
    header = f"{'模式':<12} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR':>8} {'NDCG@5':>8}"
    print(header)
    print("-" * len(header))
    for mode in ["text", "hybrid", "visual", "fusion"]:
        if mode not in retrieval_summary:
            continue
        m = retrieval_summary[mode]
        print(
            f"{mode:<12} "
            f"{m['recall@1']:>8.4f} "
            f"{m['recall@3']:>8.4f} "
            f"{m['recall@5']:>8.4f} "
            f"{m['mrr']:>8.4f} "
            f"{m['ndcg@5']:>8.4f}"
        )


def _print_qa_table(qa_summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("问答对比结果 (EM / F1 / ANLS / Recall@3 / CitationAcc)")
    print("=" * 80)
    header = f"{'模式':<14} {'EM':>8} {'F1':>8} {'ANLS':>8} {'R@3':>8} {'CiteAcc':>8} {'Latency':>8}"
    print(header)
    print("-" * len(header))
    for mode in ["text", "multimodal"]:
        if mode not in qa_summary:
            continue
        m = qa_summary[mode]
        print(
            f"{mode:<14} "
            f"{m['em']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{m['anls']:>8.4f} "
            f"{m['recall@3']:>8.4f} "
            f"{m['citation_accuracy']:>8.4f} "
            f"{m['avg_latency_seconds']:>7.2f}s"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="纯文本 RAG vs 多模态 RAG 对比评测。"
    )
    parser.add_argument("--questions", required=True, help="评测问题 JSONL 文件。")
    parser.add_argument("--text-index-dir", required=True, help="纯文本索引目录（由 build_text_index.py 构建）。")
    parser.add_argument("--hybrid-index-dir", required=True, help="Hybrid 多模态索引目录。")
    parser.add_argument("--visual-index-dir", default=None, help="可选 visual 索引目录，用于对比 visual/fusion。")
    parser.add_argument("--multimodal-type", default="hybrid", choices=["hybrid", "fusion"],
                        help="多模态检索器类型（默认 hybrid）。")
    parser.add_argument("--out", required=True, help="输出 JSON 报告路径。")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top-k（默认 5）。")
    parser.add_argument("--qa-top-k", type=int, default=3, help="QA 检索 top-k（默认 3）。")
    parser.add_argument("--model-id", default=None, help="可选 LLM model id 覆盖。")
    parser.add_argument("--load-in-4bit", action="store_true", help="启用 4-bit 模型加载。")
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 N 条问题。")
    parser.add_argument("--skip-qa", action="store_true", help="跳过 QA 评测，仅做检索对比。")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    questions = _load_questions(args.questions)
    if not questions:
        print("[ERROR] No valid questions found.")
        return 1
    if args.limit is not None and args.limit > 0:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions.")

    # --- load indexes ---
    print("Loading text index...")
    text_index = TextIndex.load(args.text_index_dir)
    print("Loading hybrid index...")
    hybrid_index = HybridPageIndex.load(args.hybrid_index_dir)

    visual_index = None
    if args.visual_index_dir:
        print("Loading visual index...")
        visual_index = VisualPageIndex.load(args.visual_index_dir)

    # --- retrieval comparison ---
    print("\nRunning retrieval comparison...")
    retrieval_result = run_retrieval_comparison(
        questions=questions,
        text_index=text_index,
        hybrid_index=hybrid_index,
        visual_index=visual_index,
        top_k=args.top_k,
    )
    _print_retrieval_table(retrieval_result["summary"])

    report: Dict[str, Any] = {
        "questions_file": args.questions,
        "num_questions": len(questions),
        "retrieval": retrieval_result["summary"],
        "retrieval_details": retrieval_result["details"],
    }

    if not args.skip_qa:
        # --- QA comparison ---
        print("\nInitializing text QA engine...")
        text_engine = TextDocQAEngine(
            index_dir=args.text_index_dir,
            model_id=args.model_id,
            top_k=args.qa_top_k,
            load_in_4bit=args.load_in_4bit,
        )
        print("Initializing multimodal QA engine...")
        multimodal_engine = DocQAEngine(
            index_dir=args.hybrid_index_dir,
            model_id=args.model_id,
            top_k=args.qa_top_k,
            load_in_4bit=args.load_in_4bit,
            retriever_type=args.multimodal_type,
            visual_index_dir=args.visual_index_dir,
            text_index_dir=args.text_index_dir if args.multimodal_type == "fusion" else None,
        )

        print("Running QA comparison...")
        qa_result = run_qa_comparison(
            questions=questions,
            text_engine=text_engine,
            multimodal_engine=multimodal_engine,
        )
        _print_qa_table(qa_result["summary"])

        report["qa"] = qa_result["summary"]
        report["qa_predictions"] = qa_result["predictions"]

        # --- delta summary ---
        text_qa = qa_result["summary"]["text"]
        mm_qa = qa_result["summary"]["multimodal"]
        delta = {
            "em_delta": round(mm_qa["em"] - text_qa["em"], 4),
            "f1_delta": round(mm_qa["f1"] - text_qa["f1"], 4),
            "anls_delta": round(mm_qa["anls"] - text_qa["anls"], 4),
            "recall@3_delta": round(mm_qa["recall@3"] - text_qa["recall@3"], 4),
            "citation_accuracy_delta": round(mm_qa["citation_accuracy"] - text_qa["citation_accuracy"], 4),
        }
        report["delta"] = delta

        print("\n" + "=" * 80)
        print("多模态 vs 纯文本 差值 (正值表示多模态更优)")
        print("=" * 80)
        for key, val in delta.items():
            print(f"  {key}: {val:+.4f}")

    # --- save report ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Comparison report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
