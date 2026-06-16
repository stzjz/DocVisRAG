"""Run short-answer HomeworkQA QA with combined indexes and per-PDF filtering."""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from homeworkqa_visual_utils import metadata_rerank_visual_pages
from src.docvisrag.eval import citation_accuracy, exact_match, mrr, recall_at_k, relaxed_accuracy, simple_anls, token_f1
from src.docvisrag.retrieve import (
    HybridPageIndex,
    TextIndex,
    VisualPageIndex,
    text_chunks_to_page_results,
    weighted_reciprocal_rank_fusion,
)
from src.docvisrag.vlm import QwenVLClient


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
            if isinstance(row, dict):
                row.setdefault("type", "text")
                row.setdefault("evidence_pages", [])
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
    scopes: Dict[str, set[PageKey]] = {}
    for item in cfg.get("benchmarks", []):
        if not isinstance(item, dict):
            continue
        manifest = str(item.get("manifest", "")).strip()
        if not manifest:
            continue
        name = str(item.get("name", "")).strip()
        doc_name = _strip_homeworkqa_prefix(name) if name else Path(manifest).parent.name
        pages = _load_json(manifest)
        allowed: set[PageKey] = set()
        for page in pages if isinstance(pages, list) else []:
            if isinstance(page, dict):
                allowed.add((str(page.get("doc_id", "")).strip(), int(page.get("page_index", -1))))
        if allowed:
            scopes[doc_name] = allowed
    return scopes


def _resolve_image_path(image_path: str, index_dir: str) -> str:
    img = Path(image_path)
    if img.is_absolute() and img.exists():
        return str(img)
    candidate_cwd = (Path.cwd() / img).resolve()
    if candidate_cwd.exists():
        return str(candidate_cwd)
    candidate_from_index = (Path(index_dir).expanduser().resolve().parent / img).resolve()
    if candidate_from_index.exists():
        return str(candidate_from_index)
    raise FileNotFoundError(f"Image path not found: {image_path}")


def _extract_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    for marker in ["答案：", "答案:", "Answer:", "answer:"]:
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            return tail.splitlines()[0].strip()
    return text.splitlines()[0].strip()


def _extract_pages_from_text(text: str) -> List[int]:
    pages: List[int] = []
    for num in re.findall(r"\d+", str(text or "")):
        pages.append(int(num))
    out: List[int] = []
    seen = set()
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _build_qa_prompt(sample: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "你是一个严格的文档问答助手。",
        "只能依据给定 PDF 页面图像、页面摘要和 OCR 文本回答。",
        "如果证据不足，回答“文档中未找到明确依据”。",
        "请给出简短答案。事实题直接输出日期、数字、名称或短语；需要解释时也尽量一句话回答。",
        "必须按下面格式输出：",
        "答案：<简短答案>",
        "依据：<一句话说明>",
        "引用：第 X 页",
        "",
        f"问题：{str(sample.get('question', '')).strip()}",
        "",
        "候选证据：",
    ]
    for i, row in enumerate(evidence, start=1):
        page_idx = int(row.get("page_index", -1))
        lines.append(f"[证据 {i}] 第 {page_idx} 页")
        lines.append(f"页面摘要：{row.get('summary', '')}")
        lines.append(f"OCR文本：{row.get('ocr_text_preview', '')}")
        for j, match in enumerate(row.get("text_matches", []) or [], start=1):
            if isinstance(match, dict) and match.get("text"):
                lines.append(f"OCR命中片段 {j}：{match.get('text')}")
        lines.append("")
    return "\n".join(lines).strip()


def _summarize(rows: Sequence[Dict[str, Any]], group_field: str | None = None) -> Dict[str, Any]:
    if group_field is None:
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
            "avg_latency_seconds": _avg([float(r.get("latency_seconds", 0.0)) for r in rows]),
        }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)
    return {k: _summarize(v) for k, v in sorted(grouped.items())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HomeworkQA short-answer QA with combined-index per-PDF filtering.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--batch-config", required=True)
    parser.add_argument("--text-index-dir", required=True)
    parser.add_argument("--hybrid-index-dir", required=True)
    parser.add_argument("--visual-index-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--mode", default="fusion", choices=["text", "hybrid", "visual", "fusion"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--fusion-text-weight", type=float, default=0.2)
    parser.add_argument("--fusion-hybrid-weight", type=float, default=5.0)
    parser.add_argument("--fusion-visual-weight", type=float, default=5.0)
    parser.add_argument("--fusion-text-candidates", type=int, default=0)
    parser.add_argument("--fusion-hybrid-candidates", type=int, default=0)
    parser.add_argument("--fusion-visual-candidates", type=int, default=0)
    parser.add_argument("--visual-rerank-mode", default="metadata", choices=["none", "metadata"])
    parser.add_argument("--visual-rerank-alpha", type=float, default=0.1, help="Visual score weight for metadata rerank; 0 means metadata-only.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    questions = _load_jsonl(args.questions)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("[ERROR] No questions found.")
        return 1

    try:
        scopes = _load_doc_scopes(args.batch_config)
        text_index = TextIndex.load(args.text_index_dir) if args.mode in {"text", "fusion"} else None
        hybrid_index = HybridPageIndex.load(args.hybrid_index_dir) if args.mode in {"hybrid", "fusion", "text", "visual"} else None
        visual_index = VisualPageIndex.load(args.visual_index_dir) if args.mode in {"visual", "fusion"} else None
        vlm = QwenVLClient(model_id=args.model_id or "Qwen/Qwen3-VL-4B-Instruct", load_in_4bit=args.load_in_4bit)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Init HomeworkQA QA failed: {exc}")
        return 1

    text_total = len(text_index.metadata) if text_index is not None else 0
    hybrid_total = len(hybrid_index.metadata) if hybrid_index is not None else 0
    visual_total = len(visual_index.metadata) if visual_index is not None else 0
    text_candidate_k = _candidate_depth(args.fusion_text_candidates, text_total, args.top_k * 20)
    hybrid_candidate_k = _candidate_depth(args.fusion_hybrid_candidates, hybrid_total, args.top_k * 10)
    visual_candidate_k = _candidate_depth(args.fusion_visual_candidates, visual_total, args.top_k * 10)
    hybrid_meta_by_page = {_page_key(r): r for r in hybrid_index.metadata} if hybrid_index is not None else {}

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        iterator = tqdm(questions, desc=f"HomeworkQA {args.mode} QA", unit="q", dynamic_ncols=True, disable=args.no_progress)
        for sample in iterator:
            question = str(sample.get("question", "")).strip()
            doc_name = _question_doc_name(sample)
            allowed_pages = scopes.get(doc_name)
            if not question or not allowed_pages:
                continue

            text_pages: List[Dict[str, Any]] = []
            hybrid_pages: List[Dict[str, Any]] = []
            visual_pages: List[Dict[str, Any]] = []
            try:
                if text_index is not None:
                    text_chunks = _filter_rows(text_index.search(question, top_k=text_candidate_k), allowed_pages)
                    text_pages = _dedup_pages(text_chunks_to_page_results(text_chunks, top_k=None))
                    for page in text_pages:
                        hrow = hybrid_meta_by_page.get(_page_key(page))
                        if hrow:
                            for field in ["image_path", "summary", "ocr_text_preview"]:
                                if not page.get(field) and hrow.get(field):
                                    page[field] = hrow.get(field)
                if hybrid_index is not None:
                    hybrid_pages = _dedup_pages(_filter_rows(hybrid_index.search(question, top_k=hybrid_candidate_k), allowed_pages))
                if visual_index is not None:
                    visual_pages = _dedup_pages(_filter_rows(visual_index.search(question, top_k=visual_candidate_k), allowed_pages))
                    for page in visual_pages:
                        hrow = hybrid_meta_by_page.get(_page_key(page))
                        if hrow:
                            for field in ["summary", "ocr_text_preview", "doc_id"]:
                                if not page.get(field) and hrow.get(field):
                                    page[field] = hrow.get(field)
                    if args.visual_rerank_mode == "metadata":
                        visual_pages = metadata_rerank_visual_pages(
                            visual_pages,
                            query=question,
                            hybrid_meta_by_page=hybrid_meta_by_page,
                            alpha=args.visual_rerank_alpha,
                        )
                if args.mode == "text":
                    evidence = text_pages[: args.top_k]
                elif args.mode == "hybrid":
                    evidence = hybrid_pages[: args.top_k]
                elif args.mode == "visual":
                    evidence = visual_pages[: args.top_k]
                else:
                    evidence = weighted_reciprocal_rank_fusion(
                        {"text": text_pages, "hybrid": hybrid_pages, "visual": visual_pages},
                        weights={
                            "text": args.fusion_text_weight,
                            "hybrid": args.fusion_hybrid_weight,
                            "visual": args.fusion_visual_weight,
                        },
                        top_k=args.top_k,
                    )
            except Exception as exc:  # noqa: BLE001
                evidence = []
                retrieval_error = str(exc)
            else:
                retrieval_error = None

            image_paths: List[str] = []
            resolved_evidence: List[Dict[str, Any]] = []
            started = time.perf_counter()
            try:
                for row in evidence:
                    out_row = dict(row)
                    out_row["image_path"] = _resolve_image_path(str(row.get("image_path", "")), args.hybrid_index_dir)
                    resolved_evidence.append(out_row)
                    image_paths.append(out_row["image_path"])
                raw = vlm.answer_images(
                    image_paths=image_paths,
                    question=_build_qa_prompt(sample, resolved_evidence),
                    max_new_tokens=args.max_new_tokens,
                )
                pred_answer = _extract_answer(raw)
                error = retrieval_error
            except Exception as exc:  # noqa: BLE001
                raw = ""
                pred_answer = ""
                error = str(exc)
            latency = round(time.perf_counter() - started, 3)

            gold_answer = str(sample.get("answer", "")).strip()
            gold_pages = [int(x) for x in sample.get("evidence_pages", [])]
            retrieved_pages = [int(x.get("page_index", -1)) for x in evidence]
            citation_pages = _extract_pages_from_text(raw)
            result = {
                "id": sample.get("id", ""),
                "doc_id": doc_name,
                "type": sample.get("type", ""),
                "mode": args.mode,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "raw_pred": raw,
                "gold_pages": gold_pages,
                "retrieved_pages": retrieved_pages,
                "em": exact_match(pred_answer, gold_answer),
                "f1": token_f1(pred_answer, gold_answer),
                "anls": simple_anls(pred_answer, gold_answer),
                "relaxed_accuracy": relaxed_accuracy(pred_answer, gold_answer),
                "retrieval_recall@1": recall_at_k(retrieved_pages, gold_pages, 1) if gold_pages else 0.0,
                "retrieval_recall@3": recall_at_k(retrieved_pages, gold_pages, 3) if gold_pages else 0.0,
                "retrieval_mrr": mrr(retrieved_pages, gold_pages) if gold_pages else 0.0,
                "citation_accuracy": citation_accuracy(citation_pages, gold_pages) if gold_pages else 0.0,
                "latency_seconds": latency,
                "error": error,
            }
            result_rows.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "benchmark": "homeworkqa",
        "task": "short_answer_qa",
        "mode": args.mode,
        "config": {
            "questions": str(Path(args.questions)),
            "top_k": args.top_k,
            "qa_model_id": args.model_id or "Qwen/Qwen3-VL-4B-Instruct",
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
        "summary": _summarize(result_rows),
        "by_doc": _summarize(result_rows, "doc_id"),
        "by_type": _summarize(result_rows, "type"),
        "predictions": str(out_path),
    }
    summary_path = Path(args.summary_out).expanduser().resolve() if args.summary_out else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_path}")
    print(f"[OK] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
