"""Generate a harder HomeworkQA split from verified source questions.

The hard split keeps the same PDF scope as HomeworkQA but asks multi-evidence
questions. It does not overwrite the base benchmark.
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PAGE_HINT_RE = re.compile(r"第[一二三四五六七八九十0-9]+页|第一页|第二页|第三页|第四页|第五页|第六页")


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


def _pages(row: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for x in row.get("evidence_pages", []) or []:
        try:
            p = int(x)
        except Exception:
            continue
        if p not in out:
            out.append(p)
    return out


def _union_pages(rows: Sequence[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for row in rows:
        for p in _pages(row):
            if p not in pages:
                pages.append(p)
    return sorted(pages)


def _clean_question(q: str) -> str:
    text = str(q).strip().rstrip("？?")
    text = PAGE_HINT_RE.sub("对应位置", text)
    return text


def _is_good(row: Dict[str, Any]) -> bool:
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    return bool(q and a and _pages(row))


def _pair_score(a: Dict[str, Any], b: Dict[str, Any]) -> tuple[int, int, int]:
    pages_a, pages_b = set(_pages(a)), set(_pages(b))
    diff_page = 1 if pages_a != pages_b else 0
    diff_type = 1 if str(a.get("type", "")) != str(b.get("type", "")) else 0
    page_span = max(_union_pages([a, b])) - min(_union_pages([a, b])) if _union_pages([a, b]) else 0
    return diff_page, diff_type, page_span


def _make_multi_hop(doc_rows: List[Dict[str, Any]], doc_id: str, n: int, seed: int) -> List[Dict[str, Any]]:
    rows = [r for r in doc_rows if _is_good(r)]
    rng = _stable_rng(seed, doc_id)
    pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            pairs.append((a, b))
    rng.shuffle(pairs)
    pairs.sort(key=lambda x: _pair_score(x[0], x[1]), reverse=True)

    out: List[Dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    source_use: Dict[str, int] = defaultdict(int)

    def add_pair(a: Dict[str, Any], b: Dict[str, Any], max_source_use: int) -> bool:
        if len(out) >= n:
            return False
        aid, bid = str(a.get("id", "")), str(b.get("id", ""))
        key = tuple(sorted([aid, bid]))
        if key in used_pairs:
            return False
        if source_use[aid] >= max_source_use or source_use[bid] >= max_source_use:
            return False
        used_pairs.add(key)
        source_use[aid] += 1
        source_use[bid] += 1
        idx = len(out) + 1
        qa = _clean_question(str(a.get("question", "")))
        qb = _clean_question(str(b.get("question", "")))
        question = f"请基于整份《{doc_id}》PDF 同时回答两个相关问题：第一，{qa}？第二，{qb}？"
        answer = f"第一：{str(a.get('answer', '')).strip()}；第二：{str(b.get('answer', '')).strip()}"
        source_modalities = [str(a.get("modality", "text")), str(b.get("modality", "text"))]
        modality = "multi_evidence_multimodal" if any(m != "text" for m in source_modalities) else "text"
        out.append(
            {
                "id": f"homeworkqa_hard_{doc_id}_{idx:03d}",
                "benchmark": "homeworkqa_hard",
                "doc_id": doc_id,
                "doc_path": a.get("doc_path", ""),
                "source_pdf": a.get("source_pdf", ""),
                "question": question,
                "answer": answer,
                "evidence_pages": _union_pages([a, b]),
                "type": "multi_hop",
                "difficulty": "hard",
                "modality": modality,
                "source_modalities": source_modalities,
                "input_scope": "full_pdf",
                "source_question_ids": [aid, bid],
                "question_generation": "deterministic_balanced_multi_evidence_from_verified_homeworkqa",
            }
        )
        return True

    for max_source_use in range(1, 5):
        for a, b in pairs:
            add_pair(a, b, max_source_use)
            if len(out) >= n:
                return out
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HomeworkQA-Hard multi-evidence questions.")
    parser.add_argument("--questions", default="data/bench_full/homeworkqa/questions.jsonl")
    parser.add_argument("--batch-config", default="data/bench_full/homeworkqa/batch_config.json")
    parser.add_argument("--out-dir", default="data/bench_full/homeworkqa_hard")
    parser.add_argument("--questions-per-pdf", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260616)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_jsonl(args.questions)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            by_doc[doc_id].append(row)

    out_rows: List[Dict[str, Any]] = []
    out_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id, doc_rows in sorted(by_doc.items()):
        hard_rows = _make_multi_hop(doc_rows, doc_id, max(1, args.questions_per_pdf), args.seed)
        out_by_doc[doc_id] = hard_rows
        out_rows.extend(hard_rows)

    out_dir = Path(args.out_dir).expanduser().resolve()
    _write_jsonl(out_dir / "questions.jsonl", out_rows)
    for doc_id, doc_rows in out_by_doc.items():
        _write_jsonl(out_dir / "questions_by_pdf" / f"{doc_id}.jsonl", doc_rows)

    batch_src = Path(args.batch_config).expanduser().resolve()
    if batch_src.exists():
        batch = _load_json(batch_src)
        for item in batch.get("benchmarks", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            doc_name = name.removeprefix("homeworkqa_")
            item["name"] = name.replace("homeworkqa_", "homeworkqa_hard_", 1)
            item["benchmark"] = "homeworkqa_hard"
            item["questions"] = str(out_dir / "questions_by_pdf" / f"{doc_name}.jsonl")
            item["dataset_id"] = "local_homeworkqa_hard"
        (out_dir / "batch_config.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "benchmark": "homeworkqa_hard",
        "source_questions": str(Path(args.questions)),
        "num_questions": len(out_rows),
        "questions_per_pdf": args.questions_per_pdf,
        "seed": args.seed,
        "difficulty_policy": "multi-evidence questions generated from pairs of verified HomeworkQA questions, preferring different pages and types",
        "by_doc": {doc: len(items) for doc, items in sorted(out_by_doc.items())},
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
