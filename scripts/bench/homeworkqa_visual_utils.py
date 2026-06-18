"""Utilities for HomeworkQA visual retrieval tuning."""

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PageKey = Tuple[str, int]


def page_key(row: Mapping[str, Any]) -> PageKey:
    return str(row.get("doc_id", "")).strip(), int(row.get("page_index", -1))


def _char_terms(text: str) -> set[str]:
    chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", str(text).lower())
    merged = "".join(chunks)
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


def lexical_score(query: str, page_text: str) -> float:
    q_terms = _char_terms(query)
    if not q_terms:
        return 0.0
    p_terms = _char_terms(page_text)
    if not p_terms:
        return 0.0
    score = len(q_terms & p_terms) / max(1, len(q_terms))
    q_nums = set(re.findall(r"\d+(?:\.\d+)?", str(query)))
    if q_nums:
        p_nums = set(re.findall(r"\d+(?:\.\d+)?", str(page_text)))
        score += 0.5 * (len(q_nums & p_nums) / max(1, len(q_nums)))
    return float(score)


def _norm_scores(rows: Sequence[Dict[str, Any]]) -> Dict[PageKey, float]:
    vals = [float(r.get("score", 0.0)) for r in rows]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    out: Dict[PageKey, float] = {}
    for row in rows:
        raw = float(row.get("score", 0.0))
        out[page_key(row)] = 1.0 if hi <= lo else (raw - lo) / (hi - lo)
    return out


def metadata_rerank_visual_pages(
    rows: Sequence[Dict[str, Any]],
    query: str,
    hybrid_meta_by_page: Mapping[PageKey, Mapping[str, Any]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """Rerank visual candidates with page OCR/summary metadata.

    alpha is the visual-score weight. 1.0 preserves pure visual ranking;
    0.0 means metadata-only reranking inside the visual candidate set.
    """

    alpha = max(0.0, min(1.0, float(alpha)))
    visual_norm = _norm_scores(rows)
    candidate_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        key = page_key(row)
        hrow = hybrid_meta_by_page.get(key, {})
        page_text = "\n".join(
            [
                str(hrow.get("summary", "")),
                str(hrow.get("ocr_text_preview", "")),
                str(hrow.get("search_text", "")),
            ]
        )
        out = dict(row)
        for field in ["image_path", "summary", "ocr_text_preview", "doc_id"]:
            if not out.get(field) and hrow.get(field):
                out[field] = hrow.get(field)
        out["visual_rank"] = rank
        out["visual_norm"] = visual_norm.get(key, 0.0)
        out["metadata_score"] = lexical_score(query, page_text)
        out["visual_metadata_rerank_alpha"] = alpha
        candidate_rows.append(out)
    return sorted(
        candidate_rows,
        key=lambda x: (
            -(
                alpha * float(x.get("visual_norm", 0.0))
                + (1.0 - alpha) * float(x.get("metadata_score", 0.0))
            ),
            int(x.get("visual_rank", 999999)),
        ),
    )
