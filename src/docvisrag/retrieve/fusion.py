from typing import Dict, List, Mapping, Sequence, Tuple


def _make_key(item: Dict) -> Tuple[str, int]:
    doc_id = str(item.get("doc_id", "")).strip()
    page_index = int(item.get("page_index", -1))
    if doc_id:
        return doc_id, page_index

    image_path = str(item.get("image_path", "")).strip()
    return image_path, page_index


def text_chunks_to_page_results(
    text_chunks: Sequence[Dict],
    top_k: int | None = None,
    max_snippets_per_page: int = 3,
) -> List[Dict]:
    """Aggregate OCR chunk hits into page-level rows.

    Text retrieval is chunk-level, while hybrid/visual retrieval is page-level.
    This keeps the strongest text signal per page and stores OCR snippets so
    multimodal QA can see the exact answer-bearing context.
    """

    pages: Dict[Tuple[str, int], Dict] = {}
    first_seen_rank: Dict[Tuple[str, int], int] = {}

    for rank, chunk in enumerate(text_chunks, start=1):
        page_index = int(chunk.get("page_index", -1))
        if page_index <= 0:
            continue

        key = _make_key(chunk)
        if key not in pages:
            pages[key] = {
                "doc_id": str(chunk.get("doc_id", "")),
                "page_index": page_index,
                "score": float(chunk.get("score", 0.0)),
                "text_score": float(chunk.get("score", 0.0)),
                "text_rank": rank,
                "text_matches": [],
            }
            first_seen_rank[key] = rank

        page = pages[key]
        score = float(chunk.get("score", 0.0))
        if score > float(page.get("text_score", 0.0)):
            page["text_score"] = score
            page["score"] = score
            page["text_rank"] = rank

        matches = page.setdefault("text_matches", [])
        text = str(chunk.get("text", "")).strip()
        if text and len(matches) < max_snippets_per_page:
            matches.append(
                {
                    "text": text,
                    "score": score,
                    "bbox": chunk.get("bbox", []),
                    "confidence": float(chunk.get("confidence", 0.0)),
                    "rank": rank,
                }
            )

    ranked = sorted(
        pages.values(),
        key=lambda row: (-float(row.get("text_score", 0.0)), first_seen_rank[_make_key(row)]),
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def weighted_reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Dict]],
    weights: Mapping[str, float] | None = None,
    top_k: int = 3,
    k: int = 60,
) -> List[Dict]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    weights = weights or {}
    merged: Dict[Tuple[str, int], Dict] = {}
    rrf_scores: Dict[Tuple[str, int], float] = {}
    source_scores: Dict[Tuple[str, int], Dict[str, float]] = {}
    source_ranks: Dict[Tuple[str, int], Dict[str, int]] = {}

    for source, rows in ranked_lists.items():
        weight = float(weights.get(source, 1.0))
        if weight <= 0:
            continue

        for rank, row in enumerate(rows, start=1):
            key = _make_key(row)
            if key[1] <= 0:
                continue

            if key not in merged:
                merged[key] = dict(row)
            else:
                keep = merged[key]
                for field in [
                    "doc_id",
                    "image_path",
                    "local_index_image_path",
                    "summary",
                    "ocr_text_preview",
                ]:
                    if not keep.get(field) and row.get(field):
                        keep[field] = row.get(field)

                if row.get("text_matches"):
                    existing = keep.setdefault("text_matches", [])
                    seen = {str(x.get("text", "")) for x in existing if isinstance(x, dict)}
                    for match in row.get("text_matches", []):
                        if not isinstance(match, dict):
                            continue
                        text = str(match.get("text", ""))
                        if text and text not in seen:
                            existing.append(match)
                            seen.add(text)

            rrf_scores[key] = rrf_scores.get(key, 0.0) + (weight / (k + rank))
            source_scores.setdefault(key, {})[source] = float(row.get("score", 0.0))
            source_ranks.setdefault(key, {})[source] = rank

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict] = []
    for key, score in ranked[:top_k]:
        row = dict(merged[key])
        row["score"] = float(score)
        row["fusion_score"] = float(score)
        row["source_scores"] = source_scores.get(key, {})
        row["source_ranks"] = source_ranks.get(key, {})
        out.append(row)
    return out


def reciprocal_rank_fusion(
    hybrid_results: List[Dict],
    visual_results: List[Dict],
    top_k: int = 3,
    k: int = 60,
) -> List[Dict]:
    return weighted_reciprocal_rank_fusion(
        ranked_lists={"hybrid": hybrid_results, "visual": visual_results},
        weights={"hybrid": 1.0, "visual": 1.0},
        top_k=top_k,
        k=k,
    )
