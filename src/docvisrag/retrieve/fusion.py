from typing import Dict, List, Tuple


def _make_key(item: Dict) -> Tuple[str, int]:
    image_path = str(item.get("image_path", "")).strip()
    page_index = int(item.get("page_index", -1))
    return image_path, page_index


def reciprocal_rank_fusion(
    hybrid_results: List[Dict],
    visual_results: List[Dict],
    top_k: int = 3,
    k: int = 60,
) -> List[Dict]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    merged: Dict[Tuple[str, int], Dict] = {}
    rrf_scores: Dict[Tuple[str, int], float] = {}

    for rank, row in enumerate(hybrid_results, start=1):
        key = _make_key(row)
        merged[key] = dict(row)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))

    for rank, row in enumerate(visual_results, start=1):
        key = _make_key(row)
        if key not in merged:
            merged[key] = dict(row)
        else:
            keep = merged[key]
            if not keep.get("summary") and row.get("summary"):
                keep["summary"] = row.get("summary")
            if not keep.get("ocr_text_preview") and row.get("ocr_text_preview"):
                keep["ocr_text_preview"] = row.get("ocr_text_preview")
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict] = []
    for key, score in ranked[:top_k]:
        row = dict(merged[key])
        row["score"] = float(score)
        row["fusion_score"] = float(score)
        out.append(row)
    return out
