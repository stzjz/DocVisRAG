import re
import unicodedata
from collections import Counter
from typing import List


def _drop_punct(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def normalize_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = _drop_punct(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if " " in normalized:
        return [t for t in normalized.split(" ") if t]
    return list(normalized)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, replace))
        prev = curr
    return prev[-1]


def simple_anls(pred: str, gold: str) -> float:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    dist = _levenshtein(p, g)
    denom = max(len(p), len(g))
    if denom == 0:
        return 1.0

    score = 1.0 - (dist / denom)
    return score if score >= 0.5 else 0.0


def recall_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int) -> float:
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if not gold_pages:
        return 0.0

    topk = set(int(x) for x in retrieved_pages[:k])
    gold = set(int(x) for x in gold_pages)
    return 1.0 if topk & gold else 0.0


def mrr(retrieved_pages: List[int], gold_pages: List[int]) -> float:
    gold = set(int(x) for x in gold_pages)
    if not gold:
        return 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if int(page) in gold:
            return 1.0 / rank
    return 0.0
