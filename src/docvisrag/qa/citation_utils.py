"""共享引用解析工具。

提供统一的引用格式解析，支持：
- 页码引用："第 X 页"
- 图号引用："图 X" / "图X" / "Figure X"
- 表号引用："表 X" / "表X" / "Table X"

同时也提供提示词模板中的引用格式说明，以及从版面分析结果
构建图/表上下文的工具函数。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── 引用解析 ──────────────────────────────────────────────

def parse_citations(text: str) -> List[str]:
    """从文本中提取所有引用（页码+图号+表号），保持顺序并去重。

    识别格式：
      - 第 X 页 / 第X页
      - 图 X / 图X / Figure X / FigureX
      - 表 X / 表X / Table X / TableX
    """
    if not text:
        return []

    patterns = [
        (r"第\s*\d+\s*页", _normalize_page),
        (r"(?:图|Figure)\s*\d+", _normalize_figure),
        (r"(?:表|Table)\s*\d+", _normalize_table),
    ]

    all_matches: List[Tuple[int, str]] = []
    for pattern, normalizer in patterns:
        for m in re.finditer(pattern, text):
            norm = normalizer(m.group())
            all_matches.append((m.start(), norm))

    all_matches.sort(key=lambda x: x[0])

    seen: set = set()
    ordered: List[str] = []
    for _, norm in all_matches:
        if norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered


def parse_citations_with_types(text: str) -> List[Dict[str, str]]:
    """解析引用并返回带类型的结构化列表。

    返回: [{"type": "page"|"figure"|"table", "label": "第 3 页", "number": 3}, ...]
    """
    if not text:
        return []

    patterns = [
        (r"第\s*(\d+)\s*页", "page", "第 {n} 页"),
        (r"(?:图|Figure)\s*(\d+)", "figure", "图 {n}"),
        (r"(?:表|Table)\s*(\d+)", "table", "表 {n}"),
    ]

    all_matches: List[Tuple[int, Dict[str, Any]]] = []
    for pattern, ctype, label_template in patterns:
        for m in re.finditer(pattern, text):
            number = int(m.group(1))
            all_matches.append(
                (
                    m.start(),
                    {
                        "type": ctype,
                        "label": label_template.format(n=number),
                        "number": number,
                    },
                )
            )

    all_matches.sort(key=lambda x: x[0])

    seen: set = set()
    ordered: List[Dict[str, str]] = []
    for _, item in all_matches:
        key = (item["type"], item["number"])
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _normalize_page(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def _normalize_figure(raw: str) -> str:
    num = re.search(r"\d+", raw)
    n = num.group() if num else "?"
    return f"图 {n}"


def _normalize_table(raw: str) -> str:
    num = re.search(r"\d+", raw)
    n = num.group() if num else "?"
    return f"表 {n}"


# ── 提示词模板 ────────────────────────────────────────────

CITATION_INSTRUCTION = (
    "引用必须包含具体的来源定位。至少使用“第 X 页”格式；"
    "当证据涉及图表或表格时，必须同时使用“图 Y”或“表 Z”格式。"
    "示例：“第 3 页，图 1”或“第 5 页，表 2”。"
)


def make_citation_instruction() -> str:
    """返回标准化的引用格式说明，用于 prompt 中。"""
    return CITATION_INSTRUCTION


# ── 图/表上下文构建 ───────────────────────────────────────

def load_layout_context(
    layout_jsonl: Optional[str],
) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    """从 layout JSONL 加载图/表编号索引。

    返回: {page_index: {"figures": [(label, text), ...], "tables": [(label, text), ...]}}
    """
    if not layout_jsonl:
        return {}

    path = Path(layout_jsonl)
    if not path.exists():
        return {}

    regions = _load_jsonl(str(path))

    # 按页面分组，为 figure 和 table 分配序号
    pages: Dict[int, List[Dict]] = {}
    for r in regions:
        pi = int(r.get("page_index", -1))
        pages.setdefault(pi, []).append(r)

    context: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
    for page_index, page_regions in pages.items():
        # 按 bbox y0 排序（从上到下），保证编号顺序合理
        page_regions.sort(key=lambda r: float(r.get("bbox", [0, 0, 0, 0])[1]))
        fig_num = 0
        tbl_num = 0
        figures: List[Tuple[str, str]] = []
        tables: List[Tuple[str, str]] = []
        for r in page_regions:
            rtype = str(r.get("region_type", ""))
            text = str(r.get("text", ""))[:200]
            if rtype == "figure":
                fig_num += 1
                figures.append((f"图 {fig_num}", text))
            elif rtype == "table":
                tbl_num += 1
                tables.append((f"表 {tbl_num}", text))
        if figures or tables:
            context[page_index] = {"figures": figures, "tables": tables}
    return context


def build_figure_table_context_lines(
    page_index: int,
    context: Dict[int, Dict[str, List[Tuple[str, str]]]],
) -> List[str]:
    """为指定页面构建图/表上下文的提示词行。"""
    lines: List[str] = []
    page_ctx = context.get(page_index)
    if not page_ctx:
        return lines

    figures = page_ctx.get("figures", [])
    tables = page_ctx.get("tables", [])

    if figures:
        labels = ", ".join(label for label, _ in figures)
        lines.append(f"本页图：{labels}")
    if tables:
        labels = ", ".join(label for label, _ in tables)
        lines.append(f"本页表：{labels}")
    return lines


def build_evidence_figure_table_summary(
    evidence: List[Dict],
    context: Dict[int, Dict[str, List[Tuple[str, str]]]],
) -> Dict[int, Dict[str, Any]]:
    """为 evidence 列表中的每一页构建图/表摘要。

    返回: {page_index: {"figure_labels": [...], "table_labels": [...]}}
    """
    result: Dict[int, Dict[str, Any]] = {}
    for row in evidence:
        pi = int(row.get("page_index", -1))
        if pi in result:
            continue
        page_ctx = context.get(pi, {})
        result[pi] = {
            "figure_labels": [label for label, _ in page_ctx.get("figures", [])],
            "table_labels": [label for label, _ in page_ctx.get("tables", [])],
        }
    return result


# ── 内部工具 ──────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items
