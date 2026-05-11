import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


CATEGORIES = [
    "OCR 识别错误",
    "检索未召回正确页面",
    "检索排序错误",
    "图表/表格读取错误",
    "生成模型幻觉",
    "引用页码错误",
    "标准答案或标注不清楚",
]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
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
            rows.append(obj)
    return rows


def _guess_category(row: Dict[str, Any]) -> str:
    qtype = str(row.get("type", "text"))
    em = float(row.get("em", 0.0))
    f1 = float(row.get("f1", 0.0))
    anls = float(row.get("anls", 0.0))
    error_msg = str(row.get("error", "") or "")

    gold_pages = [int(x) for x in row.get("gold_pages", [])]
    evidence_pages = [int(x) for x in row.get("pred_evidence_pages", [])]
    citation_pages = [int(x) for x in row.get("pred_citation_pages", [])]

    if em >= 1.0:
        return ""

    if not str(row.get("gold_answer", "")).strip():
        return "标准答案或标注不清楚"
    if error_msg:
        return "生成模型幻觉"

    if gold_pages and not any(p in set(gold_pages) for p in evidence_pages):
        return "检索未召回正确页面"

    if gold_pages and evidence_pages and evidence_pages[0] not in set(gold_pages):
        return "检索排序错误"

    if citation_pages and gold_pages and not any(p in set(gold_pages) for p in citation_pages):
        return "引用页码错误"

    if qtype in {"chart", "table"} and f1 < 0.5:
        return "图表/表格读取错误"

    if anls < 0.3:
        return "生成模型幻觉"

    if f1 < 0.4:
        return "OCR 识别错误"

    return "标准答案或标注不清楚"


def _avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate markdown error analysis from predictions jsonl.")
    parser.add_argument("--predictions", required=True, help="Path to predictions jsonl")
    parser.add_argument("--out", default="data/outputs/error_analysis.md", help="Output markdown path")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        rows = _load_jsonl(args.predictions)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load predictions: {exc}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bucket: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CATEGORIES}
    ems, f1s, anls_list = [], [], []

    for row in rows:
        ems.append(float(row.get("em", 0.0)))
        f1s.append(float(row.get("f1", 0.0)))
        anls_list.append(float(row.get("anls", 0.0)))

        category = _guess_category(row)
        if category:
            bucket[category].append(row)

    lines: List[str] = []
    lines.append("# DocVisRAG Error Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Samples: {len(rows)}")
    lines.append(f"- EM: {_avg(ems):.4f}")
    lines.append(f"- F1: {_avg(f1s):.4f}")
    lines.append(f"- ANLS: {_avg(anls_list):.4f}")
    lines.append("")

    lines.append("## Category Summary")
    for c in CATEGORIES:
        lines.append(f"- {c}: {len(bucket[c])}")
    lines.append("")

    lines.append("## Detailed Error Cases")
    lines.append("")

    for c in CATEGORIES:
        lines.append(f"### {c}")
        lines.append("")
        lines.append("| id | type | question | gold_answer | pred_answer | note |")
        lines.append("|---|---|---|---|---|---|")

        if bucket[c]:
            for row in bucket[c][:20]:
                qid = str(row.get("id", ""))
                qtype = str(row.get("type", ""))
                question = str(row.get("question", "")).replace("|", " ")
                gold = str(row.get("gold_answer", "")).replace("|", " ")
                pred = str(row.get("pred_answer", "")).replace("|", " ")
                note = (
                    f"gold_pages={row.get('gold_pages', [])}; "
                    f"evidence_pages={row.get('pred_evidence_pages', [])}; "
                    f"citations={row.get('pred_citation_pages', [])}"
                )
                lines.append(f"| {qid} | {qtype} | {question} | {gold} | {pred} | {note} |")
        else:
            lines.append("| - | - | - | - | - | (无样本，请人工补充) |")

        lines.append("")
        lines.append("人工复核建议：")
        lines.append("- 复查原始页面图像、OCR 文本、摘要和预测引用页。")
        lines.append("- 判断是标注问题、检索问题还是生成问题。")
        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    print("[OK] Error analysis markdown generated.")
    print(f"- output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
