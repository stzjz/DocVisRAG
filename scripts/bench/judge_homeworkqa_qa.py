"""Judge HomeworkQA short-answer predictions with a local text LLM."""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from tqdm.auto import tqdm


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
                rows.append(row)
    return rows


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _sample_key(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("question") or "").strip()


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _sample_key(row)
            if key:
                keys.add(key)
    return keys


def _build_prompt(row: Dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    gold = str(row.get("gold_answer", "")).strip()
    pred = str(row.get("pred_answer", "")).strip()
    raw = str(row.get("raw_pred", "")).strip()
    if raw and raw != pred:
        pred_block = f"{pred}\n\n模型完整输出：\n{raw}"
    else:
        pred_block = pred
    return "\n".join(
        [
            "你是 HomeworkQA 简答题的严格但合理的判分员。",
            "请只根据题目、标准答案和模型答案判断语义是否正确，不要引入外部知识。",
            "判分规则：",
            "1. 模型答案与标准答案语义等价，或包含标准答案所需的关键字段，判为 correct=true。",
            "2. 数字、单位、名称、选项、步骤或条件有关键错误，判为 correct=false。",
            "3. 模型答案只回答了部分关键内容，且不足以完整回答问题，判为 correct=false。",
            "4. 模型答案说找不到依据、答非所问、空答案或与标准答案冲突，判为 correct=false。",
            "5. 允许中英文同义表达、格式差异、轻微冗余。",
            "",
            "请输出严格 JSON，不要输出 Markdown，不要输出额外解释。格式如下：",
            '{"correct": true, "score": 1.0, "reason": "一句话中文理由"}',
            "",
            f"题目：{question}",
            f"标准答案：{gold}",
            f"模型答案：{pred_block}",
        ]
    )


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    candidates = [raw]
    matches = re.findall(r"\{.*?\}", raw, flags=re.S)
    candidates.extend(matches)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    lowered = raw.lower()
    if re.search(r"\btrue\b|正确|可以算对|算对", lowered):
        return {"correct": True, "score": 1.0, "reason": raw[:200]}
    if re.search(r"\bfalse\b|错误|不正确|不能算对|不算对", lowered):
        return {"correct": False, "score": 0.0, "reason": raw[:200]}
    return {"correct": False, "score": 0.0, "reason": f"无法解析 judge 输出：{raw[:200]}"}


class TextJudge:
    def __init__(self, model: str, load_in_4bit: bool = False) -> None:
        self.model_ref = str(Path(model).expanduser()) if Path(model).expanduser().exists() else model
        self.load_in_4bit = load_in_4bit
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "local_files_only": True,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_ref, trust_remote_code=True, **kwargs)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def judge(self, prompt: str, max_new_tokens: int) -> str:
        self.load()
        import torch

        messages = [
            {"role": "system", "content": "你是严谨的中文作业问答评测员，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_questions": len(rows),
        "judge_accuracy": _avg([1.0 if r.get("judge_correct") else 0.0 for r in rows]),
        "judge_score": _avg([float(r.get("judge_score", 0.0)) for r in rows]),
        "em": _avg([float(r.get("em", 0.0)) for r in rows]),
        "f1": _avg([float(r.get("f1", 0.0)) for r in rows]),
        "anls": _avg([float(r.get("anls", 0.0)) for r in rows]),
        "relaxed_accuracy": _avg([float(r.get("relaxed_accuracy", 0.0)) for r in rows]),
        "retrieval_recall@1": _avg([float(r.get("retrieval_recall@1", 0.0)) for r in rows]),
        "retrieval_recall@3": _avg([float(r.get("retrieval_recall@3", 0.0)) for r in rows]),
        "citation_accuracy": _avg([float(r.get("citation_accuracy", 0.0)) for r in rows]),
    }


def _summarize_by(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field, "")), []).append(row)
    return {k: _summarize(v) for k, v in sorted(groups.items())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge HomeworkQA short-answer predictions with a local text LLM.")
    parser.add_argument("--predictions", required=True, help="Input predictions JSONL from eval_homeworkqa_qa.py")
    parser.add_argument("--out", required=True, help="Output judged JSONL")
    parser.add_argument("--summary-out", default=None, help="Output summary JSON")
    parser.add_argument("--judge-model", required=True, help="Local model path or HF id cached locally")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in --out")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    predictions = _load_jsonl(args.predictions)
    if args.limit and args.limit > 0:
        predictions = predictions[: args.limit]
    if not predictions:
        print("[ERROR] No predictions found.")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_keys(out_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    judge = TextJudge(args.judge_model, load_in_4bit=args.load_in_4bit)
    judged_rows: List[Dict[str, Any]] = []
    if args.resume and out_path.exists():
        judged_rows.extend(_load_jsonl(out_path))

    todo = [row for row in predictions if _sample_key(row) not in existing]
    with out_path.open(mode, encoding="utf-8") as f:
        iterator = tqdm(todo, desc="HomeworkQA LLM judge", unit="q", dynamic_ncols=True, disable=args.no_progress)
        for row in iterator:
            prompt = _build_prompt(row)
            try:
                judge_raw = judge.judge(prompt, max_new_tokens=args.max_new_tokens)
                parsed = _extract_json(judge_raw)
                score = float(parsed.get("score", 1.0 if parsed.get("correct") else 0.0))
                score = max(0.0, min(1.0, score))
                judged = dict(row)
                judged.update(
                    {
                        "judge_model": args.judge_model,
                        "judge_correct": bool(parsed.get("correct")),
                        "judge_score": score,
                        "judge_reason": str(parsed.get("reason", "")).strip(),
                        "judge_raw": judge_raw,
                        "judge_error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                judged = dict(row)
                judged.update(
                    {
                        "judge_model": args.judge_model,
                        "judge_correct": False,
                        "judge_score": 0.0,
                        "judge_reason": "",
                        "judge_raw": "",
                        "judge_error": str(exc),
                    }
                )
            judged_rows.append(judged)
            f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "benchmark": "homeworkqa",
        "task": "short_answer_llm_judge",
        "judge_model": args.judge_model,
        "predictions": str(Path(args.predictions).expanduser().resolve()),
        "judgements": str(out_path),
        "summary": _summarize(judged_rows),
        "by_mode": _summarize_by(judged_rows, "mode"),
        "by_doc": _summarize_by(judged_rows, "doc_id"),
        "by_type": _summarize_by(judged_rows, "type"),
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
