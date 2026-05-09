from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.docvisrag.retrieve import HybridPageIndex
from src.docvisrag.vlm import QwenVLClient


@dataclass
class QAResult:
    question: str
    answer: str
    evidence: List[dict]
    citations: List[str]
    uncertainty: Optional[str]


class DocQAEngine:
    def __init__(
        self,
        index_dir: str,
        model_id: str | None = None,
        top_k: int = 3,
        load_in_4bit: bool = False,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        self.index_dir = index_dir
        self.top_k = top_k
        self.max_new_tokens = 512
        self.index = HybridPageIndex.load(index_dir)
        self.vlm = QwenVLClient(
            model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            load_in_4bit=load_in_4bit,
        )

    def _resolve_image_path(self, image_path: str) -> str:
        img = Path(image_path)
        if img.is_absolute() and img.exists():
            return str(img)

        candidate_cwd = (Path.cwd() / img).resolve()
        if candidate_cwd.exists():
            return str(candidate_cwd)

        candidate_from_index = (Path(self.index_dir).expanduser().resolve().parent / img).resolve()
        if candidate_from_index.exists():
            return str(candidate_from_index)

        raise FileNotFoundError(f"Image path from index metadata not found: {image_path}")

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        marker = f"{section_name}："
        start = text.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        tail = text[start:]
        for next_marker in ["\n依据：", "\n引用：", "\n不确定性："]:
            pos = tail.find(next_marker)
            if pos >= 0:
                return tail[:pos].strip()
        return tail.strip()

    @staticmethod
    def _parse_citations(citation_text: str) -> List[str]:
        cites = []
        for part in citation_text.replace("，", ",").split(","):
            c = part.strip()
            if not c:
                continue
            if not c.startswith("第"):
                continue
            if "页" not in c:
                continue
            cites.append(c)
        return sorted(set(cites))

    def _build_prompt(self, question: str, evidence: List[Dict]) -> str:
        lines = [
            "你只能依据给定页面图像、页面摘要和 OCR 文本回答。",
            "如果证据不足，回答“文档中未找到明确依据”。",
            "必须输出以下格式：",
            "答案：",
            "依据：",
            "引用：",
            "不确定性：",
            "引用必须写成“第 X 页”。",
            "",
            f"用户问题：{question.strip()}",
            "",
            "候选页面证据如下：",
        ]
        for i, row in enumerate(evidence, start=1):
            lines.extend(
                [
                    f"[候选页 {i}] 第 {row['page_index']} 页",
                    f"页面摘要：{row.get('summary', '')}",
                    f"OCR文本：{row.get('ocr_text_preview', '')}",
                    f"检索分数：{row.get('score', 0.0):.4f}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def answer(self, question: str) -> QAResult:
        if not question or not question.strip():
            raise ValueError("question must be non-empty.")

        retrieved = self.index.search(question, top_k=self.top_k)
        if not retrieved:
            return QAResult(
                question=question,
                answer="文档中未找到明确依据",
                evidence=[],
                citations=[],
                uncertainty="未检索到相关页面",
            )

        evidence: List[Dict] = []
        image_paths: List[str] = []
        for row in retrieved:
            resolved = self._resolve_image_path(str(row.get("image_path", "")))
            evidence_row = dict(row)
            evidence_row["image_path"] = resolved
            evidence.append(evidence_row)
            image_paths.append(resolved)

        prompt = self._build_prompt(question, evidence)
        raw = self.vlm.answer_images(
            image_paths=image_paths,
            question=prompt,
            max_new_tokens=self.max_new_tokens,
        )

        answer_text = self._extract_section(raw, "答案") or raw.strip()
        evidence_text = self._extract_section(raw, "依据")
        citation_text = self._extract_section(raw, "引用")
        uncertainty_text = self._extract_section(raw, "不确定性")
        citations = self._parse_citations(citation_text)

        if not citations:
            fallback = [f"第 {int(x['page_index'])} 页" for x in evidence]
            citations = sorted(set(fallback))

        return QAResult(
            question=question,
            answer=answer_text,
            evidence=[
                {
                    "page_index": int(x.get("page_index", -1)),
                    "image_path": x.get("image_path", ""),
                    "summary": x.get("summary", ""),
                    "ocr_text_preview": x.get("ocr_text_preview", ""),
                    "score": float(x.get("score", 0.0)),
                    "model_evidence": evidence_text,
                }
                for x in evidence
            ],
            citations=citations,
            uncertainty=uncertainty_text or "无",
        )
