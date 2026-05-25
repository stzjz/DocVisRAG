import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.docvisrag.retrieve import TextIndex
from src.docvisrag.vlm import QwenVLClient


@dataclass
class TextQAResult:
    question: str
    answer: str
    evidence: List[dict] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    uncertainty: Optional[str] = None


class TextDocQAEngine:
    def __init__(
        self,
        index_dir: str,
        model_id: str | None = None,
        top_k: int = 5,
        load_in_4bit: bool = False,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        self.index_dir = index_dir
        self.top_k = top_k
        self.max_new_tokens = 512

        self.text_index = TextIndex.load(index_dir)
        self.llm = QwenVLClient(
            model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            load_in_4bit=load_in_4bit,
        )

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        if not text:
            return ""
        markers = [f"{section_name}：", f"{section_name}:"]
        start = -1
        marker_len = 0
        for marker in markers:
            pos = text.find(marker)
            if pos >= 0:
                start = pos
                marker_len = len(marker)
                break
        if start < 0:
            return ""

        tail = text[start + marker_len:]
        next_keys = [
            "\n答案：", "\n依据：",
            "\n引用：", "\n不确定性：",
            "\n答案:", "\n依据:",
            "\n引用:", "\n不确定性:",
        ]
        cut = len(tail)
        for key in next_keys:
            pos = tail.find(key)
            if pos >= 0:
                cut = min(cut, pos)
        return tail[:cut].strip()

    @staticmethod
    def _parse_citations(citation_text: str) -> List[str]:
        if not citation_text:
            return []
        matches = re.findall(r"第\s*\d+\s*页", citation_text)
        if not matches:
            return []
        seen = set()
        ordered: List[str] = []
        for c in matches:
            norm = re.sub(r"\s+", " ", c).strip()
            if norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        lines = [
            "你是文档问答助手。",
            "你只能依据给定的 OCR 文本片段回答。",
            "如果证据不足，回答“文档中未找到明确依据”。",
            "必须输出：",
            "答案：",
            "依据：",
            "引用：",
            "不确定性：",
            "引用必须写成“第 X 页”。",
            "",
            f"用户问题：{question.strip()}",
            "",
            "候选文本片段：",
        ]
        for i, chunk in enumerate(chunks, start=1):
            page = chunk.get("page_index", -1)
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            lines.extend(
                [
                    f"[片段 {i}] 第 {page} 页 (分数: {score:.4f})",
                    f"内容：{text}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        return self.text_index.search(question, top_k=self.top_k)

    def answer(self, question: str) -> TextQAResult:
        if not question or not question.strip():
            raise ValueError("question must be non-empty.")

        chunks = self._retrieve(question)
        if not chunks:
            return TextQAResult(
                question=question,
                answer="文档中未找到明确依据",
                evidence=[],
                citations=[],
                uncertainty="检索阶段未找到相关文本块",
            )

        prompt = self._build_prompt(question, chunks)
        raw = self.llm.answer_text(question=prompt, max_new_tokens=self.max_new_tokens)

        answer_text = self._extract_section(raw, "答案") or raw.strip()
        evidence_text = self._extract_section(raw, "依据")
        citation_text = self._extract_section(raw, "引用")
        uncertainty_text = self._extract_section(raw, "不确定性")
        citations = self._parse_citations(citation_text)

        if not citations:
            pages = sorted({int(c.get("page_index", -1)) for c in chunks if c.get("page_index", -1) > 0})
            citations = [f"第 {p} 页" for p in pages]
            citations = list(dict.fromkeys(citations))

        return TextQAResult(
            question=question,
            answer=answer_text,
            evidence=[
                {
                    "page_index": int(c.get("page_index", -1)),
                    "text": c.get("text", ""),
                    "score": float(c.get("score", 0.0)),
                    "model_evidence": evidence_text,
                }
                for c in chunks
            ],
            citations=citations,
            uncertainty=uncertainty_text or "未说明",
        )
