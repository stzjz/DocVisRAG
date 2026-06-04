import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.docvisrag.qa.citation_utils import (
    build_figure_table_context_lines,
    load_layout_context,
    make_citation_instruction,
    parse_citations,
)
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
        layout_jsonl: str | None = None,
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

        # 加载版面分析数据用于图/表引用
        self._layout_context = load_layout_context(layout_jsonl)

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        if not text:
            return ""

        aliases = {
            "答案": ["答案", "Answer", "answer"],
            "依据": ["依据", "Evidence", "evidence"],
            "引用": ["引用", "Citation", "citation"],
            "不确定性": ["不确定性", "Uncertainty", "uncertainty"],
        }
        names = aliases.get(section_name, [section_name])
        markers = []
        for name in names:
            markers.extend([f"{name}：", f"{name}:", f"{name} "])

        start = -1
        marker_len = 0
        for marker in markers:
            pos = text.find(marker)
            if pos >= 0 and (start < 0 or pos < start):
                start = pos
                marker_len = len(marker)
        if start < 0:
            return ""

        tail = text[start + marker_len:]
        next_keys = [
            "\n答案：", "\n答案:", "\nAnswer:", "\nanswer:",
            "\n依据：", "\n依据:", "\nEvidence:", "\nevidence:",
            "\n引用：", "\n引用:", "\nCitation:", "\ncitation:",
            "\n不确定性：", "\n不确定性:", "\nUncertainty:", "\nuncertainty:",
        ]
        cut = len(tail)
        for key in next_keys:
            pos = tail.find(key)
            if pos >= 0:
                cut = min(cut, pos)
        return tail[:cut].strip()

    @staticmethod
    def _parse_citations(citation_text: str) -> List[str]:
        """从 LLM 输出中提取所有引用（页码+图号+表号）。

        复用 citation_utils.parse_citations 统一解析。
        """
        return parse_citations(citation_text)

    @staticmethod
    def _is_english(text: str) -> bool:
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        return ascii_chars > len(text) * 0.3

    def _build_prompt(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        eng = self._is_english(question)

        if eng:
            lines = [
                "You are a document QA assistant.",
                "Answer questions based ONLY on the provided OCR text fragments.",
                "If evidence is insufficient, answer: \"Not enough evidence.\"",
                "",
                "IMPORTANT: Provide a SHORT, CONCISE answer (1-5 words for factual questions).",
                "If the question asks for a number, color, name, date, or single fact, just output that value.",
                "",
                "You MUST output in this format:",
                "Answer: <your short answer>",
                "Evidence: <source>",
                "Citation: Page X",
                "Uncertainty: <low|medium|high>",
                make_citation_instruction(),
                "",
                f"Question: {question.strip()}",
                "",
                "Relevant text fragments:",
            ]
            for i, chunk in enumerate(chunks, start=1):
                page = chunk.get("page_index", -1)
                text = chunk.get("text", "")
                score = chunk.get("score", 0.0)
                lines.append(f"[Fragment {i}] Page {page} (score: {score:.4f})")
                lines.append(f"Content: {text}")
                ft_lines = build_figure_table_context_lines(
                    page_index=int(page) if page is not None else -1,
                    context=self._layout_context,
                )
                for ft_line in ft_lines:
                    lines.append(ft_line)
                lines.append("")
        else:
            lines = [
                "你是文档问答助手。",
                "你只能依据给定的 OCR 文本片段回答。",
                "如果证据不足，回答'文档中未找到明确依据'。",
                "",
                "注意：请给出简短精炼的答案。事实性问题直接输出答案即可。",
                "",
                "必须输出：",
                "答案：<简短答案>",
                "依据：<证据来源>",
                "引用：第 X 页",
                "不确定性：<低|中|高>",
                make_citation_instruction(),
                "",
                f"用户问题：{question.strip()}",
                "",
                "候选文本片段：",
            ]
            for i, chunk in enumerate(chunks, start=1):
                page = chunk.get("page_index", -1)
                text = chunk.get("text", "")
                score = chunk.get("score", 0.0)
                lines.append(f"[片段 {i}] 第 {page} 页 (分数: {score:.4f})")
                lines.append(f"内容：{text}")
                ft_lines = build_figure_table_context_lines(
                    page_index=int(page) if page is not None else -1,
                    context=self._layout_context,
                )
                for ft_line in ft_lines:
                    lines.append(ft_line)
                lines.append("")
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
            page_cites = [f"第 {p} 页" for p in pages]
            # 附加版面分析中的图/表标签
            ft_cites: List[str] = []
            for c in chunks:
                pi = int(c.get("page_index", -1))
                ctx = self._layout_context.get(pi, {})
                for label, _ in ctx.get("figures", []):
                    ft_cites.append(label)
                for label, _ in ctx.get("tables", []):
                    ft_cites.append(label)
            citations = page_cites + list(dict.fromkeys(ft_cites))

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
