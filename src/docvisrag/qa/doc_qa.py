import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.docvisrag.retrieve import (
    HybridPageIndex,
    VisualPageIndex,
    reciprocal_rank_fusion,
)
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
        retriever_type: str = "hybrid",
        visual_index_dir: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        retriever_type = (retriever_type or "hybrid").strip().lower()
        if retriever_type not in {"hybrid", "visual", "fusion"}:
            raise ValueError(
                f"retriever_type must be one of hybrid/visual/fusion, got: {retriever_type}"
            )
        self.index_dir = index_dir
        self.top_k = top_k
        self.max_new_tokens = 512
        self.retriever_type = retriever_type
        self.visual_index_dir = visual_index_dir
        self.hybrid_index: Optional[HybridPageIndex] = None
        self.visual_index: Optional[VisualPageIndex] = None

        if retriever_type == "hybrid":
            self.hybrid_index = HybridPageIndex.load(index_dir)
        elif retriever_type == "visual":
            self.hybrid_index = self._try_load_hybrid(index_dir)
            vdir = self._resolve_visual_index_dir(index_dir=index_dir, visual_index_dir=visual_index_dir)
            self.visual_index = VisualPageIndex.load(vdir)
        else:
            self.hybrid_index = HybridPageIndex.load(index_dir)
            vdir = self._resolve_visual_index_dir(index_dir=index_dir, visual_index_dir=visual_index_dir)
            self.visual_index = VisualPageIndex.load(vdir)

        self.vlm = QwenVLClient(
            model_id=model_id or "Qwen/Qwen2.5-VL-3B-Instruct",
            load_in_4bit=load_in_4bit,
        )

    @staticmethod
    def _try_load_hybrid(index_dir: str) -> Optional[HybridPageIndex]:
        try:
            return HybridPageIndex.load(index_dir)
        except Exception:
            return None

    @staticmethod
    def _resolve_visual_index_dir(index_dir: str, visual_index_dir: str | None) -> str:
        if visual_index_dir:
            path = Path(visual_index_dir).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"visual_index_dir not found: {path}")
            return str(path)

        base = Path(index_dir).expanduser().resolve()
        candidates = [
            base.parent / "visual_index",
            base.parent / "hybrid_index" / ".." / "visual_index",
        ]
        for cand in candidates:
            resolved = cand.resolve()
            if resolved.exists():
                return str(resolved)
        raise FileNotFoundError(
            "visual index directory not found. "
            "Please pass visual_index_dir explicitly when retriever_type is visual/fusion."
        )

    @staticmethod
    def _result_key(row: Dict) -> Tuple[str, int]:
        return (Path(str(row.get("image_path", ""))).name, int(row.get("page_index", -1)))

    @staticmethod
    def _enrich_visual_results(
        visual_results: List[Dict],
        hybrid_results: List[Dict],
    ) -> List[Dict]:
        by_key = {DocQAEngine._result_key(x): x for x in hybrid_results}
        enriched: List[Dict] = []
        for row in visual_results:
            out = dict(row)
            key = DocQAEngine._result_key(out)
            hrow = by_key.get(key)
            if hrow:
                out.setdefault("summary", hrow.get("summary", ""))
                out.setdefault("ocr_text_preview", hrow.get("ocr_text_preview", ""))
                out.setdefault("doc_id", hrow.get("doc_id", out.get("doc_id", "")))
            else:
                out.setdefault("summary", "")
                out.setdefault("ocr_text_preview", "")
            enriched.append(out)
        return enriched

    def _retrieve(self, question: str) -> List[Dict]:
        if self.retriever_type == "hybrid":
            assert self.hybrid_index is not None
            return self.hybrid_index.search(question, top_k=self.top_k)

        if self.retriever_type == "visual":
            assert self.visual_index is not None
            visual = self.visual_index.search(question, top_k=self.top_k)
            hybrid_for_enrich: List[Dict] = []
            if self.hybrid_index is not None:
                try:
                    hybrid_for_enrich = self.hybrid_index.search(question, top_k=max(self.top_k * 2, 10))
                except Exception:
                    hybrid_for_enrich = []
            return self._enrich_visual_results(visual, hybrid_for_enrich)

        assert self.hybrid_index is not None and self.visual_index is not None
        hybrid = self.hybrid_index.search(question, top_k=max(self.top_k * 2, 10))
        visual = self.visual_index.search(question, top_k=max(self.top_k * 2, 10))
        visual = self._enrich_visual_results(visual, hybrid)
        return reciprocal_rank_fusion(hybrid_results=hybrid, visual_results=visual, top_k=self.top_k)

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

        tail = text[start + marker_len :]
        next_keys = ["\n答案：", "\n依据：", "\n引用：", "\n不确定性：", "\n答案:", "\n依据:", "\n引用:", "\n不确定性:"]
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
        # keep order and deduplicate
        seen = set()
        ordered: List[str] = []
        for c in matches:
            norm = re.sub(r"\s+", " ", c).strip()
            if norm not in seen:
                seen.add(norm)
                ordered.append(norm)
        return ordered

    def _build_prompt(self, question: str, evidence: List[Dict]) -> str:
        lines = [
            "你是文档问答助手。",
            "你只能依据给定页面图像、页面摘要和 OCR 文本回答。",
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
            "候选证据：",
        ]
        for i, row in enumerate(evidence, start=1):
            lines.extend(
                [
                    f"[证据 {i}] 第 {row['page_index']} 页",
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

        retrieved = self._retrieve(question)
        if not retrieved:
            return QAResult(
                question=question,
                answer="文档中未找到明确依据",
                evidence=[],
                citations=[],
                uncertainty="检索阶段未找到候选页面",
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
            citations = [f"第 {int(x['page_index'])} 页" for x in evidence]
            citations = list(dict.fromkeys(citations))

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
            uncertainty=uncertainty_text or "未说明",
        )
