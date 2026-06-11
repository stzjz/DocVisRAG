import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.docvisrag.qa.citation_utils import (
    build_figure_table_context_lines,
    load_layout_context,
    make_citation_instruction,
    parse_citations,
)
from src.docvisrag.retrieve import (
    HybridPageIndex,
    TextIndex,
    VisualPageIndex,
    reciprocal_rank_fusion,
    text_chunks_to_page_results,
    weighted_reciprocal_rank_fusion,
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
        layout_jsonl: str | None = None,
        text_index_dir: str | None = None,
        fusion_text_weight: float = 2.0,
        fusion_hybrid_weight: float = 0.3,
        fusion_visual_weight: float = 0.1,
        fusion_text_candidates: int | None = None,
        fusion_hybrid_candidates: int | None = None,
        fusion_visual_candidates: int | None = None,
        manifest_path: str | None = None,
        summary_jsonl: str | None = None,
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
        self.text_index_dir = text_index_dir
        self.fusion_weights = {
            "text": float(fusion_text_weight),
            "hybrid": float(fusion_hybrid_weight),
            "visual": float(fusion_visual_weight),
        }
        self.fusion_text_candidates = fusion_text_candidates
        self.fusion_hybrid_candidates = fusion_hybrid_candidates
        self.fusion_visual_candidates = fusion_visual_candidates
        self.hybrid_index: Optional[HybridPageIndex] = None
        self.visual_index: Optional[VisualPageIndex] = None
        self.text_index: Optional[TextIndex] = None
        self._use_text_index = False
        if text_index_dir:
            try:
                self.text_index = TextIndex.load(text_index_dir)
                self._use_text_index = True
            except Exception:
                pass
        self._manifest_map = {}
        if manifest_path:
            self._load_manifest(manifest_path)
        self._page_summaries = {}
        if summary_jsonl:
            self._load_summaries(summary_jsonl)

        if retriever_type == "hybrid":
            if self._use_text_index:
                self.hybrid_index = self._try_load_hybrid(index_dir)
            else:
                self.hybrid_index = HybridPageIndex.load(index_dir)
        elif retriever_type == "visual":
            self.hybrid_index = self._try_load_hybrid(index_dir)
            vdir = self._resolve_visual_index_dir(index_dir=index_dir, visual_index_dir=visual_index_dir)
            self.visual_index = VisualPageIndex.load(vdir)
        else:
            if self._use_text_index:
                self.hybrid_index = self._try_load_hybrid(index_dir)
            else:
                self.hybrid_index = HybridPageIndex.load(index_dir)
            vdir = self._resolve_visual_index_dir(index_dir=index_dir, visual_index_dir=visual_index_dir)
            self.visual_index = VisualPageIndex.load(vdir)
            if text_index_dir:
                self.text_index = TextIndex.load(text_index_dir)

        self.vlm = QwenVLClient(
            model_id=model_id or "Qwen/Qwen2.5-VL-7B-Instruct",
            load_in_4bit=load_in_4bit,
        )

        # 加载版面分析数据用于图/表引用
        self._layout_context = load_layout_context(layout_jsonl)

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


    def _load_manifest(self, manifest_path):
        import json
        mf = Path(manifest_path).expanduser().resolve()
        if not mf.exists(): return
        try: data = json.loads(mf.read_text(encoding="utf-8"))
        except: return
        if not isinstance(data, list): return
        for p in data:
            if not isinstance(p, dict): continue
            self._manifest_map[(str(p.get("doc_id","")), int(p.get("page_index",-1)))] = str(p.get("image_path",""))

    @staticmethod
    def _load_jsonl(path):
        import json
        rows = []
        with Path(path).open("r",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: rows.append(json.loads(line))
                except: continue
        return rows

    def _load_summaries(self, path):
        for row in self._load_jsonl(path):
            key = (str(row.get("doc_id","")), int(row.get("page_index",-1)))
            s = str(row.get("summary","")).strip()
            if s: self._page_summaries[key] = s

    @staticmethod
    def _key(doc_id, page_index):
        return (str(doc_id), int(page_index))

    def _retrieve_text_chunks_and_aggregate(self, question):
        assert self.text_index is not None
        chunks = self.text_index.search(question, top_k=max(self.top_k * 4, 20))
        page_best = {}
        for c in chunks:
            key = self._key(str(c.get("doc_id","")), int(c.get("page_index",-1)))
            score = float(c.get("score",0.0))
            if key not in page_best or score > page_best[key][0]:
                page_best[key] = (score, [c])
        ranked = sorted(page_best.items(), key=lambda kv: kv[1][0], reverse=True)
        results = []
        for (doc_id, pi), (score, c_list) in ranked[:self.top_k]:
            ocr = " ".join(str(c.get("text","")) for c in c_list if str(c.get("text","")).strip())
            summary = self._page_summaries.get((doc_id, pi), "")
            img = self._manifest_map.get((doc_id, pi), "")
            results.append({"doc_id":doc_id,"page_index":pi,"score":score,"image_path":img,"summary":summary,"ocr_text_preview":ocr[:500]})
        return results

    @staticmethod
    def _result_key(row: Dict) -> Tuple[str, int]:
        return (Path(str(row.get("image_path", ""))).name, int(row.get("page_index", -1)))

    @staticmethod
    def _page_key(row: Dict) -> Tuple[str, int]:
        return (str(row.get("doc_id", "")), int(row.get("page_index", -1)))

    def _enrich_text_page_results(self, text_pages: List[Dict]) -> List[Dict]:
        if self.hybrid_index is None:
            return text_pages

        by_page = {self._page_key(row): row for row in self.hybrid_index.metadata}
        enriched: List[Dict] = []
        for row in text_pages:
            out = dict(row)
            hrow = by_page.get(self._page_key(out))
            if hrow:
                for field in ["image_path", "summary", "ocr_text_preview", "doc_id"]:
                    if not out.get(field) and hrow.get(field):
                        out[field] = hrow.get(field)
            enriched.append(out)
        return enriched

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
        if self.retriever_type == "hybrid" and self._use_text_index:
            return self._retrieve_text_chunks_and_aggregate(question)
        if self.retriever_type == "hybrid":
            if self.hybrid_index is not None:
                return self.hybrid_index.search(question, top_k=self.top_k)
            if self._use_text_index:
                return self._retrieve_text_chunks_and_aggregate(question)
            raise RuntimeError("Hybrid retriever is unavailable.")

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

        assert self.visual_index is not None
        default_candidate_k = max(self.top_k * 4, 20)
        hybrid_candidate_k = self.fusion_hybrid_candidates or default_candidate_k
        visual_candidate_k = self.fusion_visual_candidates or default_candidate_k
        page_candidate_k = max(hybrid_candidate_k, visual_candidate_k)
        text_candidate_k = self.fusion_text_candidates or max(page_candidate_k * 3, 50)

        visual = self.visual_index.search(question, top_k=visual_candidate_k)
        if self.hybrid_index is None:
            if not (self._use_text_index and self.text_index is not None):
                raise RuntimeError("Fusion retriever requires a hybrid or text index.")
            text_pages = self._retrieve_text_chunks_and_aggregate(question)
            visual = self._enrich_visual_results(visual, text_pages)
            return reciprocal_rank_fusion(hybrid_results=text_pages, visual_results=visual, top_k=self.top_k)

        hybrid = self.hybrid_index.search(question, top_k=hybrid_candidate_k)
        visual = self._enrich_visual_results(visual, hybrid)

        if self.text_index is None:
            return weighted_reciprocal_rank_fusion(
                ranked_lists={"hybrid": hybrid, "visual": visual},
                weights={"hybrid": self.fusion_weights["hybrid"], "visual": self.fusion_weights["visual"]},
                top_k=self.top_k,
            )

        text_chunks = self.text_index.search(question, top_k=text_candidate_k)
        text_pages = text_chunks_to_page_results(
            text_chunks,
            top_k=page_candidate_k,
            max_snippets_per_page=3,
        )
        text_pages = self._enrich_text_page_results(text_pages)
        return weighted_reciprocal_rank_fusion(
            ranked_lists={"text": text_pages, "hybrid": hybrid, "visual": visual},
            weights=self.fusion_weights,
            top_k=self.top_k,
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
    def _extract_short_answer(answer_text):
        text = (answer_text or "").strip()
        if not text: return ""
        for marker in ["Answer:", "answer:", "答案：", "答案:"]:
            pos = text.find(marker)
            if pos >= 0:
                after = text[pos + len(marker):]
                line = after.split("\n")[0].strip()
                if line: text = line; break
        first_line = text.split("\n")[0].strip()
        import re
        colon_match = re.match(r"^([A-Za-z\u4e00-\u9fff\s]+):\s*(.+)", first_line)
        if colon_match:
            prefix, suffix = colon_match.group(1).strip(), colon_match.group(2).strip()
            if len(prefix.split()) <= 5 and not any(c.isdigit() for c in prefix):
                first_line = suffix
        return first_line.strip()

    @staticmethod
    def _parse_citations(citation_text: str) -> List[str]:
        """从 VLM 输出中提取所有引用（页码+图号+表号）。

        复用 citation_utils.parse_citations 统一解析。
        """
        return parse_citations(citation_text)

    @staticmethod
    def _is_english(text: str) -> bool:
        """Detect if the question is primarily in English."""
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        return ascii_chars > len(text) * 0.3

    def _build_prompt(self, question: str, evidence: List[Dict]) -> str:
        eng = self._is_english(question)

        if eng:
            lines = [
                "You are a document QA assistant.",
                "Answer questions based ONLY on the provided page images, summaries and OCR text.",
                "If evidence is insufficient, answer: \"Not enough evidence in the document.\"",
                "",
                "IMPORTANT: Provide a SHORT, CONCISE answer (1-5 words for factual questions).",
                "If the question asks for a number, color, name, date, or single fact, just output that value.",
                "Do NOT output long explanations unless the question explicitly asks for one.",
                "",
                "You MUST output in this format:",
                "Answer: <your short answer>",
                "Evidence: <which page(s) support this>",
                "Citation: Page X",
                "Uncertainty: <low|medium|high>",
                make_citation_instruction(),
                "",
                f"Question: {question.strip()}",
                "",
                "Evidence pages:",
            ]
            for i, row in enumerate(evidence, start=1):
                page_idx = int(row.get("page_index", -1))
                lines.append(f"[Page {page_idx}] Score: {row.get('score', 0.0):.3f}")
                lines.append(f"Summary: {row.get('summary', '')}")
                lines.append(f"OCR: {row.get('ocr_text_preview', '')}")
                text_matches = row.get("text_matches", []) or []
                for j, match in enumerate(text_matches, start=1):
                    if isinstance(match, dict) and match.get("text"):
                        lines.append(f"Matched OCR {j}: {match.get('text')}")

                ft_lines = build_figure_table_context_lines(
                    page_index=page_idx,
                    context=self._layout_context,
                )
                for ft_line in ft_lines:
                    lines.append(ft_line)
                lines.append("")
        else:
            lines = [
                "你是文档问答助手。",
                "你只能依据给定页面图像、页面摘要和 OCR 文本回答。",
                "如果证据不足，回答'文档中未找到明确依据'。",
                "",
                "注意：请给出简短精炼的答案。如果问题是事实性问题（数字、名称、日期等），直接输出答案，不要长篇解释。",
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
                "候选证据：",
            ]
            for i, row in enumerate(evidence, start=1):
                page_idx = int(row.get("page_index", -1))
                lines.append(f"[证据 {i}] 第 {page_idx} 页")
                lines.append(f"页面摘要：{row.get('summary', '')}")
                lines.append(f"OCR文本：{row.get('ocr_text_preview', '')}")
                text_matches = row.get("text_matches", []) or []
                for j, match in enumerate(text_matches, start=1):
                    if isinstance(match, dict) and match.get("text"):
                        lines.append(f"OCR命中片段 {j}：{match.get('text')}")

                ft_lines = build_figure_table_context_lines(
                    page_index=page_idx,
                    context=self._layout_context,
                )
                for ft_line in ft_lines:
                    lines.append(ft_line)

                lines.append(f"检索分数：{row.get('score', 0.0):.4f}")
                lines.append("")
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
        short_answer = self._extract_short_answer(answer_text)
        evidence_text = self._extract_section(raw, "依据")
        citation_text = self._extract_section(raw, "引用")
        uncertainty_text = self._extract_section(raw, "不确定性")
        citations = self._parse_citations(citation_text)

        if not citations:
            # 回退：从 evidence 生成页码引用
            page_cites = [f"第 {int(x['page_index'])} 页" for x in evidence]
            page_cites = list(dict.fromkeys(page_cites))
            # 附加版面分析中的图/表标签
            ft_cites: List[str] = []
            for x in evidence:
                pi = int(x.get("page_index", -1))
                ctx = self._layout_context.get(pi, {})
                for label, _ in ctx.get("figures", []):
                    ft_cites.append(label)
                for label, _ in ctx.get("tables", []):
                    ft_cites.append(label)
            citations = page_cites + list(dict.fromkeys(ft_cites))

        return QAResult(
            question=question,
            answer=short_answer,
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
