import datetime as dt
import json
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from PIL import Image, ImageDraw

from src.docvisrag.ingest import (
    build_page_summaries,
    ingest_document,
    resolve_ocr_backend,
    run_ocr_on_manifest,
    save_manifest,
)
from src.docvisrag.qa import DocQAEngine
from src.docvisrag.retrieve import HybridPageIndex, VisualPageIndex


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes"
MAX_DEMO_PAGES = 10


@dataclass
class BuildOptions:
    dpi: int
    load_in_4bit: bool
    max_pages: int
    ocr_backend: str
    visual_mode: str


def _new_session_id() -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"session_{ts}_{short}"


def _friendly_model_error(err_text: str) -> str:
    lower = (err_text or "").lower()
    if "failed to load vlm model" in lower or "failed to load qwen" in lower:
        tips = [
            "模型加载失败，常见原因：",
            "1) 模型未下载且当前网络不可用；",
            "2) 显存不足；",
            "3) transformers 版本不支持该模型；",
            "4) model id 配置错误。",
        ]
        if "local_files_only" in lower or "local disk and outgoing traffic has been disabled" in lower:
            tips.append("5) 当前处于本地缓存/离线模式，但缓存里还没有对应模型。")
        return "\n".join(tips + ["", f"原始错误：{err_text}"])
    return err_text


def _copy_uploaded_file(uploaded_path: str, dst_dir: Path) -> Path:
    src = Path(uploaded_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"上传文件不存在：{src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _session_paths(session_id: str) -> Dict[str, Path]:
    output_dir = OUTPUT_ROOT / session_id
    hybrid_index_dir = INDEX_ROOT / session_id
    visual_index_dir = hybrid_index_dir / "visual_index"
    return {
        "output_dir": output_dir,
        "hybrid_index_dir": hybrid_index_dir,
        "visual_index_dir": visual_index_dir,
        "manifest": output_dir / "manifest.json",
        "ocr": output_dir / "ocr.jsonl",
        "summaries": output_dir / "page_summaries.jsonl",
        "build_meta": output_dir / "build_meta.json",
    }


def _elapsed_seconds(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 3)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_visual_mode(visual_mode: str | None) -> str:
    value = (visual_mode or "required").strip().lower()
    if value not in {"required", "auto", "skip"}:
        raise ValueError(f"visual_mode must be one of required/auto/skip, got {value}.")
    return value


def _normalize_build_options(
    dpi: int,
    load_in_4bit: bool,
    max_pages: int,
    ocr_backend: str,
    visual_mode: str,
) -> BuildOptions:
    dpi_value = int(dpi)
    if dpi_value <= 0:
        raise ValueError(f"DPI must be positive, got {dpi_value}.")

    max_pages_value = int(max_pages)
    if max_pages_value <= 0:
        raise ValueError(f"max_pages must be positive, got {max_pages_value}.")

    return BuildOptions(
        dpi=dpi_value,
        load_in_4bit=bool(load_in_4bit),
        max_pages=max_pages_value,
        ocr_backend=resolve_ocr_backend(ocr_backend),
        visual_mode=_normalize_visual_mode(visual_mode),
    )


def _format_evidence_text(result: Any) -> str:
    lines: List[str] = []
    lines.append("依据：")
    if not result.evidence:
        lines.append("无")
    else:
        for i, ev in enumerate(result.evidence, start=1):
            lines.extend(
                [
                    f"[{i}] 第 {ev['page_index']} 页 score={ev['score']:.4f}",
                    f"image_path={ev['image_path']}",
                    f"summary={ev['summary']}",
                    f"ocr_preview={ev['ocr_text_preview']}",
                    "",
                ]
            )

    lines.append("引用：")
    lines.append("; ".join(result.citations) if result.citations else "无")
    lines.append("")
    lines.append("不确定性：")
    lines.append(result.uncertainty or "无")
    return "\n".join(lines)


def _tokenize_query(text: str) -> List[str]:
    value = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    tokens = [tok for tok in value.split() if len(tok) >= 2]
    if tokens:
        return tokens
    compact = "".join(ch for ch in value if not ch.isspace())
    return [compact] if compact else []


def _load_ocr_blocks_by_page(ocr_jsonl: str | None) -> Dict[int, List[Dict[str, Any]]]:
    if not ocr_jsonl:
        return {}

    path = Path(ocr_jsonl).expanduser()
    if not path.exists():
        return {}

    blocks: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            page_index = int(row.get("page_index", -1))
            if page_index < 0:
                continue
            blocks.setdefault(page_index, []).append(row)
    return blocks


def _select_relevant_blocks(
    blocks: List[Dict[str, Any]],
    query_tokens: List[str],
    max_blocks: int = 18,
) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for block in blocks:
        text = str(block.get("text", "")).lower()
        score = sum(1 for tok in query_tokens if tok and tok in text)
        if score > 0:
            scored.append((score, block))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block for _, block in scored[:max_blocks]]

    return blocks[:max_blocks]


def _draw_highlighted_evidence(
    evidence: List[Dict[str, Any]],
    question: str,
    ocr_jsonl: str | None,
    output_dir: str | None,
) -> List[Tuple[str, str]]:
    blocks_by_page = _load_ocr_blocks_by_page(ocr_jsonl)
    if not blocks_by_page or not output_dir:
        return _gallery_from_evidence(evidence)

    highlight_dir = Path(output_dir) / "evidence_highlights"
    highlight_dir.mkdir(parents=True, exist_ok=True)
    query_tokens = _tokenize_query(question)
    items: List[Tuple[str, str]] = []

    for rank, ev in enumerate(evidence, start=1):
        image_path = str(ev.get("image_path", "")).strip()
        if not image_path:
            continue

        page_index = int(ev.get("page_index", -1))
        selected_blocks = _select_relevant_blocks(blocks_by_page.get(page_index, []), query_tokens)
        score = float(ev.get("score", 0.0))
        caption = f"第 {page_index} 页 | score={score:.4f}"

        if not selected_blocks:
            items.append((image_path, caption))
            continue

        try:
            with Image.open(image_path) as img:
                canvas = img.convert("RGB")
            draw = ImageDraw.Draw(canvas, "RGBA")
            for block in selected_blocks:
                bbox = block.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = [float(x) for x in bbox]
                draw.rectangle([x0, y0, x1, y1], fill=(255, 214, 64, 82), outline=(230, 128, 0, 230), width=3)

            out_path = highlight_dir / f"rank_{rank}_page_{page_index}.png"
            canvas.save(out_path)
            items.append((str(out_path), f"{caption} | highlighted={len(selected_blocks)}"))
        except Exception:
            items.append((image_path, caption))

    return items


def _gallery_from_evidence(evidence: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for ev in evidence:
        image_path = str(ev.get("image_path", "")).strip()
        if not image_path:
            continue
        page_index = int(ev.get("page_index", -1))
        score = float(ev.get("score", 0.0))
        caption = f"第 {page_index} 页 | score={score:.4f}"
        items.append((image_path, caption))
    return items


def build_index(
    uploaded_file: str,
    dpi: int,
    load_in_4bit: bool,
    max_pages: int,
    ocr_backend: str,
    visual_mode: str,
    state: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if not uploaded_file:
        return state, "请先上传 PDF 或图片，再点击“构建索引”。"

    session_id = _new_session_id()
    paths = _session_paths(session_id)
    status_lines = [f"开始构建，会话：{session_id}"]
    total_start = time.perf_counter()
    stage_times: Dict[str, float] = {}

    try:
        options = _normalize_build_options(dpi, load_in_4bit, max_pages, ocr_backend, visual_mode)
        build_config = {
            "session_id": session_id,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "dpi": options.dpi,
            "load_in_4bit": options.load_in_4bit,
            "max_pages": options.max_pages,
            "ocr_backend": options.ocr_backend,
            "visual_mode": options.visual_mode,
        }

        stage_start = time.perf_counter()
        source_file = _copy_uploaded_file(uploaded_file, paths["output_dir"] / "source")
        stage_times["copy_source"] = _elapsed_seconds(stage_start)
        status_lines.append(f"已复制上传文件：{source_file}（{_format_seconds(stage_times['copy_source'])}）")

        stage_start = time.perf_counter()
        pages = ingest_document(
            input_path=str(source_file),
            output_dir=str(paths["output_dir"]),
            dpi=options.dpi,
        )
        rendered_pages = len(pages)
        stage_times["render"] = _elapsed_seconds(stage_start)
        status_lines.append(f"文档渲染完成，共 {rendered_pages} 页（{_format_seconds(stage_times['render'])}）")

        if len(pages) > options.max_pages:
            dropped = pages[options.max_pages:]
            pages = pages[:options.max_pages]
            for p in dropped:
                try:
                    Path(p.image_path).unlink(missing_ok=True)
                except Exception:
                    pass
            status_lines.append(f"PDF 页数较多，Demo 仅处理前 {options.max_pages} 页。")

        save_manifest(pages, str(paths["manifest"]))
        status_lines.append(f"manifest 已保存：{paths['manifest']}")

        stage_start = time.perf_counter()
        ocr_summary = run_ocr_on_manifest(
            str(paths["manifest"]),
            str(paths["ocr"]),
            backend=options.ocr_backend,
        )
        stage_times["ocr"] = _elapsed_seconds(stage_start)
        status_lines.append(
            f"OCR 已完成：{paths['ocr']} | backend={ocr_summary['backend']} | "
            f"pages={ocr_summary['num_pages']} | blocks={ocr_summary['total_blocks']} "
            f"（{_format_seconds(stage_times['ocr'])}）"
        )

        stage_start = time.perf_counter()
        build_page_summaries(
            str(paths["manifest"]),
            str(paths["summaries"]),
            load_in_4bit=options.load_in_4bit,
        )
        stage_times["page_summaries"] = _elapsed_seconds(stage_start)
        status_lines.append(f"页面摘要已完成：{paths['summaries']}（{_format_seconds(stage_times['page_summaries'])}）")

        stage_start = time.perf_counter()
        hybrid = HybridPageIndex()
        hybrid.build(
            manifest_path=str(paths["manifest"]),
            ocr_jsonl=str(paths["ocr"]),
            summary_jsonl=str(paths["summaries"]),
            index_dir=str(paths["hybrid_index_dir"]),
        )
        stage_times["hybrid_index"] = _elapsed_seconds(stage_start)
        status_lines.append(f"Hybrid 索引构建完成：{paths['hybrid_index_dir']}（{_format_seconds(stage_times['hybrid_index'])}）")

        visual_ready = False
        visual_error = ""
        stage_start = time.perf_counter()
        if options.visual_mode == "skip":
            stage_times["visual_index"] = _elapsed_seconds(stage_start)
            status_lines.append(f"Visual 索引构建已按配置跳过（{_format_seconds(stage_times['visual_index'])}）")
        else:
            try:
                visual = VisualPageIndex()
                visual.build(
                    manifest_path=str(paths["manifest"]),
                    index_dir=str(paths["visual_index_dir"]),
                )
                stage_times["visual_index"] = _elapsed_seconds(stage_start)
                status_lines.append(
                    f"Visual/ColPali 索引构建完成：{paths['visual_index_dir']}（{_format_seconds(stage_times['visual_index'])}）"
                )
                visual_ready = True
            except Exception as exc:  # noqa: BLE001
                visual_error = str(exc)
                stage_times["visual_index"] = _elapsed_seconds(stage_start)
                if options.visual_mode == "required":
                    raise RuntimeError(
                        "Visual/ColPali index build is required but failed. "
                        "To run hybrid-only, set Visual Build Mode to auto or skip.\n"
                        f"Original error: {exc}"
                    ) from exc
                status_lines.append(
                    f"Visual/ColPali 索引构建失败，已降级保留 hybrid（{_format_seconds(stage_times['visual_index'])}）：{exc}"
                )

        total_elapsed = _elapsed_seconds(total_start)
        meta = {
            "config": build_config,
            "source_file": str(source_file),
            "rendered_pages": rendered_pages,
            "indexed_pages": len(pages),
            "paths": {
                "output_dir": str(paths["output_dir"]),
                "manifest": str(paths["manifest"]),
                "ocr": str(paths["ocr"]),
                "summaries": str(paths["summaries"]),
                "hybrid_index_dir": str(paths["hybrid_index_dir"]),
                "visual_index_dir": str(paths["visual_index_dir"]),
            },
            "ocr_summary": ocr_summary,
            "visual_ready": visual_ready,
            "visual_error": visual_error,
            "stage_times_seconds": stage_times,
            "total_time_seconds": total_elapsed,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(paths["build_meta"], meta)
        status_lines.append(f"构建元数据已保存：{paths['build_meta']}")
        status_lines.append(f"总耗时：{_format_seconds(total_elapsed)}")

        new_state = {
            "ready": True,
            "session_id": session_id,
            "source_file": str(source_file),
            "output_dir": str(paths["output_dir"]),
            "index_dir": str(paths["hybrid_index_dir"]),
            "visual_index_dir": str(paths["visual_index_dir"]),
            "visual_ready": visual_ready,
            "visual_error": visual_error,
            "manifest_path": str(paths["manifest"]),
            "ocr_path": str(paths["ocr"]),
            "summary_path": str(paths["summaries"]),
            "build_meta_path": str(paths["build_meta"]),
            "page_count": len(pages),
            "load_in_4bit": options.load_in_4bit,
            "ocr_backend": options.ocr_backend,
            "max_pages": options.max_pages,
            "visual_mode": options.visual_mode,
            "stage_times_seconds": stage_times,
            "total_time_seconds": total_elapsed,
        }
        return new_state, "\n".join(status_lines)

    except Exception as exc:  # noqa: BLE001
        err = _friendly_model_error(str(exc))
        trace = traceback.format_exc(limit=1)
        fail_state = {
            "ready": False,
            "session_id": session_id,
            "output_dir": str(paths["output_dir"]),
            "index_dir": str(paths["hybrid_index_dir"]),
            "visual_index_dir": str(paths["visual_index_dir"]),
            "visual_ready": False,
        }
        return fail_state, "\n".join(status_lines + [f"[ERROR] {err}", trace])


def ask_question(
    question: str,
    top_k: int,
    retriever_type: str,
    state: Dict[str, Any],
) -> Tuple[str, str, List[Tuple[str, str]]]:
    if not state or not state.get("ready"):
        return "请先完成“构建索引”。", "", []
    if not question or not question.strip():
        return "请输入问题。", "", []

    retriever_type = (retriever_type or "hybrid").strip().lower()
    if retriever_type in {"visual", "fusion"} and not state.get("visual_ready", False):
        if retriever_type == "visual":
            visual_error = str(state.get("visual_error", "")).strip()
            message = "当前 session 未构建 visual index，无法使用纯 visual 检索。"
            if visual_error:
                message += f"\nVisual 构建错误：{visual_error}"
            message += "\n可以切换到 fusion/hybrid，或安装 Byaldi/ColPali 依赖后重新构建。"
            return message, "", []
        retriever_type = "hybrid"

    try:
        engine = DocQAEngine(
            index_dir=str(state["index_dir"]),
            top_k=int(top_k),
            load_in_4bit=bool(state.get("load_in_4bit", False)),
            retriever_type=retriever_type,
            visual_index_dir=str(state.get("visual_index_dir", "")) if state.get("visual_index_dir") else None,
        )
        result = engine.answer(question.strip())

        answer_lines = [
            "答案：",
            result.answer,
            "",
            "实际检索器：",
            retriever_type,
            "",
            "引用：",
            "; ".join(result.citations) if result.citations else "无",
            "",
            "不确定性：",
            result.uncertainty or "无",
        ]
        answer_text = "\n".join(answer_lines)
        evidence_text = _format_evidence_text(result)
        gallery_items = _draw_highlighted_evidence(
            evidence=result.evidence,
            question=question.strip(),
            ocr_jsonl=str(state.get("ocr_path", "")),
            output_dir=str(state.get("output_dir", "")),
        )
        return answer_text, evidence_text, gallery_items

    except Exception as exc:  # noqa: BLE001
        err = _friendly_model_error(str(exc))
        return f"[ERROR] {err}", "", []


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DocVisRAG Demo") as demo:
        gr.Markdown("## DocVisRAG Demo")
        gr.Markdown("上传 PDF/图片，构建索引后提问，查看答案与检索页面预览。")

        session_state = gr.State({"ready": False})

        with gr.Row():
            file_input = gr.File(
                label="上传文档（PDF/PNG/JPG/JPEG/WEBP）",
                type="filepath",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
            )
            dpi_input = gr.Number(label="DPI", value=180, precision=0)
            load_in_4bit_input = gr.Checkbox(label="4bit 量化", value=False)
            max_pages_input = gr.Slider(label="Max Pages", minimum=1, maximum=30, step=1, value=10)
            ocr_backend_input = gr.Dropdown(
                label="OCR Backend",
                choices=["auto", "paddle", "tesseract"],
                value="auto",
            )
            visual_mode_input = gr.Dropdown(
                label="Visual Build Mode",
                choices=["required", "auto", "skip"],
                value="required",
            )
            build_btn = gr.Button("构建索引", variant="primary")

        build_status = gr.Textbox(label="构建日志", lines=12)

        with gr.Row():
            question_input = gr.Textbox(label="问题", placeholder="请输入你的问题")
            retriever_dropdown = gr.Dropdown(
                label="Retriever",
                choices=["hybrid", "visual", "fusion"],
                value="fusion",
            )
            topk_slider = gr.Slider(label="top_k", minimum=1, maximum=5, step=1, value=3)
            ask_btn = gr.Button("提问", variant="primary")

        answer_output = gr.Textbox(label="答案", lines=8)
        evidence_output = gr.Textbox(label="检索证据", lines=16)
        gallery_output = gr.Gallery(label="检索页面预览（含 OCR 证据高亮）", columns=3, height=360)

        build_btn.click(
            fn=build_index,
            inputs=[
                file_input,
                dpi_input,
                load_in_4bit_input,
                max_pages_input,
                ocr_backend_input,
                visual_mode_input,
                session_state,
            ],
            outputs=[session_state, build_status],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, topk_slider, retriever_dropdown, session_state],
            outputs=[answer_output, evidence_output, gallery_output],
        )

    return demo


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
