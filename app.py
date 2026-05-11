import datetime as dt
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from src.docvisrag.ingest import build_page_summaries, ingest_document, run_ocr_on_manifest, save_manifest
from src.docvisrag.qa import DocQAEngine
from src.docvisrag.retrieve import HybridPageIndex


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes"
MAX_DEMO_PAGES = 10


def _new_session_id() -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"session_{ts}_{short}"


def _friendly_model_error(err_text: str) -> str:
    lower = (err_text or "").lower()
    if "failed to load vlm model" in lower or "failed to load qwen" in lower:
        return (
            "模型加载失败，常见原因：\n"
            "1) 模型未下载且当前网络不可用；\n"
            "2) 显存不足；\n"
            "3) transformers 版本不支持该模型；\n"
            "4) model id 配置错误。\n"
            f"\n原始错误：{err_text}"
        )
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
    index_dir = INDEX_ROOT / session_id
    return {
        "output_dir": output_dir,
        "index_dir": index_dir,
        "manifest": output_dir / "manifest.json",
        "ocr": output_dir / "ocr.jsonl",
        "summaries": output_dir / "page_summaries.jsonl",
    }


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


def build_index(uploaded_file: str, dpi: int, state: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if not uploaded_file:
        return state, "请先上传 PDF 或图片，再点击“构建索引”。"

    session_id = _new_session_id()
    paths = _session_paths(session_id)
    status_lines = [f"开始构建，会话：{session_id}"]

    try:
        source_file = _copy_uploaded_file(uploaded_file, paths["output_dir"] / "source")
        status_lines.append(f"已复制上传文件：{source_file}")

        pages = ingest_document(
            input_path=str(source_file),
            output_dir=str(paths["output_dir"]),
            dpi=int(dpi),
        )
        status_lines.append(f"文档渲染完成，共 {len(pages)} 页")

        if len(pages) > MAX_DEMO_PAGES:
            dropped = pages[MAX_DEMO_PAGES:]
            pages = pages[:MAX_DEMO_PAGES]
            for p in dropped:
                try:
                    Path(p.image_path).unlink(missing_ok=True)
                except Exception:
                    pass
            status_lines.append(f"PDF 页数较多，Demo 仅处理前 {MAX_DEMO_PAGES} 页。")

        save_manifest(pages, str(paths["manifest"]))
        status_lines.append(f"manifest 已保存：{paths['manifest']}")

        run_ocr_on_manifest(str(paths["manifest"]), str(paths["ocr"]))
        status_lines.append(f"OCR 已完成：{paths['ocr']}")

        build_page_summaries(str(paths["manifest"]), str(paths["summaries"]))
        status_lines.append(f"页面摘要已完成：{paths['summaries']}")

        index = HybridPageIndex()
        index.build(
            manifest_path=str(paths["manifest"]),
            ocr_jsonl=str(paths["ocr"]),
            summary_jsonl=str(paths["summaries"]),
            index_dir=str(paths["index_dir"]),
        )
        status_lines.append(f"Hybrid 索引构建完成：{paths['index_dir']}")

        new_state = {
            "ready": True,
            "session_id": session_id,
            "source_file": str(source_file),
            "output_dir": str(paths["output_dir"]),
            "index_dir": str(paths["index_dir"]),
            "manifest_path": str(paths["manifest"]),
            "page_count": len(pages),
        }
        return new_state, "\n".join(status_lines)

    except Exception as exc:  # noqa: BLE001
        err = _friendly_model_error(str(exc))
        trace = traceback.format_exc(limit=1)
        fail_state = {
            "ready": False,
            "session_id": session_id,
            "output_dir": str(paths["output_dir"]),
            "index_dir": str(paths["index_dir"]),
        }
        return fail_state, "\n".join(status_lines + [f"[ERROR] {err}", trace])


def ask_question(
    question: str,
    top_k: int,
    state: Dict[str, Any],
) -> Tuple[str, str, List[Tuple[str, str]]]:
    if not state or not state.get("ready"):
        return "请先完成“构建索引”。", "", []
    if not question or not question.strip():
        return "请输入问题。", "", []

    try:
        engine = DocQAEngine(
            index_dir=str(state["index_dir"]),
            top_k=int(top_k),
        )
        result = engine.answer(question.strip())

        answer_lines = [
            "答案：",
            result.answer,
            "",
            "引用：",
            "; ".join(result.citations) if result.citations else "无",
            "",
            "不确定性：",
            result.uncertainty or "无",
        ]
        answer_text = "\n".join(answer_lines)
        evidence_text = _format_evidence_text(result)
        gallery_items = _gallery_from_evidence(result.evidence)
        return answer_text, evidence_text, gallery_items

    except Exception as exc:  # noqa: BLE001
        err = _friendly_model_error(str(exc))
        return f"[ERROR] {err}", "", []


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DocVisRAG Demo") as demo:
        gr.Markdown("## DocVisRAG 阶段 7 Demo")
        gr.Markdown("上传 PDF/图片，构建索引后提问，查看答案与检索页面预览。")

        session_state = gr.State({"ready": False})

        with gr.Row():
            file_input = gr.File(
                label="上传文档（PDF/PNG/JPG/JPEG/WEBP）",
                type="filepath",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"],
            )
            dpi_input = gr.Number(label="DPI", value=180, precision=0)
            build_btn = gr.Button("构建索引", variant="primary")

        build_status = gr.Textbox(label="构建日志", lines=12)

        with gr.Row():
            question_input = gr.Textbox(label="问题", placeholder="请输入你的问题")
            topk_slider = gr.Slider(label="top_k", minimum=1, maximum=5, step=1, value=3)
            ask_btn = gr.Button("提问", variant="primary")

        answer_output = gr.Textbox(label="答案", lines=8)
        evidence_output = gr.Textbox(label="检索证据", lines=16)
        gallery_output = gr.Gallery(label="检索页面预览", columns=3, height=360)

        build_btn.click(
            fn=build_index,
            inputs=[file_input, dpi_input, session_state],
            outputs=[session_state, build_status],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, topk_slider, session_state],
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
