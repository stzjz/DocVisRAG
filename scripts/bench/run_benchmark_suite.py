import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_ALIASES = {
    "docvqa": "docvqa",
    "chartqa": "chartqa",
    "chartvqa": "chartqa",
    "textvqa": "textvqa",
}


@dataclass
class BenchSpec:
    name: str
    manifest: str
    questions: str
    benchmark: Optional[str] = None
    dataset_id: Optional[str] = None
    split: Optional[str] = None
    prepare_limit: Optional[int] = None


@dataclass
class RunResult:
    name: str
    run_dir: str
    success: bool
    retrieval_eval: str
    qa_predictions: str
    qa_summary: str
    error_analysis: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_markdown(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _normalize_jsonl_to_utf8_no_bom(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8-sig")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def _run_cmd(cmd: List[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as logf:
        logf.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError(f"Command failed to start stdout stream: {' '.join(cmd)}")

        for line in proc.stdout:
            logf.write(line)
            print(line, end="")

        proc.stdout.close()
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed ({return_code}): {' '.join(cmd)}")


def _canonical_benchmark(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    k = value.strip().lower()
    return BENCHMARK_ALIASES.get(k)


def _infer_benchmark_type(spec: BenchSpec) -> Optional[str]:
    if spec.benchmark:
        return _canonical_benchmark(spec.benchmark)

    hay = " ".join([spec.name, spec.manifest, spec.questions]).lower()
    for key in ["docvqa", "chartqa", "chartvqa", "textvqa"]:
        if key in hay:
            return _canonical_benchmark(key)
    return None


def _common_parent(a: Path, b: Path) -> Optional[Path]:
    pa = a.parent.resolve()
    pb = b.parent.resolve()
    return pa if pa == pb else None


def _auto_prepare_if_missing(spec: BenchSpec, args: argparse.Namespace, log_file: Path) -> BenchSpec:
    manifest_path = Path(spec.manifest).expanduser().resolve()
    questions_path = Path(spec.questions).expanduser().resolve()

    need_prepare = args.force_prepare or (not manifest_path.exists()) or (not questions_path.exists())
    if not need_prepare:
        return spec

    if not args.auto_prepare_missing:
        missing = []
        if not manifest_path.exists():
            missing.append(str(manifest_path))
        if not questions_path.exists():
            missing.append(str(questions_path))
        raise FileNotFoundError(f"Missing required inputs and auto prepare disabled: {missing}")

    bench_type = _infer_benchmark_type(spec)
    if bench_type not in {"docvqa", "chartqa", "textvqa"}:
        raise RuntimeError(
            "Cannot infer benchmark type for auto prepare. "
            "Please set spec.benchmark in batch config, or include docvqa/chartqa/chartvqa/textvqa in --name/path."
        )

    target_dir = _common_parent(manifest_path, questions_path)
    if target_dir is None:
        target_dir = Path(args.auto_prepare_out_root).expanduser().resolve() / bench_type

    cmd = [
        sys.executable,
        "scripts/bench/prepare_benchmark.py",
        "--benchmark",
        bench_type,
        "--out-dir",
        str(target_dir),
    ]

    dataset_id = spec.dataset_id or args.auto_prepare_dataset_id
    if dataset_id:
        cmd += ["--dataset-id", dataset_id]

    split = spec.split or args.auto_prepare_split
    if split:
        cmd += ["--split", split]

    prepare_limit = spec.prepare_limit if spec.prepare_limit is not None else args.auto_prepare_limit
    if prepare_limit is not None:
        cmd += ["--limit", str(prepare_limit)]

    if args.auto_prepare_streaming:
        cmd += ["--streaming"]

    reason = "force_prepare" if args.force_prepare else "missing_files"
    print(f"[INFO] auto preparing benchmark '{bench_type}' for '{spec.name}' (reason={reason})...")
    try:
        _run_cmd(cmd, log_file)
    except Exception as exc:  # noqa: BLE001
        generated_manifest = target_dir / "manifest.json"
        generated_questions = target_dir / "questions.jsonl"
        if generated_manifest.exists() and generated_questions.exists():
            with log_file.open("a", encoding="utf-8") as f:
                f.write(
                    "\n[WARN] prepare command returned non-zero, but required files exist. "
                    f"Continue. Error: {exc}\n"
                )
            print("[WARN] prepare step non-zero but files exist; continue with generated inputs.")
        else:
            raise

    generated_manifest = target_dir / "manifest.json"
    generated_questions = target_dir / "questions.jsonl"
    if not generated_manifest.exists() or not generated_questions.exists():
        raise RuntimeError(
            "Auto prepare finished but required files are missing: "
            f"manifest={generated_manifest.exists()} questions={generated_questions.exists()}"
        )

    spec.manifest = str(generated_manifest)
    spec.questions = str(generated_questions)
    spec.benchmark = bench_type
    return spec


def _load_bench_specs(args: argparse.Namespace) -> List[BenchSpec]:
    if args.batch_config:
        cfg_path = Path(args.batch_config).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Batch config not found: {cfg_path}")
        data = _read_json(cfg_path)
        rows = data.get("benchmarks", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("Batch config must be a list or {\"benchmarks\": [...]}")

        specs: List[BenchSpec] = []
        for i, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"batch item #{i} is not an object")

            name = str(row.get("name", f"bench_{i:02d}"))
            benchmark_raw = row.get("benchmark")
            benchmark = _canonical_benchmark(str(benchmark_raw)) if benchmark_raw else None
            manifest = row.get("manifest")
            questions = row.get("questions")

            if (manifest is None or questions is None) and benchmark:
                root = Path(args.auto_prepare_out_root).expanduser().resolve() / benchmark
                manifest = str(root / "manifest.json")
                questions = str(root / "questions.jsonl")

            if manifest is None or questions is None:
                raise ValueError(
                    f"batch item '{name}' must provide manifest/questions, "
                    "or provide benchmark for auto paths"
                )

            specs.append(
                BenchSpec(
                    name=name,
                    manifest=str(manifest),
                    questions=str(questions),
                    benchmark=benchmark,
                    dataset_id=str(row["dataset_id"]) if row.get("dataset_id") else None,
                    split=str(row["split"]) if row.get("split") else None,
                    prepare_limit=int(row["prepare_limit"]) if row.get("prepare_limit") is not None else None,
                )
            )
        return specs

    if not args.manifest or not args.questions:
        if not args.name:
            raise ValueError("Provide --manifest/--questions or at least --name for auto prepare.")

        requested_name = args.name.strip().lower()
        benchmark = _canonical_benchmark(requested_name)
        if benchmark is None:
            raise ValueError(
                "For auto prepare single-run, --name must be one of: docvqa, chartqa/chartvqa, textvqa."
            )

        root = Path(args.auto_prepare_out_root).expanduser().resolve() / requested_name
        return [
            BenchSpec(
                name=args.name,
                manifest=str(root / "manifest.json"),
                questions=str(root / "questions.jsonl"),
                benchmark=benchmark,
            )
        ]

    return [
        BenchSpec(
            name=args.name or "benchmark",
            manifest=args.manifest,
            questions=args.questions,
            benchmark=_canonical_benchmark(args.name) if args.name else None,
        )
    ]


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_prepare_meta(questions_src: Path) -> Dict[str, Any]:
    meta_path = questions_src.parent / "prepare_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return _read_json(meta_path)
    except Exception:  # noqa: BLE001
        return {}


def _write_run_dataset_format(run_dir: Path, run_meta: Dict[str, Any]) -> None:
    lines = [
        "# Dataset Format",
        "",
        f"- benchmark: `{run_meta.get('benchmark') or ''}`",
        f"- dataset_id: `{run_meta.get('dataset_id') or ''}`",
        f"- split: `{run_meta.get('split') or ''}`",
        "",
        "## Input Schema",
        "",
        "### manifest.json",
        "- JSON array, one item per page.",
        "- Fields: `doc_id`, `source_path`, `page_index`(1-based), `image_path`, `width`, `height`.",
        "",
        "### questions.jsonl",
        "- JSONL, one item per question.",
        "- Fields: `id`, `doc_path`, `question`, `answer`, `evidence_pages`(1-based int list), `type`.",
        "",
        "## Output Schema",
        "",
        "### results/eval_retrieval.json",
        "- Overall retrieval metrics and per-question details.",
        "",
        "### results/predictions.jsonl",
        "- QA predictions, citations, and evidence page info.",
        "",
        "### summary.json",
        "- Run summary with benchmark metadata and aggregated metrics.",
    ]
    _write_markdown(run_dir / "dataset_format.md", lines)


def _run_single_benchmark(spec: BenchSpec, args: argparse.Namespace, suite_dir: Path) -> RunResult:
    run_dir = suite_dir / spec.name
    inputs_dir = run_dir / "inputs"
    inter_dir = run_dir / "intermediate"
    results_dir = run_dir / "results"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"

    for d in [inputs_dir, inter_dir, results_dir, reports_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "pipeline.log"

    spec = _auto_prepare_if_missing(spec, args, log_file)

    manifest_src = Path(spec.manifest).expanduser().resolve()
    questions_src = Path(spec.questions).expanduser().resolve()
    if not manifest_src.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_src}")
    if not questions_src.exists():
        raise FileNotFoundError(f"Questions not found: {questions_src}")

    prepare_meta = _load_prepare_meta(questions_src)

    manifest_snapshot = inputs_dir / "manifest.snapshot.json"
    questions_copy = inputs_dir / "questions.jsonl"
    _safe_copy(manifest_src, manifest_snapshot)
    _normalize_jsonl_to_utf8_no_bom(questions_src, questions_copy)

    run_meta = {
        "name": spec.name,
        "benchmark": spec.benchmark,
        "dataset_id": prepare_meta.get("dataset_id", spec.dataset_id),
        "split": prepare_meta.get("split", spec.split),
        "retriever_type": args.retriever_type,
        "created_at": datetime.now().isoformat(),
        "source_manifest": str(manifest_src),
        "source_questions": str(questions_src),
        "args": vars(args),
    }
    _write_json(run_dir / "run_meta.json", run_meta)
    _write_run_dataset_format(run_dir, run_meta)

    ocr_jsonl = inter_dir / "ocr.jsonl"
    summaries_jsonl = inter_dir / "page_summaries.jsonl"
    index_dir = inter_dir / "hybrid_index"
    visual_index_dir = inter_dir / "visual_index"

    retrieval_json = results_dir / "eval_retrieval.json"
    predictions_jsonl = results_dir / "predictions.jsonl"
    predictions_summary_json = results_dir / "predictions.summary.json"
    error_analysis_md = reports_dir / "error_analysis.md"

    _run_cmd(
        [sys.executable, "scripts/ingest/run_ocr.py", "--manifest", str(manifest_src), "--out", str(ocr_jsonl)],
        log_file,
    )

    cmd_summary = [
        sys.executable,
        "scripts/ingest/build_page_summaries.py",
        "--manifest",
        str(manifest_src),
        "--out",
        str(summaries_jsonl),
    ]
    if args.summary_model_id:
        cmd_summary += ["--model-id", args.summary_model_id]
    _run_cmd(cmd_summary, log_file)

    _run_cmd(
        [
            sys.executable,
            "scripts/retrieve/build_hybrid_index.py",
            "--manifest",
            str(manifest_src),
            "--ocr",
            str(ocr_jsonl),
            "--summaries",
            str(summaries_jsonl),
            "--index-dir",
            str(index_dir),
            "--model-name",
            args.index_model_name,
        ],
        log_file,
    )

    if args.retriever_type in {"visual", "fusion"}:
        cmd_visual = [
            sys.executable,
            "scripts/retrieve/build_visual_index.py",
            "--manifest",
            str(manifest_src),
            "--index-dir",
            str(visual_index_dir),
        ]
        if args.visual_model_id:
            cmd_visual += ["--model-id", args.visual_model_id]
        _run_cmd(cmd_visual, log_file)

    cmd_eval_retrieval = [
        sys.executable,
        "scripts/eval/eval_retrieval.py",
        "--questions",
        str(questions_copy),
        "--index-dir",
        str(index_dir),
        "--retriever-type",
        str(args.retriever_type),
        "--out",
        str(retrieval_json),
        "--top-k",
        str(args.retrieval_top_k),
    ]
    if args.retriever_type in {"visual", "fusion"}:
        cmd_eval_retrieval += ["--visual-index-dir", str(visual_index_dir)]
    _run_cmd(cmd_eval_retrieval, log_file)

    qa_success = True
    if not args.skip_qa:
        cmd_qa = [
            sys.executable,
            "scripts/eval/eval_qa.py",
            "--questions",
            str(questions_copy),
            "--index-dir",
            str(index_dir),
            "--retriever-type",
            str(args.retriever_type),
            "--out",
            str(predictions_jsonl),
            "--top-k",
            str(args.qa_top_k),
            "--max-new-tokens",
            str(args.qa_max_new_tokens),
        ]
        if args.retriever_type in {"visual", "fusion"}:
            cmd_qa += ["--visual-index-dir", str(visual_index_dir)]
        if args.qa_limit is not None:
            cmd_qa += ["--limit", str(args.qa_limit)]
        if args.qa_model_id:
            cmd_qa += ["--model-id", args.qa_model_id]
        if args.qa_load_in_4bit:
            cmd_qa += ["--load-in-4bit"]

        try:
            _run_cmd(cmd_qa, log_file)
            src_summary = predictions_jsonl.with_suffix(predictions_jsonl.suffix + ".summary.json")
            if src_summary.exists():
                _safe_copy(src_summary, predictions_summary_json)

            _run_cmd(
                [
                    sys.executable,
                    "scripts/eval/make_error_analysis.py",
                    "--predictions",
                    str(predictions_jsonl),
                    "--out",
                    str(error_analysis_md),
                ],
                log_file,
            )
        except Exception as exc:  # noqa: BLE001
            qa_success = False
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n[WARN] QA stage failed: {exc}\n")

    summary_payload: Dict[str, Any] = {
        "name": spec.name,
        "benchmark": run_meta.get("benchmark"),
        "dataset_id": run_meta.get("dataset_id"),
        "split": run_meta.get("split"),
        "run_dir": str(run_dir),
        "retrieval_eval": str(retrieval_json),
        "qa_predictions": str(predictions_jsonl),
        "qa_summary": str(predictions_summary_json),
        "error_analysis": str(error_analysis_md),
        "qa_success": qa_success,
    }

    try:
        retrieval_obj = _read_json(retrieval_json)
        summary_payload["retrieval_overall"] = retrieval_obj.get("overall", {})
    except Exception:  # noqa: BLE001
        pass

    if predictions_summary_json.exists():
        try:
            qa_obj = _read_json(predictions_summary_json)
            summary_payload["qa_overall"] = {
                "num_questions": qa_obj.get("num_questions", 0),
                "em": qa_obj.get("em", 0.0),
                "f1": qa_obj.get("f1", 0.0),
                "anls": qa_obj.get("anls", 0.0),
            }
        except Exception:  # noqa: BLE001
            pass

    _write_json(run_dir / "summary.json", summary_payload)
    return RunResult(
        name=spec.name,
        run_dir=str(run_dir),
        success=True,
        retrieval_eval=str(retrieval_json),
        qa_predictions=str(predictions_jsonl),
        qa_summary=str(predictions_summary_json),
        error_analysis=str(error_analysis_md),
    )


def _write_suite_benchmarks_md(suite_dir: Path) -> None:
    lines = [
        "# Suite Benchmarks",
        "",
        "| run_name | benchmark | dataset_id | split | qa_success |",
        "|---|---|---|---|---|",
    ]

    for run_dir in sorted([p for p in suite_dir.iterdir() if p.is_dir()]):
        summary_file = run_dir / "summary.json"
        if not summary_file.exists():
            continue
        try:
            s = _read_json(summary_file)
        except Exception:  # noqa: BLE001
            continue
        lines.append(
            "| "
            f"{run_dir.name} | {s.get('benchmark','')} | {s.get('dataset_id','')} | {s.get('split','')} | {s.get('qa_success','')} |"
        )

    _write_markdown(suite_dir / "benchmarks.md", lines)


def _write_suite_format_readme(suite_dir: Path) -> None:
    lines = [
        "# Suite Output Format",
        "",
        "## Directory Layout",
        "",
        "```text",
        "suite_xxx/",
        "  benchmarks.md",
        "  overview.json",
        "  suite_meta.json",
        "  <run_name>/",
        "    dataset_format.md",
        "    inputs/",
        "    intermediate/",
        "    results/",
        "    reports/",
        "    logs/",
        "    run_meta.json",
        "    summary.json",
        "```",
        "",
        "## Notes",
        "- Archive is NOT generated by default. Add `--archive` only when needed.",
        "- Auto-prepare outputs benchmark files into `data/bench/<name>/...` for single-run mode.",
    ]
    _write_markdown(suite_dir / "README.md", lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end benchmark pipeline and keep all intermediates/results in one clean subdirectory."
    )

    parser.add_argument("--manifest", default=None, help="Single benchmark manifest path")
    parser.add_argument("--questions", default=None, help="Single benchmark questions jsonl path")
    parser.add_argument("--name", default=None, help="Single benchmark name")

    parser.add_argument("--batch-config", default=None, help="Batch config JSON path")

    parser.add_argument("--out-root", default="data/bench_runs", help="Benchmark output root dir")
    parser.add_argument("--suite-name", default=None, help="Suite directory name")

    parser.add_argument("--index-model-name", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--summary-model-id", default=None)
    parser.add_argument(
        "--retriever-type",
        default="hybrid",
        choices=["hybrid", "visual", "fusion"],
        help="Retriever type used for retrieval/QA evaluation in benchmark suite.",
    )
    parser.add_argument(
        "--visual-model-id",
        default=None,
        help="Optional visual retriever model id for building visual index.",
    )

    parser.add_argument("--qa-model-id", default=None)
    parser.add_argument("--qa-load-in-4bit", action="store_true")
    parser.add_argument("--qa-top-k", type=int, default=3)
    parser.add_argument("--qa-max-new-tokens", type=int, default=512)
    parser.add_argument("--qa-limit", type=int, default=None)
    parser.add_argument("--skip-qa", action="store_true")

    parser.add_argument("--retrieval-top-k", type=int, default=5)

    parser.add_argument("--auto-prepare-missing", action="store_true", default=True)
    parser.add_argument("--no-auto-prepare-missing", action="store_false", dest="auto_prepare_missing")
    parser.add_argument("--force-prepare", action="store_true", help="Force re-download/convert benchmark inputs")
    parser.add_argument("--auto-prepare-out-root", default="data/bench")
    parser.add_argument("--auto-prepare-dataset-id", default=None)
    parser.add_argument("--auto-prepare-split", default=None)
    parser.add_argument("--auto-prepare-limit", type=int, default=None)
    parser.add_argument("--auto-prepare-streaming", action="store_true")

    parser.add_argument("--archive", action="store_true", help="Generate tar.gz archive (disabled by default)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        specs = _load_bench_specs(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Invalid benchmark specification: {exc}")
        return 1

    suite_name = args.suite_name or f"suite_{_timestamp()}"
    out_root = Path(args.out_root).expanduser().resolve()
    suite_dir = out_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        suite_dir / "suite_meta.json",
        {
            "created_at": datetime.now().isoformat(),
            "suite_name": suite_name,
            "num_benchmarks": len(specs),
            "args": vars(args),
            "benchmarks": [asdict(s) for s in specs],
        },
    )

    results: List[RunResult] = []
    failed: List[str] = []

    for spec in specs:
        print(f"\n===== Running benchmark: {spec.name} =====")
        try:
            result = _run_single_benchmark(spec, args, suite_dir)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            failed.append(spec.name)
            print(f"[ERROR] benchmark {spec.name} failed: {exc}")

    overview = {
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "total": len(specs),
        "success": len(results),
        "failed": failed,
        "runs": [asdict(r) for r in results],
    }
    _write_json(suite_dir / "overview.json", overview)
    _write_suite_benchmarks_md(suite_dir)
    _write_suite_format_readme(suite_dir)

    archive_path = None
    if args.archive:
        archive_path = shutil.make_archive(str(suite_dir), "gztar", root_dir=str(suite_dir))

    print("\n[OK] Benchmark suite finished.")
    print(f"- suite_dir: {suite_dir}")
    print(f"- success: {len(results)}/{len(specs)}")
    if failed:
        print(f"- failed: {failed}")
    if archive_path:
        print(f"- archive: {archive_path}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

