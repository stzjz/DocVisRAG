#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

INPUT_REL="${INPUT_REL:-data/samples/test_stage1.png}"
OUTPUT_ROOT_REL="${OUTPUT_ROOT_REL:-data/outputs/loopback_smoke}"
INDEX_ROOT_REL="${INDEX_ROOT_REL:-data/indexes/loopback_smoke}"
BENCH_ROOT_REL="${BENCH_ROOT_REL:-data/bench_runs}"

QUESTION_TEXT="${QUESTION_TEXT:-请概括这个文档的主要内容。}"
TOP_K="${TOP_K:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

RUN_VISUAL="${RUN_VISUAL:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_DEMO="${RUN_DEMO:-1}"
RUN_LAYOUT="${RUN_LAYOUT:-1}"
RUN_ENV_VISUAL="${RUN_ENV_VISUAL:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
DEMO_PORT="${DEMO_PORT:-7860}"
DEMO_START_TIMEOUT="${DEMO_START_TIMEOUT:-90}"

HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

LOG_ROOT="$PROJECT_DIR/$OUTPUT_ROOT_REL/logs"
RUN_DIR="$PROJECT_DIR/$OUTPUT_ROOT_REL"
INDEX_DIR="$PROJECT_DIR/$INDEX_ROOT_REL"
INPUT_PATH="$PROJECT_DIR/$INPUT_REL"
MANIFEST_PATH="$RUN_DIR/manifest.json"
OCR_PATH="$RUN_DIR/ocr.jsonl"
LAYOUT_PATH="$RUN_DIR/layout.jsonl"
CHUNKS_PATH="$RUN_DIR/layout_chunks.jsonl"
SUMMARIES_PATH="$RUN_DIR/page_summaries.jsonl"
TEXT_INDEX_DIR="$INDEX_DIR/text_index"
LAYOUT_INDEX_DIR="$INDEX_DIR/layout_index"
HYBRID_INDEX_DIR="$INDEX_DIR/hybrid_index"
VISUAL_INDEX_DIR="$INDEX_DIR/visual_index"
BENCH_QUESTIONS_PATH="$RUN_DIR/loopback_questions.jsonl"
SUITE_NAME="loopback_$(date +%Y%m%d_%H%M%S)"
DEMO_PID=""

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_ROOT/$name.log"
  mkdir -p "$LOG_ROOT"
  log "Running: $name"
  (
    cd "$PROJECT_DIR"
    "$@"
  ) 2>&1 | tee "$log_file"
}

cleanup() {
  if [[ -n "$DEMO_PID" ]]; then
    kill "$DEMO_PID" >/dev/null 2>&1 || true
    wait "$DEMO_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_ENDPOINT HF_HUB_DISABLE_XET

[[ -x "$PYTHON_BIN" ]] || die "Python not found in venv: $PYTHON_BIN"
require_file "$INPUT_PATH"

mkdir -p "$RUN_DIR" "$INDEX_DIR" "$LOG_ROOT"
rm -rf "$RUN_DIR" "$INDEX_DIR"
mkdir -p "$RUN_DIR" "$INDEX_DIR" "$LOG_ROOT"

log "Project dir: $PROJECT_DIR"
log "Input: $INPUT_REL"
log "Output root: $OUTPUT_ROOT_REL"
log "Index root: $INDEX_ROOT_REL"

run_step env_check "$PYTHON_BIN" scripts/env/check_env.py
if truthy "$RUN_ENV_VISUAL"; then
  run_step env_check_visual "$PYTHON_BIN" scripts/env/check_env.py --visual
fi

run_step ingest_render \
  "$PYTHON_BIN" scripts/ingest/ingest_render.py \
  --input "$INPUT_REL" \
  --output "$OUTPUT_ROOT_REL" \
  --dpi 180

run_step page_qa \
  "$PYTHON_BIN" scripts/qa/page_qa.py \
  --manifest "$OUTPUT_ROOT_REL/manifest.json" \
  --page 1 \
  --question "$QUESTION_TEXT"

run_step run_ocr \
  "$PYTHON_BIN" scripts/ingest/run_ocr.py \
  --manifest "$OUTPUT_ROOT_REL/manifest.json" \
  --out "$OUTPUT_ROOT_REL/ocr.jsonl"

if truthy "$RUN_LAYOUT"; then
  run_step run_layout \
    "$PYTHON_BIN" scripts/ingest/run_layout.py \
    --manifest "$OUTPUT_ROOT_REL/manifest.json" \
    --out "$OUTPUT_ROOT_REL/layout.jsonl" \
    --backend auto

  run_step run_layout_chunk \
    "$PYTHON_BIN" scripts/ingest/run_layout_chunk.py \
    --layout "$OUTPUT_ROOT_REL/layout.jsonl" \
    --ocr "$OUTPUT_ROOT_REL/ocr.jsonl" \
    --out "$OUTPUT_ROOT_REL/layout_chunks.jsonl"

  if [[ -s "$CHUNKS_PATH" ]]; then
    run_step build_layout_index \
      "$PYTHON_BIN" scripts/retrieve/build_layout_index.py \
      --chunks "$OUTPUT_ROOT_REL/layout_chunks.jsonl" \
      --index-dir "$INDEX_ROOT_REL/layout_index"
  else
    log "Layout chunks are empty for this sample, skipping layout index build"
  fi
fi

run_step build_text_index \
  "$PYTHON_BIN" scripts/retrieve/build_text_index.py \
  --ocr "$OUTPUT_ROOT_REL/ocr.jsonl" \
  --index-dir "$INDEX_ROOT_REL/text_index"

run_step text_search \
  "$PYTHON_BIN" scripts/retrieve/text_search.py \
  --index-dir "$INDEX_ROOT_REL/text_index" \
  --question "$QUESTION_TEXT" \
  --top-k "$TOP_K"

summary_cmd=("$PYTHON_BIN" scripts/ingest/build_page_summaries.py --manifest "$OUTPUT_ROOT_REL/manifest.json" --out "$OUTPUT_ROOT_REL/page_summaries.jsonl")
if truthy "$LOAD_IN_4BIT"; then
  summary_cmd+=(--load-in-4bit)
fi
run_step build_page_summaries "${summary_cmd[@]}"

run_step build_hybrid_index \
  "$PYTHON_BIN" scripts/retrieve/build_hybrid_index.py \
  --manifest "$OUTPUT_ROOT_REL/manifest.json" \
  --ocr "$OUTPUT_ROOT_REL/ocr.jsonl" \
  --summaries "$OUTPUT_ROOT_REL/page_summaries.jsonl" \
  --index-dir "$INDEX_ROOT_REL/hybrid_index"

run_step hybrid_search \
  "$PYTHON_BIN" scripts/retrieve/hybrid_search.py \
  --index-dir "$INDEX_ROOT_REL/hybrid_index" \
  --question "$QUESTION_TEXT" \
  --top-k "$TOP_K"

doc_qa_cmd=("$PYTHON_BIN" scripts/qa/doc_qa.py --index-dir "$INDEX_ROOT_REL/hybrid_index" --question "$QUESTION_TEXT" --top-k "$TOP_K" --max-new-tokens "$MAX_NEW_TOKENS")
if truthy "$LOAD_IN_4BIT"; then
  doc_qa_cmd+=(--load-in-4bit)
fi
run_step doc_qa_hybrid "${doc_qa_cmd[@]}"

run_step make_loopback_questions \
  "$PYTHON_BIN" -c "
import json
from pathlib import Path

manifest = json.loads(Path('$MANIFEST_PATH').read_text(encoding='utf-8'))
summary_row = json.loads(Path('$SUMMARIES_PATH').read_text(encoding='utf-8').splitlines()[0])
sample = {
    'id': 'loopback_q1',
    'doc_path': manifest[0]['source_path'],
    'question': '$QUESTION_TEXT',
    'answer': summary_row['summary'],
    'evidence_pages': [1],
    'type': 'summary',
}
Path('$BENCH_QUESTIONS_PATH').write_text(json.dumps(sample, ensure_ascii=False) + '\n', encoding='utf-8')
print(Path('$BENCH_QUESTIONS_PATH'))
"

if truthy "$RUN_VISUAL"; then
  run_step build_visual_index \
    "$PYTHON_BIN" scripts/retrieve/build_visual_index.py \
    --manifest "$OUTPUT_ROOT_REL/manifest.json" \
    --index-dir "$INDEX_ROOT_REL/visual_index"

  run_step visual_search \
    "$PYTHON_BIN" scripts/retrieve/visual_search.py \
    --index-dir "$INDEX_ROOT_REL/visual_index" \
    --question "$QUESTION_TEXT" \
    --top-k "$TOP_K"

  fusion_qa_cmd=(
    "$PYTHON_BIN" scripts/qa/doc_qa.py
    --index-dir "$INDEX_ROOT_REL/hybrid_index"
    --visual-index-dir "$INDEX_ROOT_REL/visual_index"
    --retriever-type fusion
    --question "$QUESTION_TEXT"
    --top-k "$TOP_K"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )
  if truthy "$LOAD_IN_4BIT"; then
    fusion_qa_cmd+=(--load-in-4bit)
  fi
  run_step doc_qa_fusion "${fusion_qa_cmd[@]}"
fi

if truthy "$RUN_BENCHMARK"; then
  bench_cmd=(
    "$PYTHON_BIN" scripts/bench/run_benchmark_suite.py
    --suite-name "$SUITE_NAME"
    --name loopback_hybrid
    --manifest "$OUTPUT_ROOT_REL/manifest.json"
    --questions "$OUTPUT_ROOT_REL/loopback_questions.jsonl"
    --out-root "$BENCH_ROOT_REL"
    --retriever-type hybrid
    --qa-top-k "$TOP_K"
    --qa-max-new-tokens "$MAX_NEW_TOKENS"
  )
  if truthy "$LOAD_IN_4BIT"; then
    bench_cmd+=(--qa-load-in-4bit)
  fi
  run_step bench_hybrid "${bench_cmd[@]}"

  if truthy "$RUN_VISUAL"; then
    bench_fusion_cmd=(
      "$PYTHON_BIN" scripts/bench/run_benchmark_suite.py
      --suite-name "${SUITE_NAME}_fusion"
      --name loopback_fusion
      --manifest "$OUTPUT_ROOT_REL/manifest.json"
      --questions "$OUTPUT_ROOT_REL/loopback_questions.jsonl"
      --out-root "$BENCH_ROOT_REL"
      --retriever-type fusion
      --qa-top-k "$TOP_K"
      --qa-max-new-tokens "$MAX_NEW_TOKENS"
    )
    if truthy "$LOAD_IN_4BIT"; then
      bench_fusion_cmd+=(--qa-load-in-4bit)
    fi
    run_step bench_fusion "${bench_fusion_cmd[@]}"
  fi
fi

if truthy "$RUN_DEMO"; then
  local_demo_log="$LOG_ROOT/demo.log"
  log "Starting Gradio demo on port $DEMO_PORT"
  (
    cd "$PROJECT_DIR"
    "$PYTHON_BIN" app.py
  ) >"$local_demo_log" 2>&1 &
  DEMO_PID="$!"

  demo_ready=0
  for ((i = 0; i < DEMO_START_TIMEOUT; i++)); do
    if curl -fsS "http://127.0.0.1:${DEMO_PORT}/" >/dev/null 2>&1; then
      demo_ready=1
      break
    fi
    sleep 1
  done

  if [[ "$demo_ready" != "1" ]]; then
    die "Gradio demo did not become ready on port $DEMO_PORT within ${DEMO_START_TIMEOUT}s"
  fi

  log "Gradio demo responded successfully"
  kill "$DEMO_PID" >/dev/null 2>&1 || true
  wait "$DEMO_PID" >/dev/null 2>&1 || true
  DEMO_PID=""
fi

log "Loopback validation completed"
printf 'outputs: %s\n' "$RUN_DIR"
printf 'indexes: %s\n' "$INDEX_DIR"
printf 'logs: %s\n' "$LOG_ROOT"
