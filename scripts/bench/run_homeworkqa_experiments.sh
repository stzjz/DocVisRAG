#!/usr/bin/env bash
set -euo pipefail

# One-key runner for HomeworkQA / HomeworkQA-MC.
# Main QA defaults:
#   hybrid = answer-level fusion over text + raw hybrid answers
#   fusion = answer-level fusion over text + raw hybrid + visual + page-fusion answers
# Raw page/block hybrid and page/block fusion variants are reported under ablation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
PY="${PY:-/data1/home/zengjian/miniconda3/envs/docvisrag/bin/python}"
DEVICE="${CUDA_VISIBLE_DEVICES:-3}"
RUN_ROOT="data/bench_runs/homeworkqa_experiments_${TS}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-/data1/home/zengjian/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
TEXT_MODEL="${TEXT_MODEL:-BAAI/bge-small-zh-v1.5}"
VISUAL_MODEL="${VISUAL_MODEL:-vidore/colqwen2-v1.0}"

DATA_ROOT="${DATA_ROOT:-data/bench_full/homeworkqa}"
QA_QUESTIONS="${QA_QUESTIONS:-${DATA_ROOT}/questions.jsonl}"
MC_QUESTIONS="${MC_QUESTIONS:-data/bench_full/homeworkqa_mc/questions.jsonl}"
BATCH_CONFIG="${BATCH_CONFIG:-${DATA_ROOT}/batch_config.json}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/.work/combined/manifest.json}"
OCR="${OCR:-${DATA_ROOT}/.work/combined/ocr.jsonl}"
SUMMARIES="${SUMMARIES:-${DATA_ROOT}/.work/combined/page_summaries.jsonl}"

TOP_K_QA="${TOP_K_QA:-5}"
TOP_K_MC="${TOP_K_MC:-5}"
MAX_NEW_TOKENS_QA="${MAX_NEW_TOKENS_QA:-96}"
MAX_NEW_TOKENS_MC="${MAX_NEW_TOKENS_MC:-16}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-128}"
ANSWER_FUSION_MAX_NEW_TOKENS="${ANSWER_FUSION_MAX_NEW_TOKENS:-256}"

BUILD_INDEX=1
BUILD_VISUAL=1
RUN_QA=1
RUN_MC=1
RUN_JUDGE=1
SMOKE=0
REUSE_VISUAL=""
LIMIT_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bench/run_homeworkqa_experiments.sh [options]

Options:
  --run-root DIR          Output root. Default: data/bench_runs/homeworkqa_experiments_TIMESTAMP
  --device ID             CUDA_VISIBLE_DEVICES value. Default: 3
  --smoke                 Run one sample for each command.
  --reuse-visual DIR      Use an existing visual index instead of rebuilding it.
  --skip-index            Do not rebuild text/hybrid indexes.
  --skip-visual-build     Do not build visual index; requires --reuse-visual or an existing index path.
  --skip-qa               Skip HomeworkQA short-answer experiments.
  --skip-mc               Skip HomeworkQA-MC experiments.
  --skip-judge            Skip LLM judge and answer-level fusion for QA.
  --top-k-qa K            QA top-k pages. Default: 5.
  --top-k-mc K            MC top-k pages. Default: 5.
  -h, --help              Show this help.

Environment overrides:
  PY, MODEL, JUDGE_MODEL, TEXT_MODEL, VISUAL_MODEL, DATA_ROOT,
  QA_QUESTIONS, MC_QUESTIONS, BATCH_CONFIG, MANIFEST, OCR, SUMMARIES.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --smoke) SMOKE=1; LIMIT_ARGS=(--limit 1); shift ;;
    --reuse-visual) REUSE_VISUAL="$2"; BUILD_VISUAL=0; shift 2 ;;
    --skip-index) BUILD_INDEX=0; shift ;;
    --skip-visual-build) BUILD_VISUAL=0; shift ;;
    --skip-qa) RUN_QA=0; shift ;;
    --skip-mc) RUN_MC=0; shift ;;
    --skip-judge) RUN_JUDGE=0; shift ;;
    --top-k-qa) TOP_K_QA="$2"; shift 2 ;;
    --top-k-mc) TOP_K_MC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$DEVICE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DOCVISRAG_LOCAL_FILES_ONLY="${DOCVISRAG_LOCAL_FILES_ONLY:-1}"

INDEX_DIR="${RUN_ROOT}/indexes"
TEXT_BLOCK_INDEX="${INDEX_DIR}/text_block"
TEXT_PAGE_INDEX="${INDEX_DIR}/text_page"
HYBRID_PAGE_INDEX="${INDEX_DIR}/hybrid_page_ocr_only_lex02"
HYBRID_BLOCK_INDEX="${INDEX_DIR}/hybrid_block_ocr_only_lex02"
VISUAL_INDEX="${INDEX_DIR}/visual"
if [[ -n "$REUSE_VISUAL" ]]; then
  VISUAL_INDEX="$REUSE_VISUAL"
fi

QA_MAIN="${RUN_ROOT}/main/qa"
QA_ABL="${RUN_ROOT}/ablation/qa"
MC_MAIN="${RUN_ROOT}/main/mc"
MC_ABL="${RUN_ROOT}/ablation/mc"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$INDEX_DIR" "$QA_MAIN" "$QA_ABL" "$MC_MAIN" "$MC_ABL" "$LOG_DIR"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run() {
  local name="$1"
  shift
  log "$name"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$QA_QUESTIONS"
require_file "$MC_QUESTIONS"
require_file "$BATCH_CONFIG"
require_file "$MANIFEST"
require_file "$OCR"
require_file "$SUMMARIES"

log "Run root: ${RUN_ROOT}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "QA main uses answer-level hybrid/fusion; page/block variants are ablations."

if [[ "$BUILD_INDEX" -eq 1 ]]; then
  run build_text_block "$PY" scripts/retrieve/build_text_index.py \
    --ocr "$OCR" --index-dir "$TEXT_BLOCK_INDEX" --model-name "$TEXT_MODEL" \
    --granularity block

  run build_text_page "$PY" scripts/retrieve/build_text_index.py \
    --ocr "$OCR" --index-dir "$TEXT_PAGE_INDEX" --model-name "$TEXT_MODEL" \
    --granularity page

  run build_hybrid_page "$PY" scripts/retrieve/build_hybrid_index.py \
    --manifest "$MANIFEST" --ocr "$OCR" --summaries "$SUMMARIES" \
    --index-dir "$HYBRID_PAGE_INDEX" --model-name "$TEXT_MODEL" \
    --text-mode ocr_only --lexical-weight 0.2 --granularity page

  run build_hybrid_block "$PY" scripts/retrieve/build_hybrid_index.py \
    --manifest "$MANIFEST" --ocr "$OCR" --summaries "$SUMMARIES" \
    --index-dir "$HYBRID_BLOCK_INDEX" --model-name "$TEXT_MODEL" \
    --text-mode ocr_only --lexical-weight 0.2 --granularity block
fi

if [[ "$BUILD_VISUAL" -eq 1 ]]; then
  run build_visual "$PY" scripts/retrieve/build_visual_index.py \
    --manifest "$MANIFEST" --index-dir "$VISUAL_INDEX" --model-id "$VISUAL_MODEL"
fi

if [[ ! -d "$VISUAL_INDEX" ]]; then
  echo "[ERROR] Visual index not found: $VISUAL_INDEX" >&2
  echo "Use --reuse-visual DIR or allow visual index building." >&2
  exit 1
fi

run_qa_mode() {
  local out_dir="$1"
  local tag="$2"
  local mode="$3"
  local text_index="$4"
  local hybrid_index="$5"
  shift 5

  run "qa_${tag}" "$PY" scripts/bench/eval_homeworkqa_qa.py \
    --questions "$QA_QUESTIONS" \
    --batch-config "$BATCH_CONFIG" \
    --text-index-dir "$text_index" \
    --hybrid-index-dir "$hybrid_index" \
    --visual-index-dir "$VISUAL_INDEX" \
    --out "${out_dir}/${tag}.jsonl" \
    --summary-out "${out_dir}/summary_${tag}.json" \
    --mode "$mode" \
    --top-k "$TOP_K_QA" \
    --model-id "$MODEL" \
    --max-new-tokens "$MAX_NEW_TOKENS_QA" \
    "${LIMIT_ARGS[@]}" \
    "$@"
}

judge_qa() {
  local out_dir="$1"
  local tag="$2"
  if [[ "$RUN_JUDGE" -eq 0 ]]; then
    return
  fi
  run "judge_${tag}" "$PY" scripts/bench/judge_homeworkqa_qa.py \
    --predictions "${out_dir}/${tag}.jsonl" \
    --out "${out_dir}/judge_${tag}.jsonl" \
    --summary-out "${out_dir}/judge_${tag}_summary.json" \
    --judge-model "$JUDGE_MODEL" \
    --max-new-tokens "$JUDGE_MAX_NEW_TOKENS" \
    --resume \
    "${LIMIT_ARGS[@]}"
}

if [[ "$RUN_QA" -eq 1 ]]; then
  # Base answers needed by answer-level main experiments.
  run_qa_mode "$QA_MAIN" text text "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX"
  run_qa_mode "$QA_MAIN" raw_hybrid_block hybrid "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX"
  run_qa_mode "$QA_MAIN" visual visual "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX"
  run_qa_mode "$QA_MAIN" raw_page_fusion fusion "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX" \
    --fusion-text-weight 0.2 --fusion-hybrid-weight 5 --fusion-visual-weight 2

  judge_qa "$QA_MAIN" text
  judge_qa "$QA_MAIN" raw_hybrid_block
  judge_qa "$QA_MAIN" visual
  judge_qa "$QA_MAIN" raw_page_fusion

  if [[ "$RUN_JUDGE" -eq 1 ]]; then
    run qa_hybrid_answer_fusion "$PY" scripts/bench/answer_fusion_text_hybrid.py \
      --text "${QA_MAIN}/text.jsonl" \
      --hybrid "${QA_MAIN}/raw_hybrid_block.jsonl" \
      --out "${QA_MAIN}/hybrid_answer_fusion.jsonl" \
      --summary-out "${QA_MAIN}/summary_hybrid_answer_fusion.json" \
      --judge-model "$JUDGE_MODEL" \
      --max-new-tokens "$ANSWER_FUSION_MAX_NEW_TOKENS" \
      --resume

    run qa_answer_fusion "$PY" scripts/bench/answer_fusion_homeworkqa.py \
      --text "${QA_MAIN}/text.jsonl" \
      --hybrid "${QA_MAIN}/raw_hybrid_block.jsonl" \
      --visual "${QA_MAIN}/visual.jsonl" \
      --fusion "${QA_MAIN}/raw_page_fusion.jsonl" \
      --out "${QA_MAIN}/answer_fusion.jsonl" \
      --summary-out "${QA_MAIN}/summary_answer_fusion.json" \
      --judge-model "$JUDGE_MODEL" \
      --max-new-tokens "$ANSWER_FUSION_MAX_NEW_TOKENS" \
      --resume

    judge_qa "$QA_MAIN" hybrid_answer_fusion
    judge_qa "$QA_MAIN" answer_fusion
  fi

  # Ablations: page/block hybrid/fusion, visual-dominant fusion, and visual-preserving snippet fusion.
  run_qa_mode "$QA_ABL" hybrid_page hybrid "$TEXT_PAGE_INDEX" "$HYBRID_PAGE_INDEX"
  run_qa_mode "$QA_ABL" hybrid_block hybrid "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX"
  run_qa_mode "$QA_ABL" fusion_page fusion "$TEXT_PAGE_INDEX" "$HYBRID_PAGE_INDEX" \
    --fusion-text-weight 0.2 --fusion-hybrid-weight 5 --fusion-visual-weight 2
  run_qa_mode "$QA_ABL" fusion_block fusion "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX" \
    --fusion-text-weight 0.2 --fusion-hybrid-weight 5 --fusion-visual-weight 2
  run_qa_mode "$QA_ABL" fusion_visual_dominant fusion "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX" \
    --fusion-text-weight 0 --fusion-hybrid-weight 0.5 --fusion-visual-weight 10 \
    --fusion-text-candidates 0 --fusion-hybrid-candidates 50 --fusion-visual-candidates 50

  if [[ -f scripts/bench/eval_homeworkqa_qa_visual_anchor.py ]]; then
    for snippet_mode in fusion_visual_snippet fusion_block_snippet; do
      run "qa_${snippet_mode}" "$PY" scripts/bench/eval_homeworkqa_qa_visual_anchor.py \
        --questions "$QA_QUESTIONS" \
        --batch-config "$BATCH_CONFIG" \
        --text-index-dir "$TEXT_BLOCK_INDEX" \
        --hybrid-index-dir "$HYBRID_BLOCK_INDEX" \
        --visual-index-dir "$VISUAL_INDEX" \
        --out "${QA_ABL}/${snippet_mode}.jsonl" \
        --summary-out "${QA_ABL}/summary_${snippet_mode}.json" \
        --mode "$snippet_mode" \
        --top-k "$TOP_K_QA" \
        --model-id "$MODEL" \
        --max-new-tokens "$MAX_NEW_TOKENS_QA" \
        "${LIMIT_ARGS[@]}"
      judge_qa "$QA_ABL" "$snippet_mode"
    done
  fi

  for tag in hybrid_page hybrid_block fusion_page fusion_block fusion_visual_dominant; do
    judge_qa "$QA_ABL" "$tag"
  done
fi

run_mc_modes() {
  local out_dir="$1"
  local tag="$2"
  local text_index="$3"
  local hybrid_index="$4"
  shift 4

  run "mc_${tag}" "$PY" scripts/bench/eval_homeworkqa_mc.py \
    --questions "$MC_QUESTIONS" \
    --batch-config "$BATCH_CONFIG" \
    --text-index-dir "$text_index" \
    --hybrid-index-dir "$hybrid_index" \
    --visual-index-dir "$VISUAL_INDEX" \
    --out "${out_dir}/${tag}.jsonl" \
    --summary-out "${out_dir}/summary_${tag}.json" \
    --modes text hybrid visual fusion \
    --top-k "$TOP_K_MC" \
    --retrieval-query-mode question_choices \
    --model-id "$MODEL" \
    --max-new-tokens "$MAX_NEW_TOKENS_MC" \
    "${LIMIT_ARGS[@]}" \
    "$@"
}

if [[ "$RUN_MC" -eq 1 ]]; then
  run_mc_modes "$MC_MAIN" main_block "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX" \
    --fusion-text-weight 0.2 --fusion-hybrid-weight 5 --fusion-visual-weight 2

  run_mc_modes "$MC_ABL" page_level "$TEXT_PAGE_INDEX" "$HYBRID_PAGE_INDEX" \
    --fusion-text-weight 0.2 --fusion-hybrid-weight 5 --fusion-visual-weight 2

  run_mc_modes "$MC_ABL" visual_dominant "$TEXT_BLOCK_INDEX" "$HYBRID_BLOCK_INDEX" \
    --fusion-text-weight 0 --fusion-hybrid-weight 0.5 --fusion-visual-weight 10 \
    --fusion-text-candidates 0 --fusion-hybrid-candidates 50 --fusion-visual-candidates 50
fi

REPORT="${RUN_ROOT}/REPORT.md"
{
  echo "# HomeworkQA Experiment Report"
  echo
  echo "- run_root: \`${RUN_ROOT}\`"
  echo "- qa_top_k: \`${TOP_K_QA}\`"
  echo "- mc_top_k: \`${TOP_K_MC}\`"
  echo "- model: \`${MODEL}\`"
  echo "- judge_model: \`${JUDGE_MODEL}\`"
  echo "- visual_index: \`${VISUAL_INDEX}\`"
  echo
  echo "## Main QA"
  echo
  echo "| method | prediction | relaxed/auto summary | judge summary |"
  echo "|---|---|---|---|"
  echo "| text | \`main/qa/text.jsonl\` | \`main/qa/summary_text.json\` | \`main/qa/judge_text_summary.json\` |"
  echo "| hybrid-answer | \`main/qa/hybrid_answer_fusion.jsonl\` | \`main/qa/summary_hybrid_answer_fusion.json\` | \`main/qa/judge_hybrid_answer_fusion_summary.json\` |"
  echo "| visual | \`main/qa/visual.jsonl\` | \`main/qa/summary_visual.json\` | \`main/qa/judge_visual_summary.json\` |"
  echo "| fusion-answer | \`main/qa/answer_fusion.jsonl\` | \`main/qa/summary_answer_fusion.json\` | \`main/qa/judge_answer_fusion_summary.json\` |"
  echo
  echo "## Ablations"
  echo
  echo "- QA page/block/raw fusion outputs: \`ablation/qa/\`"
  echo "- MC page/block/visual-dominant outputs: \`ablation/mc/\`"
  echo
  echo "## Metric Files"
  find "$RUN_ROOT" -name 'summary_*.json' -o -name 'judge_*_summary.json' | sort | sed 's#^#- `#; s#$#`#'
} > "$REPORT"

log "Done. Report: ${REPORT}"

if command -v jq >/dev/null 2>&1; then
  log "QA judge summaries"
  find "$QA_MAIN" "$QA_ABL" -name 'judge_*_summary.json' -print0 2>/dev/null \
    | xargs -0 -r -I{} sh -c 'echo "### {}"; jq ".summary // ." "{}"'
  log "MC summaries"
  find "$MC_MAIN" "$MC_ABL" -name 'summary_*.json' -print0 2>/dev/null \
    | xargs -0 -r -I{} sh -c 'echo "### {}"; jq "." "{}"'
fi
