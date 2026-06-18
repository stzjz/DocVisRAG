# Text-aware Fusion Optimization

本文记录 DocVQA / ChartQA / TextVQA 对比实验中 `text / hybrid / fusion` 三种模式的优化背景、代码改动和推荐评估方式。小样本调参结论以 2026-06-05 之后在本服务器上重跑的 text-aware fusion 实验为准；全量 validation 检索结果见 2026-06-15 章节。

## 当前推荐配置

Text-aware fusion 将 OCR chunk 级 `text`、页面级 `hybrid` 和页面视觉 `visual` 三路结果按加权 RRF 混合。最新小样本检索结果显示，DocVQA 和 ChartQA 更适合 hybrid-heavy 配置，而不是早期的 text-heavy 默认值。

| 数据集 | 推荐权重 | 候选深度 | 结论 |
|---|---|---|---|
| DocVQA | `text=0.2, hybrid=5.0, visual=0.01` | `text=50, hybrid=10, visual=10` | `fusion >= hybrid > text`，fusion 在 R@3/R@5/MRR/NDCG@5 上更稳 |
| ChartQA | `text=1.0, hybrid=5.0, visual=0.02` | `text=50, hybrid=10, visual=10` | `fusion > hybrid > text`，按 R@1/MRR/NDCG@5 看最优 |
| TextVQA | 暂不推荐作为 fusion 正向证据 | 需重做 OCR/visual 分支 | 当前 `hybrid > fusion`，主要瓶颈是自然图片 OCR 漏检和 visual 分支偏弱 |

推荐先用 `--skip-qa` 做检索层权重扫描，再用 `--qa-limit` 做小样本生成验证。README 和仓库根目录 `ins` 中给出了可直接运行的命令模板。

注意：2026-06-15 的全量 validation 检索结果采用 OCR-only hybrid 作为全量可承受配置，结果并不支持“fusion 全量稳定优于 hybrid”。DocVQA/ChartQA 全量上 hybrid 的 R@1/MRR/NDCG@5 更高；DocVQA fusion 只在 R@3 上极小幅超过 hybrid、R@5 打平。TextVQA OCR-only fusion 略高于 hybrid，但整体指标很低，不应作为 fusion 正向证据。

## 2026-06-15 全量 validation 检索结果

本轮跑的是公开 validation/val 全量检索评估，不是 hidden test；只评估检索层，不跑 QA 生成。输出统一放在 `data/bench_runs/full_20260615/`。

全量运行配置：

- DocVQA validation：5349 题，1286 页；OCR-only hybrid，`lexical_weight=0.2`；fusion=`text=0.2, hybrid=5.0, visual=0.01`，候选深度 `50/10/10`。
- ChartQA val：1920 题，1055 张图；OCR-only hybrid，`lexical_weight=0.2`；fusion=`text=1.0, hybrid=5.0, visual=0.02`，候选深度 `50/10/10`。
- TextVQA validation：5000 题，3166 张图；OCR-only diagnostic 配置，`lexical_weight=0.2`；fusion=`text=0.2, hybrid=5.0, visual=0.1`，候选深度 `50/10/10`。本轮未重跑全量 VLM summary，因为成本较高。

| 数据集 | 模式 | 配置 | N | R@1 | R@3 | R@5 | MRR | NDCG@5 | 输出文件 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| DocVQA | text | OCR chunk text index | 5349 | 0.0580 | 0.1165 | 0.1610 | 0.0997 | 0.1100 | `full_20260615/docvqa/eval_retrieval_text.json` |
| DocVQA | hybrid | OCR-only + lexical=0.2 | 5349 | 0.2434 | 0.3085 | 0.3330 | 0.2812 | 0.2918 | `full_20260615/docvqa/eval_retrieval_hybrid_ocr_only_lex02.json` |
| DocVQA | fusion | text=0.2, hybrid=5.0, visual=0.01 | 5349 | 0.2245 | 0.3090 | 0.3330 | 0.2713 | 0.2845 | `full_20260615/docvqa/eval_retrieval_fusion_t02_h5_v001_ocr_only_lex02.json` |
| ChartQA | text | OCR chunk text index | 1920 | 0.0599 | 0.1182 | 0.1536 | 0.0997 | 0.1088 | `full_20260615/chartqa/eval_retrieval_text.json` |
| ChartQA | hybrid | OCR-only + lexical=0.2 | 1920 | 0.1953 | 0.2427 | 0.2635 | 0.2255 | 0.2318 | `full_20260615/chartqa/eval_retrieval_hybrid_ocr_only_lex02.json` |
| ChartQA | fusion | text=1.0, hybrid=5.0, visual=0.02 | 1920 | 0.1495 | 0.2240 | 0.2589 | 0.1947 | 0.2069 | `full_20260615/chartqa/eval_retrieval_fusion_t1_h5_v002_ocr_only_lex02.json` |
| TextVQA | text | tesseract OCR text index | 5000 | 0.0152 | 0.0278 | 0.0330 | 0.0216 | 0.0245 | `full_20260615/textvqa/eval_retrieval_text.json` |
| TextVQA | hybrid | OCR-only + lexical=0.2 | 5000 | 0.0246 | 0.0346 | 0.0398 | 0.0300 | 0.0324 | `full_20260615/textvqa/eval_retrieval_hybrid_ocr_only_lex02.json` |
| TextVQA | visual | ColQwen2 visual index | 5000 | 0.0004 | 0.0014 | 0.0018 | 0.0010 | 0.0012 | `full_20260615/textvqa/eval_retrieval_visual_retry.json` |
| TextVQA | fusion | text=0.2, hybrid=5.0, visual=0.1 | 5000 | 0.0246 | 0.0362 | 0.0416 | 0.0309 | 0.0336 | `full_20260615/textvqa/eval_retrieval_fusion_t02_h5_v01_ocr_only_lex02_retry.json` |

全量结论：

- DocVQA：`hybrid > fusion > text`，但 fusion 在 R@3 上略高于 hybrid、R@5 打平；主排序指标 R@1/MRR/NDCG@5 仍是 hybrid 更好。
- ChartQA：`hybrid > fusion > text`，fusion 没有复现 100 题小样本上的优势。
- TextVQA：OCR-only diagnostic 下 `fusion >= hybrid > text > visual`，fusion 相对 hybrid 只有很小提升；由于 visual 极弱、整体指标低，不能作为 text-aware fusion 有效性的强证据。
- 因此当前更稳妥的表述是：text-aware fusion 是有价值的消融方向，但 full validation 上需要继续调权重和改进 visual/OCR 分支，不能只用小样本结果宣称 fusion 全面优于 hybrid。


## 背景

DocVQA 的大量问题答案直接出现在 OCR 文本短片段中，例如数字、名称、日期、表单字段和页面中的局部短语。原始 `text` 基线使用 OCR chunk 级检索，粒度细；而原来的 `hybrid` 是页面级向量，`fusion` 只融合 `hybrid + visual` 两个页面级结果。

因此在 DocVQA 上出现 `text` 强于 `hybrid/fusion` 是合理现象：它不一定说明视觉信息无用，而是说明旧版 fusion 没有纳入最强的 OCR chunk 信号，且页面级 embedding 容易稀释短答案。

## 本次优化

### 1. Fusion 纳入 text 检索信号

新增 `weighted_reciprocal_rank_fusion`，支持多路加权 RRF：

```text
fusion = weighted_RRF(text_pages, hybrid_pages, visual_pages)
早期默认权重：text=2.0, hybrid=0.3, visual=0.1；最新推荐以文档开头的 hybrid-heavy 配置为准。
```

同时新增 `text_chunks_to_page_results`，将 text chunk 检索结果聚合成 page-level 结果，并保留每页 top OCR 命中片段。这样 fusion 排名仍然是页面级，但不会丢失 text 的精细定位能力。

### 2. 多模态 QA 注入 OCR 命中片段

`DocQAEngine` 在 `retriever_type=fusion` 且传入 `text_index_dir` 时，会：

1. 检索 hybrid 页面候选；
2. 检索 visual 页面候选；
3. 检索 text OCR chunks；
4. 将 OCR chunks 聚合到页面；
5. 通过加权 RRF 融合三路结果；
6. 把命中的 OCR 片段写入 VLM prompt。

这解决了“fusion 找到页面但 prompt 里没有答案文字”的问题。

### 3. 兼容旧流程

如果不传 `--text-index-dir`，`fusion` 仍回退为旧的 `hybrid + visual` RRF，便于做消融实验。

## 受影响文件

- `src/docvisrag/retrieve/fusion.py`：新增 text 聚合和加权 RRF。
- `src/docvisrag/qa/doc_qa.py`：fusion 可选加载 text index，并向 prompt 注入 OCR 命中片段。
- `scripts/qa/doc_qa.py`：新增 `--text-index-dir` 和 fusion 权重参数。
- `scripts/eval/eval_retrieval.py`：fusion 评估支持 `--text-index-dir`。
- `scripts/eval/eval_qa.py`：fusion QA 评估支持 `--text-index-dir`。
- `scripts/bench/run_benchmark_suite.py`：一键 benchmark 的 fusion 模式自动传递 text index。
- `scripts/bench/compare_rag.py`：对比脚本中的 fusion 检索和 QA 使用 text-aware fusion。

## 推荐三模式评估命令

建议先固定同一批 DocVQA 问题和同一套 OCR/页面渲染产物，再分别跑三种模式。基础模型后续可以直接换 3B/4B 测试，但检索参数建议先固定：

```bash
# text
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_text \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type text \
  --qa-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --qa-top-k 5 \
  --retrieval-top-k 10

# hybrid
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_hybrid \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type hybrid \
  --qa-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --qa-top-k 5 \
  --retrieval-top-k 10

# text-aware fusion: DocVQA 推荐权重，先只跑检索
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_fusion_textaware_t02_h5_v001 \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type fusion \
  --hybrid-text-mode ocr_only \
  --hybrid-lexical-weight 0.2 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 0.01 \
  --fusion-text-candidates 50 \
  --fusion-hybrid-candidates 10 \
  --fusion-visual-candidates 10 \
  --retrieval-top-k 10 \
  --skip-qa

# QA 小样本 sanity check
python scripts/bench/run_benchmark_suite.py \
  --name docvqa_fusion_textaware_t02_h5_v001_qa10 \
  --manifest data/bench/docvqa_small/manifest.json \
  --questions data/bench/docvqa_small/questions.jsonl \
  --out-root data/bench_runs \
  --retriever-type fusion \
  --hybrid-text-mode ocr_only \
  --hybrid-lexical-weight 0.2 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 0.01 \
  --fusion-text-candidates 50 \
  --fusion-hybrid-candidates 10 \
  --fusion-visual-candidates 10 \
  --qa-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --qa-top-k 5 \
  --qa-limit 10 \
  --retrieval-top-k 10
```

如果显存紧张，可先加 `--skip-qa` 只评估检索，或加 `--qa-limit N` 小样本验证。

## 解读建议

不要只看整体平均值，必须按 `type` 分桶看：

- `text` 应该在纯文本字段、短答案问题上很强；
- `hybrid` 应该对页面摘要、跨区域文字问题更稳；
- `fusion` 应该在表格、图、布局和视觉依赖问题上提升，同时尽量不牺牲 text 类问题。

如果新版 fusion 在 text 类问题上接近 text，并在 table/chart/layout 上超过 text，说明优化方向正确。


## 2026-06-05 三数据集重跑汇总

合并远端 `dev-next` 后，重新使用现有 100 题 small benchmark 和已构建索引跑了 DocVQA、ChartQA、TextVQA 的 `text / hybrid / fusion` 检索对比。输出统一放在 `data/bench_runs/rerun_20260605/`。

| 数据集 | 模式 | 配置 | R@1 | R@3 | R@5 | MRR | NDCG@5 | 输出文件 |
|---|---|---|---:|---:|---:|---:|---:|---|
| DocVQA | text | OCR chunk text index | 0.4700 | 0.7100 | 0.8200 | 0.6156 | 0.6627 | `rerun_20260605/docvqa/eval_retrieval_text.json` |
| DocVQA | hybrid | OCR-only + lexical=0.2 | 0.6600 | 0.7900 | 0.8300 | 0.7354 | 0.7549 | `rerun_20260605/docvqa/eval_retrieval_hybrid_ocr_only_lex02.json` |
| DocVQA | fusion | text=0.2, hybrid=5.0, visual=0.01 | 0.6600 | 0.8000 | 0.8400 | 0.7404 | 0.7625 | `rerun_20260605/docvqa/eval_retrieval_fusion_t02_h5_v001_ocr_only_lex02.json` |
| ChartQA | text | tesseract OCR text index | 0.1600 | 0.2600 | 0.4300 | 0.2478 | 0.2863 | `rerun_20260605/chartqa/eval_retrieval_text_tesseract.json` |
| ChartQA | hybrid | OCR-only + lexical=0.2 | 0.2300 | 0.4100 | 0.4800 | 0.3353 | 0.3619 | `rerun_20260605/chartqa/eval_retrieval_hybrid_ocr_only_lex02.json` |
| ChartQA | fusion | text=1.0, hybrid=5.0, visual=0.02 | 0.2500 | 0.3900 | 0.4800 | 0.3504 | 0.3723 | `rerun_20260605/chartqa/eval_retrieval_fusion_t1_h5_v002_ocr_only_lex02.json` |
| TextVQA | text | tesseract OCR text index | 0.0700 | 0.1300 | 0.1400 | 0.0975 | 0.1069 | `rerun_20260605/textvqa/eval_retrieval_text_tesseract.json` |
| TextVQA | hybrid | summary+OCR + lexical=0.2 | 0.1900 | 0.2500 | 0.3600 | 0.2569 | 0.2687 | `rerun_20260605/textvqa/eval_retrieval_hybrid_summary_ocr_lex02.json` |
| TextVQA | fusion | text=0.2, hybrid=5.0, visual=0.1 | 0.1200 | 0.2500 | 0.3400 | 0.2119 | 0.2272 | `rerun_20260605/textvqa/eval_retrieval_fusion_t02_h5_v01_summary_ocr_lex02.json` |

重跑结论：

- DocVQA：`fusion >= hybrid > text`。fusion 的 R@1 与 hybrid 打平，但 R@3/R@5/MRR/NDCG@5 更高。
- ChartQA：按 R@1/MRR/NDCG@5 看，`fusion > hybrid > text`；R@5 上 fusion 与 hybrid 打平。
- TextVQA：当前仍是 `hybrid > fusion > text` 或 `hybrid > text`，不能作为 fusion 优于 hybrid 的证据。主要问题是 tesseract 对自然图片文字漏检明显，visual 分支也偏弱。

## 本次 DocVQA small 实测

环境：`docvisrag` conda 环境，DocVQA validation 前 100 个问题，26 张唯一页面。检索评估使用 `retrieval-top-k=10`，QA 小样本使用本地缓存 `Qwen/Qwen3-VL-4B-Instruct`、前 10 题、`qa-top-k=5`、`max_new_tokens=128`。

### 检索评估（100 题）

| 模式 | R@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| text | 0.4700 | 0.7100 | 0.8200 | 0.6156 | 0.6627 |
| hybrid, 旧 summary-only | 0.2500 | 0.4800 | 0.5500 | 0.3886 | 0.4152 |
| hybrid, OCR-only + lexical rerank | 0.6600 | 0.7900 | 0.8300 | 0.7354 | 0.7549 |
| fusion, text=1.0/hybrid=0.6/visual=0.4 | 0.4100 | 0.7200 | 0.8000 | 0.5770 | 0.6235 |
| fusion, text=2.0/hybrid=0.3/visual=0.1 | 0.5000 | 0.7500 | 0.8300 | 0.6419 | 0.6839 |
| fusion, text=0.2/hybrid=5.0/visual=0.01 | 0.6600 | 0.8000 | 0.8400 | 0.7404 | 0.7625 |

结论：旧版 summary-only `hybrid` 明显弱于原始 `text`。加入 `ocr_only + lexical_weight=0.2` 后，hybrid 已超过 text；再用 tuned text-aware fusion（`text=0.2, hybrid=5.0, visual=0.01, candidates=50/10/10`）可在 R@3/R@5/MRR/NDCG@5 上进一步超过 hybrid，R@1 与 hybrid 打平。因此 DocVQA 检索层可以表述为 `fusion >= hybrid > text`，更严谨地说是 fusion 在多数排序指标上优于 hybrid。

### QA 小样本（前 10 题，Qwen3-VL-4B）

| 模式 | EM | F1 | ANLS | Recall@3 | CitationAcc | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0000 | 0.0000 | 0.0000 | 0.9000 | 0.9000 | 1.14s |
| hybrid | 0.5000 | 0.9242 | 0.8233 | 1.0000 | 1.0000 | 9.32s |
| fusion | 0.3000 | 0.6576 | 0.6233 | 0.7000 | 0.7000 | 9.75s |

注意：text QA 的检索页命中率高，但生成阶段只看到零散 OCR chunk，表格邻域不足，模型倾向输出 `Not enough evidence.`。下一步可以为 text QA 增加同页 OCR 邻域拼接或 page-level OCR context，避免 text 检索强但生成弱。

## 本次 TextVQA small 检索实测

环境：`docvisrag` conda 环境，TextVQA validation 前 100 个问题，64 张唯一图片。数据准备时 `hf-mirror` 多次 HEAD 超时，脚本最终退出码为 134，但已成功写出 100 条问题和 64 张图片；本轮只作为小样本检索诊断，不作为完整 TextVQA 结论。

已生成文件：

- `data/bench/textvqa_small/questions.jsonl`：100 题。
- `data/bench/textvqa_small/manifest.json`：64 张图。
- `data/bench_runs/textvqa_textaware_eval/ocr_tesseract.jsonl`：tesseract OCR，495 个 OCR blocks；不少自然图片 OCR 为 0 block。
- `data/bench_runs/textvqa_textaware_eval/text_index_tesseract`：基于 tesseract OCR 的原始 text index。
- `data/bench_runs/textvqa_textaware_eval/page_summaries_qwen3vl4b.jsonl`：Qwen3-VL-4B 页面摘要。
- `data/bench_runs/textvqa_textaware_eval/visual_index`：ColQwen visual index。

### 检索评估（100 题）

| 模式 | 配置 | R@1 | R@3 | R@5 | MRR | NDCG@5 | 输出文件 |
|---|---|---:|---:|---:|---:|---:|---|
| text, tesseract OCR | - | 0.0700 | 0.1300 | 0.1400 | 0.0975 | 0.1069 | `eval_retrieval_text_tesseract.json` |
| hybrid, OCR-only + lexical rerank | text_mode=ocr_only, lexical=0.2 | 0.1000 | 0.1200 | 0.1500 | 0.1235 | 0.1251 | `eval_retrieval_hybrid_ocr_only_lex02.json` |
| hybrid, summary-only + lexical rerank | text_mode=summary_only, lexical=0.2 | 0.1800 | 0.2400 | 0.3500 | 0.2502 | 0.2587 | `eval_retrieval_hybrid_summary_only_lex02.json` |
| hybrid, summary+OCR + lexical rerank | text_mode=summary_ocr, lexical=0.2 | 0.1900 | 0.2500 | 0.3600 | 0.2569 | 0.2687 | `eval_retrieval_hybrid_summary_ocr_lex02.json` |
| visual | ColQwen2 | 0.0300 | 0.0600 | 0.0900 | 0.0603 | 0.0597 | `eval_retrieval_visual.json` |
| fusion, OCR-only hybrid 主导 | text=0.2, hybrid=5.0, visual=0.01 | 0.1000 | 0.1200 | 0.1500 | 0.1233 | 0.1251 | `eval_retrieval_fusion_t02_h5_v001_ocr_only_lex02.json` |
| fusion, summary+OCR hybrid 主导 | text=0.2, hybrid=5.0, visual=0.01 | 0.1100 | 0.2400 | 0.3400 | 0.2055 | 0.2229 | `eval_retrieval_fusion_t02_h5_v001_summary_ocr_lex02.json` |
| fusion, summary+OCR + 更高 visual | text=0.2, hybrid=5.0, visual=0.1 | 0.1200 | 0.2500 | 0.3400 | 0.2119 | 0.2272 | `eval_retrieval_fusion_t02_h5_v01_summary_ocr_lex02.json` |

结论：TextVQA 小样本上，tesseract OCR 漏检严重，原始 `text` 很弱。Qwen3-VL-4B 生成的页面摘要对自然图片更有帮助，`summary_ocr + lexical_weight=0.2` 的 hybrid 明显超过 text。但当前 ColQwen visual 单路也偏弱，RRF fusion 加入弱 text/visual 分支后没有超过最强 hybrid。离线 score-normalized fusion 扫描同样没有超过 `summary_ocr hybrid`，说明主要瓶颈不是权重，而是 TextVQA 当前两条辅助分支信号不足。

因此 TextVQA 暂时不能作为 `fusion > hybrid > text` 的正向证据。更合适的后续优化是换更强的自然场景 OCR/文字检测（PaddleOCR 恢复、EasyOCR、CRAFT/DBNet、或 VLM OCR prompt），并把 fusion 改成“hybrid 保护 + 仅一致性加分”的策略，避免弱分支扰动高置信 hybrid 排序。

## 本次 ChartQA small 检索实测

环境：`docvisrag` conda 环境，ChartQA 前 100 个问题，49 张唯一页面。PaddleOCR 在当前机器上仍会以 `code=-4` 崩溃，本轮改用 conda-forge 安装的 `tesseract 5.5.2` 生成真实 OCR，并补齐原始 `text` 基线与 text-aware fusion。

已生成文件：

- `data/bench/chartqa_small/questions.jsonl`：100 题。
- `data/bench/chartqa_small/manifest.json`：49 页。
- `data/bench_runs/chartqa_textaware_eval/page_summaries_qwen3vl4b.jsonl`：Qwen3-VL-4B 页面摘要。
- `data/bench_runs/chartqa_textaware_eval/hybrid_index`：summary-only hybrid index。
- `data/bench_runs/chartqa_textaware_eval/visual_index`：ColQwen visual index。
- `data/bench_runs/chartqa_textaware_eval/ocr_tesseract.jsonl`：tesseract OCR，3065 个 OCR blocks。
- `data/bench_runs/chartqa_textaware_eval/text_index_tesseract`：基于 tesseract OCR 的原始 text index。

### 检索评估（100 题）

| 模式 | 权重 | R@1 | R@3 | R@5 | MRR | NDCG@5 | 输出文件 |
|---|---|---:|---:|---:|---:|---:|---|
| text, tesseract OCR | - | 0.1600 | 0.2600 | 0.4300 | 0.2478 | 0.2863 | `eval_retrieval_text_tesseract.json` |
| hybrid, summary-only | - | 0.1000 | 0.2500 | 0.3200 | 0.2079 | 0.2169 | `eval_retrieval_hybrid.json` |
| hybrid, summary+OCR | text_mode=summary_ocr | 0.1200 | 0.2400 | 0.3600 | 0.2194 | 0.2369 | `eval_retrieval_hybrid_tesseract.json` |
| hybrid, OCR-only | text_mode=ocr_only | 0.1700 | 0.2500 | 0.3400 | 0.2457 | 0.2527 | `eval_retrieval_hybrid_tesseract_ocr_only.json` |
| hybrid, OCR-only + lexical rerank | text_mode=ocr_only, lexical=0.2 | 0.2300 | 0.4100 | 0.4800 | 0.3353 | 0.3619 | `eval_retrieval_hybrid_tesseract_ocr_only_lex02.json` |
| visual | - | 0.0300 | 0.0500 | 0.0700 | 0.0624 | 0.0512 | `eval_retrieval_visual.json` |
| fusion, 默认等权 | hybrid=1.0, visual=1.0 | 0.0500 | 0.1100 | 0.2000 | 0.1196 | 0.1205 | `eval_retrieval_fusion_hv.json` |
| fusion, 弱视觉权重 + top10候选 | hybrid=1.0, visual=0.02, candidates=10/10 | 0.1100 | 0.2600 | 0.3300 | 0.2154 | 0.2260 | `eval_retrieval_fusion_h1_v0.02_c10.json` |
| text-aware fusion, 旧推荐 | text=5.0, hybrid=0.1, visual=0.02, candidates=50/10/10 | 0.1700 | 0.3100 | 0.4300 | 0.2629 | 0.2952 | `eval_retrieval_fusion_textaware_t5_h01_v002_c10.json` |
| text-aware fusion, OCR-only hybrid | text=4.0, OCR-only hybrid=0.2, visual=0.02, candidates=50/10/10 | 0.2000 | 0.3200 | 0.4300 | 0.2837 | 0.3113 | `eval_retrieval_fusion_textaware_t4_h02_v002_ocr_only_hybrid_c10.json` |
| text-aware fusion, 新推荐 | text=1.0, lexical hybrid=5.0, visual=0.02, candidates=50/10/10 | 0.2500 | 0.3900 | 0.4800 | 0.3504 | 0.3723 | `eval_retrieval_fusion_textaware_t1_h5_v002_ocr_only_lex02_hybrid_c10.json` |
| text-aware fusion, hybrid 更强 | text=1.0, lexical hybrid=8.0, visual=0.02, candidates=50/10/10 | 0.2500 | 0.4000 | 0.4700 | 0.3509 | 0.3691 | `eval_retrieval_fusion_textaware_t1_h8_v002_ocr_only_lex02_hybrid_c10.json` |
| text-aware fusion, R@5/NDCG 倾向 | text=8.0, hybrid=0.1, visual=0.02, candidates=50/10/10 | 0.1700 | 0.2700 | 0.4400 | 0.2598 | 0.2963 | `eval_retrieval_fusion_textaware_t8_h01_v002_c10.json` |
| fusion, 弱视觉权重 + 默认top40候选 | hybrid=1.0, visual=0.02, candidates=40/40 | 0.0900 | 0.2500 | 0.3100 | 0.2034 | 0.2098 | `eval_retrieval_fusion_h1_v0.02.json` |

结论：ChartQA 这个小子集上，原始 `text` 基线在 tesseract OCR 后明显强于 summary-only `hybrid` 与旧版 `fusion(hybrid+visual)`；说明图表中的标题、图例、坐标轴、数值文本仍是强信号。优化后，hybrid 使用 `ocr_only + lexical_weight=0.2`，R@1=0.2300、MRR=0.3353、NDCG@5=0.3619，已经超过原始 text。最终检索层推荐 hybrid-heavy text-aware fusion：`text=1.0, hybrid=5.0, visual=0.02, candidates=50/10/10`，R@1=0.2500、MRR=0.3504、NDCG@5=0.3723。按 R@1/MRR/NDCG@5 看，已形成 `fusion > hybrid > text`。

注意：PaddleOCR 在当前环境仍不可用，本轮 ChartQA text 使用 `tesseract 5.5.2` 生成真实 OCR。`eval_retrieval_fusion_h1_v0.02_offline.json` 仍保留为离线 top-10 RRF 扫描记录，但主结果以在线评估的 `eval_retrieval_text_tesseract.json`、`eval_retrieval_hybrid_tesseract_ocr_only_lex02.json` 和 `eval_retrieval_fusion_textaware_t1_h5_v002_ocr_only_lex02_hybrid_c10.json` 为准。


### QA 小样本（前 10 题，Qwen3-VL-4B）

生成评估使用本地缓存 `Qwen/Qwen3-VL-4B-Instruct`，`max_new_tokens=128`。该表只作为生成阶段 sanity check；完整 ChartQA QA 还需要更大样本和 Relaxed Accuracy。

| 模式 | QA top-k | 配置 | EM | F1 | ANLS | CitationAcc | Avg Latency | 输出文件 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| hybrid | 3 | - | 0.2000 | 0.4200 | 0.3667 | 0.5000 | 1.84s | `eval_qa_hybrid_10.jsonl` |
| fusion-c10 | 3 | hybrid=1.0, visual=0.02, candidates=10/10 | 0.2000 | 0.3950 | 0.3417 | 0.4000 | 1.87s | `eval_qa_fusion_h1_v0.02_c10_10.jsonl` |
| text, tesseract OCR | 5 | - | 0.0000 | 0.0250 | 0.0000 | 0.4000 | 1.15s | `eval_qa_text_tesseract_top5_10.jsonl` |
| hybrid | 5 | - | 0.2000 | 0.4200 | 0.3667 | 0.4000 | 2.20s | `eval_qa_hybrid_top5_10.jsonl` |
| hybrid, OCR-only + lexical rerank | 5 | text_mode=ocr_only, lexical=0.2 | 0.3000 | 0.3900 | 0.3500 | 0.5000 | 2.32s | `eval_qa_hybrid_ocr_only_lex02_top5_10.jsonl` |
| fusion-c10 | 5 | hybrid=1.0, visual=0.02, candidates=10/10 | 0.2000 | 0.4450 | 0.3917 | 0.5000 | 2.17s | `eval_qa_fusion_h1_v0.02_c10_top5_10.jsonl` |
| text-aware fusion, OCR-only hybrid | 5 | text=4.0, hybrid=0.2, visual=0.02, candidates=50/10/10 | 0.2000 | 0.3522 | 0.3167 | 0.4000 | 2.85s | `eval_qa_fusion_textaware_t4_h02_v002_ocr_only_hybrid_top5_10.jsonl` |
| text-aware fusion, hybrid-heavy | 5 | text=1.0, hybrid=5.0, visual=0.02, candidates=50/10/10 | 0.1000 | 0.3427 | 0.3123 | 0.5000 | 2.43s | `eval_qa_fusion_textaware_t1_h5_v002_ocr_only_lex02_top5_10.jsonl` |

结论：在前 10 题 QA sanity check 中，原始 `text` 虽然检索强，但生成阶段只看到零散 OCR 片段，EM/ANLS 仍接近 0。lexical hybrid 的 EM 提升到 0.3000，说明页面级 OCR rerank 对生成输入也有帮助。但 hybrid-heavy fusion 虽然检索层最佳，QA 小样本没有超过 lexical hybrid，说明生成阶段还需要证据压缩或 rerank，避免 text chunk 与页面证据互相干扰。后续完整实验建议固定 `top-k=5`，并补充 ChartQA 常用的 Relaxed Accuracy。
