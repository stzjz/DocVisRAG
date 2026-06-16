# HomeworkQA 自建 Benchmark

本文记录 `data/bench_full/homeworkqa/` 自建 PDF benchmark 的数据来源、问题格式、评估口径和维护规则。

## 当前版本

- 数据来源：`data/bench_full/homeworkqa/pdf/` 下 10 个作业/课程 PDF。
- 问题规模：v3 主集共 160 题，已删除过于简单的元信息题，并追加图表/手写图/电路图等多模态题。
- 正式问题文件：`data/bench_full/homeworkqa/questions.jsonl`。
- 分 PDF 问题文件：`data/bench_full/homeworkqa/questions_by_pdf/*.jsonl`。
- 批量评估配置：`data/bench_full/homeworkqa/batch_config.json`。
- 选择题版本：`data/bench_full/homeworkqa_mc/questions.jsonl`，每题 4-6 个选项。

10 个文档：

| PDF | 题数 | 内容类型 |
|---|---:|---|
| `公益服务证明.pdf` | 20 | 公益服务证明、名单表格、日期/人员/时长 |
| `圆顶.pdf` | 20 | 建筑圆顶汇报、圣母百花大教堂穹顶建造 |
| `大学物理.pdf` | 20 | 电磁学手写题解、磁场公式推导 |
| `大数据.pdf` | 20 | Sobel 边缘检测、SVM/CNN 图像分类实验 |
| `并行计算.pdf` | 20 | CUDA 并行矩阵乘法、共享内存/寄存器优化 |
| `推荐系统.pdf` | 20 | CineMatch Studio 推荐系统界面与可视化 |
| `数字电路.pdf` | 20 | 数字电路复习笔记、组合/时序逻辑 |
| `数字电路2.pdf` | 20 | 手写逻辑表达式、卡诺图、逻辑门电路 |
| `数据结构.pdf` | 20 | 栈、括号匹配、中缀/后缀表达式实验 |
| `编译原理.pdf` | 20 | LR(1)、DFA、语义动作、语法制导定义 |

## 评估口径

> 默认配置更新：从 2026-06-16 起，HomeworkQA 专用评估脚本默认对 `visual` 和 `fusion` 启用优化版 visual metadata rerank：`--visual-rerank-mode metadata --visual-rerank-alpha 0.1`。`fusion` 默认 `--fusion-visual-weight 5.0`，避免 visual 分支被 hybrid-heavy 权重压住。历史结果表仍按当时运行配置记录。

HomeworkQA 按“完整 PDF 文档级输入”设计，不按单页图片直接问答。

- 每个 PDF 是一个独立 document。
- 每个问题的 `doc_path` 指向该 PDF 渲染后的完整 `manifest.json`。
- 评估时应先对完整 PDF 建索引，再由检索器找相关页给 QA 模型。
- `evidence_pages` 只作为检索命中评估标签，不应泄露给模型。
- 问题文本避免写成“第 N 页写了什么”，而是模拟用户对完整 PDF 提问。

推荐评估方式是“合并索引 + 按 PDF 过滤候选”：

- text / hybrid / visual 三类索引只构建和加载一次，避免 10 个 PDF 逐份评估时反复启动模型。
- 评估时根据每道题的 `doc_id` 和 `batch_config.json` 找到对应 PDF 的真实索引 `doc_id`。
- 检索候选先从合并索引取回，再过滤到该题所属完整 PDF 的页面集合。
- 这样既保持“模型输入/检索范围是一整份 PDF”，又不会把其他 PDF 的页面混入该题评估。

当前使用的合并索引目录：

- `data/bench_runs/homeworkqa_eval_20260615_combined/text_index/`
- `data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02/`
- `data/bench_runs/homeworkqa_eval_20260615_combined/visual_index/`

如果需要重建索引，可基于 `.work/combined/` 下的合并 manifest/OCR/summary 文件运行：

```bash
python scripts/retrieve/build_text_index.py \
  --ocr data/bench_full/homeworkqa/.work/combined/ocr.jsonl \
  --index-dir data/bench_runs/homeworkqa_eval_20260615_combined/text_index

python scripts/retrieve/build_hybrid_index.py \
  --manifest data/bench_full/homeworkqa/.work/combined/manifest.json \
  --ocr data/bench_full/homeworkqa/.work/combined/ocr.jsonl \
  --summaries data/bench_full/homeworkqa/.work/combined/page_summaries.jsonl \
  --index-dir data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02 \
  --text-mode ocr_only \
  --lexical-weight 0.2

python scripts/retrieve/build_visual_index.py \
  --manifest data/bench_full/homeworkqa/.work/combined/manifest.json \
  --index-dir data/bench_runs/homeworkqa_eval_20260615_combined/visual_index
```

全量检索评估命令：

```bash
python scripts/bench/eval_homeworkqa_combined.py \
  --questions data/bench_full/homeworkqa/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --text-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/text_index \
  --hybrid-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02 \
  --visual-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/visual_index \
  --out data/bench_runs/homeworkqa_eval_20260615_combined/eval_retrieval_homeworkqa_combined.json \
  --top-k 5 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 0.05 \
  --fusion-text-candidates 0 \
  --fusion-hybrid-candidates 0 \
  --fusion-visual-candidates 0
```

其中候选深度设为 `0` 表示在当前合并索引内尽量取全量候选，再按 PDF 过滤。HomeworkQA 当前只有 51 页，适合用这种方式避免全局排名漏掉目标 PDF 页面。

QA 小样本验证仍建议先加 `--qa-limit` 控制成本；QA 阶段也应保持每题只使用对应完整 PDF 的检索结果，不能把 `evidence_pages` 泄露给模型。

## 2026-06-15 全量检索结果

本轮跑完 HomeworkQA 全量 200 题，评估输出：

- `data/bench_runs/homeworkqa_eval_20260615_combined/eval_retrieval_homeworkqa_combined.json`

配置：

- 合并索引：10 份 PDF，共 51 页。
- 评估范围：每题按所属 PDF 过滤候选，保持 full-PDF scope。
- hybrid：`ocr_only + lexical_weight=0.2`。
- fusion：`text=0.2, hybrid=5.0, visual=0.05`。
- 候选深度：text/hybrid/visual 均取全量候选后过滤。

| 模式 | N | R@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| text | 200 | 0.7800 | 0.9600 | 0.9950 | 0.8679 | 0.8984 |
| hybrid | 200 | 0.8050 | 0.9650 | 1.0000 | 0.8821 | 0.9101 |
| visual | 200 | 0.3650 | 0.6700 | 0.8600 | 0.5333 | 0.6113 |
| fusion | 200 | 0.8050 | 0.9650 | 1.0000 | 0.8821 | 0.9101 |

按文档看，fusion 与 hybrid 持平，明显强于 text 和 visual 单路。visual 单路在当前索引上偏弱，因此 fusion 采用 hybrid-heavy 权重；当前结果不能说明 visual 分支已经带来稳定增益，只能说明加入低权重 visual 不会破坏 hybrid 的结果。

| PDF | N | Fusion R@1 | Fusion R@3 | Fusion R@5 | Fusion MRR | Fusion NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| 公益服务证明 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 圆顶 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 大学物理 | 20 | 0.7500 | 1.0000 | 1.0000 | 0.8667 | 0.9012 |
| 大数据 | 20 | 0.6500 | 1.0000 | 1.0000 | 0.8000 | 0.8512 |
| 并行计算 | 20 | 0.9500 | 1.0000 | 1.0000 | 0.9750 | 0.9644 |
| 推荐系统 | 20 | 0.9500 | 1.0000 | 1.0000 | 0.9750 | 0.9815 |
| 数字电路 | 20 | 0.9000 | 1.0000 | 1.0000 | 0.9500 | 0.9631 |
| 数字电路2 | 20 | 0.6000 | 0.7500 | 1.0000 | 0.7292 | 0.7958 |
| 数据结构 | 20 | 0.4500 | 0.9000 | 1.0000 | 0.6333 | 0.7246 |
| 编译原理 | 20 | 0.8000 | 1.0000 | 1.0000 | 0.8917 | 0.9196 |

当前解读：

- R@5 达到 1.0000，说明多数问题的证据页能被召回，适合进入 QA 小样本验证。
- R@1 仍有优化空间，尤其是 `数据结构`、`数字电路2`、`大数据` 这类多页实验/手写公式/流程型 PDF。
- visual 单路偏弱，下一步更应该检查视觉索引质量、页面渲染分辨率和 query-image 对齐，而不是简单提高 visual 权重。


## v3 多模态题库更新

从 v3 起，HomeworkQA 不再只依赖文字/OCR 线索。主集在 v2 删除简单元信息题的基础上，追加了 23 道明确依赖 PDF 视觉内容的问题，当前共 160 题。新增题面会要求模型根据图、表、界面截图、手写示意图、电路图、卡诺图、真值表、DFA 图或曲线图回答。

数据文件：

- 主集：`data/bench_full/homeworkqa/questions.jsonl`，160 题。
- 选择题：`data/bench_full/homeworkqa_mc/questions.jsonl`，160 题。
- Hard：`data/bench_full/homeworkqa_hard/questions.jsonl`，100 题，由 v3 主集派生。
- Hard-MC：`data/bench_full/homeworkqa_hard_mc/questions.jsonl`，100 题。
- v1 备份：`data/bench_full/homeworkqa/archive/v1_200_easy_included_20260616/`。
- 被删除的简单题：`data/bench_full/homeworkqa/removed_simple_questions_20260616.jsonl`。

新增多模态覆盖：

| Modality | 题数 | 示例 |
|---|---:|---|
| table / truth table | 6 | 志愿者名单、编译表格、逻辑真值表 |
| diagram / system diagram | 5 | 圆顶结构图、CUDA 全局/共享内存示意图 |
| handwritten diagram | 2 | 物理柱面电流与坐标系手绘图 |
| line / pie / bar chart | 4 | Training Loss、Accuracy vs SVM、类型占比饼图、评分柱状图 |
| graph visualization / DFA | 2 | 推荐系统图传播视图、LR(1) DFA 图 |
| circuit / Karnaugh map | 3 | CMOS/逻辑电路图、卡诺图 |
| image result | 1 | Sobel 边缘检测结果图 |

2026-06-16 v3 检索层结果，默认启用 visual metadata rerank 与 fusion `visual_weight=5.0`：

| 模式 | N | R@1 | R@3 | R@5 | Coverage@3 | All Evidence@3 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 160 | 0.7750 | 0.9688 | 1.0000 | 0.9625 | 0.9500 | 0.8672 | 0.8987 |
| hybrid | 160 | 0.8063 | 0.9688 | 1.0000 | 0.9625 | 0.9500 | 0.8828 | 0.9103 |
| visual | 160 | 0.8313 | 0.9813 | 1.0000 | 0.9734 | 0.9625 | 0.9036 | 0.9267 |
| fusion | 160 | 0.8375 | 0.9938 | 1.0000 | 0.9859 | 0.9750 | 0.9099 | 0.9307 |

v3 text 短答 QA + Qwen2.5-7B judge 结果：

| 模式 | N | Judge Acc | Judge Score | EM | F1 | ANLS | Relaxed Acc | Retrieval R@1 | Retrieval R@3 | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 160 | 0.7313 | 0.7394 | 0.2938 | 0.6842 | 0.5305 | 0.4563 | 0.7750 | 0.9688 | 0.9000 |

输出文件：

- `data/bench_runs/homeworkqa_v3_multimodal_20260616/qa_predictions_text.jsonl`
- `data/bench_runs/homeworkqa_v3_multimodal_20260616/qa_summary_text.json`
- `data/bench_runs/homeworkqa_v3_multimodal_20260616/judge_text_qwen25_7b_summary.json`

解读：追加多模态题后，优化版 visual 和 fusion 不再只是“跟随 hybrid”。fusion 在 R@1、R@3、证据覆盖和 NDCG@5 上均高于 hybrid，更符合我们希望验证的 PDF 多模态 RAG 场景。text 的语义 Judge Acc 为 0.7313，说明纯 text 检索仍有较强端到端答题能力，但在检索层已经低于优化 visual/fusion。

## HomeworkQA-Hard 多证据版本

为了提高区分度，新增 `data/bench_full/homeworkqa_hard/`。它不覆盖普通版 HomeworkQA，而是从已核对的 200 道短答题中按 PDF 内部两两组合，生成更难的多证据问题。

当前规模：

- 10 份 PDF，每份 10 题，共 100 题。
- 每题要求同时回答两个相关子问题。
- 优先组合不同页、不同题型的源问题，并均衡使用源题，避免一个简单事实题反复出现。
- 字段 `source_question_ids` 记录来源题，`evidence_pages` 是两个来源题证据页的并集。
- 对应选择题版本：`data/bench_full/homeworkqa_hard_mc/`，每题 4-6 个选项。

生成命令：

```bash
python scripts/bench/make_homeworkqa_hard.py \
  --questions data/bench_full/homeworkqa/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --out-dir data/bench_full/homeworkqa_hard \
  --questions-per-pdf 10 \
  --seed 20260616

python scripts/bench/make_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa_hard/questions.jsonl \
  --batch-config data/bench_full/homeworkqa_hard/batch_config.json \
  --out-dir data/bench_full/homeworkqa_hard_mc \
  --min-choices 4 \
  --max-choices 6 \
  --seed 20260616
```

检索评估命令：

```bash
python scripts/bench/eval_homeworkqa_combined.py \
  --questions data/bench_full/homeworkqa_hard/questions.jsonl \
  --batch-config data/bench_full/homeworkqa_hard/batch_config.json \
  --text-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/text_index \
  --hybrid-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02 \
  --visual-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/visual_index \
  --out data/bench_runs/homeworkqa_hard_20260616/eval_retrieval_homeworkqa_hard_combined.json \
  --top-k 5 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 5.0 \
  --fusion-text-candidates 0 \
  --fusion-hybrid-candidates 0 \
  --fusion-visual-candidates 0
```

多证据题不能只看普通 R@k，因为命中任意一个证据页就会让 R@k 变高。Hard 版新增并重点关注：

- `evidence_coverage@k`：top-k 中覆盖了多少比例的证据页。
- `all_evidence@k`：top-k 是否覆盖全部证据页。

2026-06-16 检索层结果：

| 模式 | N | R@1 | R@3 | R@5 | Coverage@3 | Coverage@5 | All Evidence@3 | All Evidence@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 100 | 0.8400 | 0.9900 | 1.0000 | 0.8482 | 0.9730 | 0.6900 | 0.9400 |
| hybrid | 100 | 0.8700 | 0.9900 | 1.0000 | 0.8382 | 0.9780 | 0.6700 | 0.9500 |
| visual | 100 | 0.5600 | 0.8900 | 0.9900 | 0.6648 | 0.8827 | 0.4300 | 0.7700 |
| fusion | 100 | 0.8700 | 0.9900 | 1.0000 | 0.8382 | 0.9780 | 0.6700 | 0.9500 |

解读：Hard 版的普通 R@3 仍然很高，但 `All Evidence@3` 明显下降，说明它确实更考验多页证据覆盖；后续短答 QA 建议看 LLM Judge 是否能同时答对两个子问题，而不是只答对其中一个。

## HomeworkQA-MC 选择题版本

HomeworkQA-MC 是从短答版 HomeworkQA 派生的单选题版本，用来直接统计最终回答准确率。每题保留原始问题、标准答案和 `evidence_pages`，额外加入：

- `choices`：4-6 个选项，选项字母为 `A-F`。
- `answer_key`：正确选项字母。
- `source_question_id`：对应短答版题目 ID。

生成命令：

```bash
python scripts/bench/make_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --out-dir data/bench_full/homeworkqa_mc \
  --min-choices 4 \
  --max-choices 6 \
  --seed 20260615
```

全量选择题评估命令：

```bash
HF_HOME=/data1/home/zengjian/.cache/huggingface \
HF_HUB_CACHE=/data1/home/zengjian/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
DOCVISRAG_LOCAL_FILES_ONLY=1 \
CUDA_VISIBLE_DEVICES=3 \
python scripts/bench/eval_homeworkqa_mc.py \
  --questions data/bench_full/homeworkqa_mc/questions.jsonl \
  --batch-config data/bench_full/homeworkqa_mc/batch_config.json \
  --text-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/text_index \
  --hybrid-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02 \
  --visual-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/visual_index \
  --out data/bench_runs/homeworkqa_mc_20260615/predictions_hybrid_fusion.jsonl \
  --summary-out data/bench_runs/homeworkqa_mc_20260615/summary_hybrid_fusion.json \
  --modes hybrid fusion \
  --top-k 3 \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-new-tokens 8 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 5.0 \
  --fusion-text-candidates 0 \
  --fusion-hybrid-candidates 0 \
  --fusion-visual-candidates 0
```

本评估仍使用合并索引 + 按 PDF 过滤候选，模型只看到检索出的候选页图像、页面摘要和 OCR 文本；`evidence_pages` 不进入 prompt。

### 2026-06-15 全量 MC 结果

输出文件：

- `data/bench_runs/homeworkqa_mc_20260615/predictions_text_visual.jsonl`
- `data/bench_runs/homeworkqa_mc_20260615/summary_text_visual.json`
- `data/bench_runs/homeworkqa_mc_20260615/predictions_hybrid_fusion.jsonl`
- `data/bench_runs/homeworkqa_mc_20260615/summary_hybrid_fusion.json`

配置：Qwen/Qwen3-VL-4B-Instruct，200 题，`top_k=3`，fusion=`text=0.2, hybrid=5.0, visual=0.05`。

注意：本轮 MC 选项生成已收紧为优先同 PDF / 同题型答案池干扰项，不使用全局随机打乱后的远距离干扰项；因此比早期 smoke 版本更难，也更适合作为最终准确率参考。

| 模式 | N | Accuracy | Parse Rate | Retrieval R@1 | Retrieval R@3 | Retrieval MRR | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 200 | 0.9200 | 1.0000 | 0.7800 | 0.9600 | 0.8592 | 3.89s |
| visual | 200 | 0.8400 | 1.0000 | 0.3650 | 0.6700 | 0.4908 | 3.73s |
| hybrid | 200 | 0.9250 | 1.0000 | 0.8050 | 0.9650 | 0.8733 | 3.78s |
| fusion | 200 | 0.9350 | 1.0000 | 0.8050 | 0.9650 | 0.8733 | 3.91s |

当前排序：`fusion > hybrid > text > visual`。fusion 比 hybrid 高 1 个百分点，但页级检索 R@1/R@3/MRR 与 hybrid 完全一致，因此应表述为“fusion 在 MC 最终准确率上小幅优于 hybrid”，不能说页级检索已经显著优于 hybrid。

按 PDF 看，主要错误集中在手写/公式/语法分析类 PDF：

| PDF | Hybrid Acc | Fusion Acc | 备注 |
|---|---:|---:|---|
| 数字电路2 | 0.5000 | 0.6000 | 手写公式、逻辑表达式和电路图干扰项最难 |
| 编译原理 | 0.8500 | 0.9000 | 表格列、语法分析概念题有误 |
| 数据结构 | 0.9500 | 0.9000 | 中缀/后缀表达式与实验题目题有误 |
| 大学物理 | 0.9500 | 0.9500 | 磁通量数量级题有误 |
| 其他 6 个 PDF | 1.0000 | 1.0000 | 本轮均答对 |

当前结论：视觉单路在 MC 上只有 0.8400，主要受 visual 检索 R@1=0.3650 影响；text 单路已经达到 0.9200，说明 HomeworkQA-MC 很多题仍依赖 OCR/文本证据。fusion 的小幅收益更可能来自候选证据内容和 VLM 判别差异，而不是页级检索指标提升。

## HomeworkQA 短答版 + LLM Judge

短答版 HomeworkQA 保留开放式答案，更接近真实 PDF-RAG 问答，但字符串指标会低估中文短答和解释型答案。因此本轮同时报告 EM/F1/ANLS/relaxed accuracy 和 7B LLM 语义 judge accuracy。

全量短答 QA 命令：

```bash
HF_HOME=/data1/home/zengjian/.cache/huggingface \
HF_HUB_CACHE=/data1/home/zengjian/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
DOCVISRAG_LOCAL_FILES_ONLY=1 \
CUDA_VISIBLE_DEVICES=3 \
python scripts/bench/eval_homeworkqa_qa.py \
  --questions data/bench_full/homeworkqa/questions.jsonl \
  --batch-config data/bench_full/homeworkqa/batch_config.json \
  --text-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/text_index \
  --hybrid-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/hybrid_index_ocr_only_lex02 \
  --visual-index-dir data/bench_runs/homeworkqa_eval_20260615_combined/visual_index \
  --out data/bench_runs/homeworkqa_qa_20260616/predictions_fusion.jsonl \
  --summary-out data/bench_runs/homeworkqa_qa_20260616/summary_fusion.json \
  --mode fusion \
  --top-k 3 \
  --model-id Qwen/Qwen3-VL-4B-Instruct \
  --max-new-tokens 96 \
  --fusion-text-weight 0.2 \
  --fusion-hybrid-weight 5.0 \
  --fusion-visual-weight 5.0 \
  --fusion-text-candidates 0 \
  --fusion-hybrid-candidates 0 \
  --fusion-visual-candidates 0
```

LLM judge 命令：

```bash
HF_HOME=/data1/home/zengjian/.cache/huggingface \
HF_HUB_CACHE=/data1/home/zengjian/.cache/huggingface/hub \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=3 \
python scripts/bench/judge_homeworkqa_qa.py \
  --predictions data/bench_runs/homeworkqa_qa_20260616/predictions_fusion.jsonl \
  --out data/bench_runs/homeworkqa_qa_20260616/judge_fusion_qwen25_7b.jsonl \
  --summary-out data/bench_runs/homeworkqa_qa_20260616/judge_fusion_qwen25_7b_summary.json \
  --judge-model /data1/home/zengjian/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --max-new-tokens 128
```

Judge 模型：`Qwen/Qwen2.5-7B-Instruct`。选择理由：纯文本指令模型、中文能力稳定、Transformers/vLLM 支持成熟，适合判定预测答案与标准答案是否语义一致。

- HuggingFace repo：`Qwen/Qwen2.5-7B-Instruct`
- 本地 cache：`/data1/home/zengjian/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`
- snapshot：`/data1/home/zengjian/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`
- 模型缓存占用：约 15G；下载后 `/data1` 剩余约 33G。

### 2026-06-16 全量短答结果

输出文件：

- `data/bench_runs/homeworkqa_qa_20260616/predictions_fusion.jsonl`
- `data/bench_runs/homeworkqa_qa_20260616/predictions_hybrid.jsonl`
- `data/bench_runs/homeworkqa_qa_20260616/predictions_text.jsonl`
- `data/bench_runs/homeworkqa_qa_20260616/predictions_visual.jsonl`
- `data/bench_runs/homeworkqa_qa_20260616/judge_fusion_qwen25_7b_summary.json`
- `data/bench_runs/homeworkqa_qa_20260616/judge_hybrid_qwen25_7b_summary.json`
- `data/bench_runs/homeworkqa_qa_20260616/judge_text_qwen25_7b_summary.json`
- `data/bench_runs/homeworkqa_qa_20260616/judge_visual_qwen25_7b_summary.json`

配置：Qwen/Qwen3-VL-4B-Instruct 作答，Qwen/Qwen2.5-7B-Instruct 作 judge，200 题，`top_k=3`。fusion=`text=0.2, hybrid=5.0, visual=0.05`。

| 模式 | N | Judge Acc | Judge Score | EM | F1 | ANLS | Relaxed Acc | Retrieval R@1 | Retrieval R@3 | Citation Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 200 | 0.7800 | 0.7860 | 0.3800 | 0.7012 | 0.5884 | 0.5500 | 0.8050 | 0.9650 | 0.9000 |
| fusion | 200 | 0.7700 | 0.7765 | 0.3550 | 0.6916 | 0.5825 | 0.5350 | 0.8050 | 0.9650 | 0.9050 |
| text | 200 | 0.7700 | 0.7755 | 0.3700 | 0.7026 | 0.5879 | 0.5300 | 0.7800 | 0.9600 | 0.8650 |
| visual | 200 | 0.5550 | 0.5630 | 0.2850 | 0.5549 | 0.4455 | 0.3850 | 0.3650 | 0.6700 | 0.6450 |

按 PDF 的 fusion judge accuracy：

| PDF | N | Judge Acc | EM | F1 | ANLS | Retrieval R@1 | Retrieval R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 推荐系统 | 20 | 1.0000 | 0.5000 | 0.7692 | 0.8889 | 0.9500 | 1.0000 |
| 圆顶 | 20 | 0.9500 | 0.6500 | 0.8349 | 0.7823 | 1.0000 | 1.0000 |
| 大数据 | 20 | 0.9500 | 0.4500 | 0.7731 | 0.6175 | 0.6500 | 1.0000 |
| 数字电路 | 20 | 0.9500 | 0.2500 | 0.6929 | 0.4919 | 0.9000 | 1.0000 |
| 公益服务证明 | 20 | 0.9000 | 0.5000 | 0.6972 | 0.5969 | 1.0000 | 1.0000 |
| 数据结构 | 20 | 0.8000 | 0.4500 | 0.7541 | 0.6849 | 0.4500 | 0.9000 |
| 并行计算 | 20 | 0.6500 | 0.2000 | 0.6831 | 0.5528 | 0.9500 | 1.0000 |
| 大学物理 | 20 | 0.6000 | 0.1000 | 0.5690 | 0.4622 | 0.7500 | 1.0000 |
| 编译原理 | 20 | 0.5500 | 0.3500 | 0.6235 | 0.4133 | 0.8000 | 1.0000 |
| 数字电路2 | 20 | 0.3500 | 0.1000 | 0.5188 | 0.3342 | 0.6000 | 0.7500 |

当前解读：

- 短答语义准确率排序为 `hybrid > fusion = text > visual`。hybrid 为 `0.7800`，fusion 和 text 都是 `0.7700`，visual 只有 `0.5550`。
- fusion 没有超过 hybrid，原因和页级检索一致：当前 visual 分支太弱，fusion 的检索 R@1/R@3 与 hybrid 持平，而低权重 visual 没能提供稳定增益。
- text 的 Judge Acc 与 fusion 持平，说明 HomeworkQA 里大量题仍由 OCR/文本证据主导；但 text 的 citation accuracy 和 R@1 低于 hybrid。
- visual 的 R@1 只有 `0.3650`，R@3 只有 `0.6700`，导致 QA 与 judge 均显著落后；这基本验证了当前视觉索引质量/图文对齐是短板。
- 短答语义准确率明显高于 EM，例如 hybrid 的 Judge Acc `0.7800` 对 EM `0.3800`，开放式中文答案不能只看 exact match。
- 检索 R@3 到 `0.9650` 后，judge accuracy 仍只有 `0.7800` 左右，说明剩余错误还包括多页证据综合、公式/手写内容理解、答案格式抽取等 QA 问题。
- `数字电路2`、`编译原理`、`大学物理` 是当前短答最薄弱的三类；它们更依赖手写公式、逻辑表达式、语法分析表或推导细节。
- MC 主指标仍直接看 Accuracy；短答主指标建议看 `Judge Acc`，同时附带 EM/F1/ANLS 作为可复现的字符串指标。


## Visual 索引调优记录

当前纯 visual 索引使用 `vidore/colqwen2-v1.0`/Byaldi。对 HomeworkQA 这类中文、公式、OCR 密集的作业 PDF，纯视觉检索偏弱：R@1=`0.3650`，R@3=`0.6700`，R@5=`0.8600`。简单 query rewrite 基本无收益，最好的一档只到 R@3=`0.6800`。

本轮加入 `visual metadata rerank`：先用 visual index 产生候选页，再在候选页内部用 hybrid 索引里的 `summary`、`ocr_text_preview`、`search_text` 做轻量重排。参数 `--visual-rerank-alpha` 表示 visual 原始分数权重，`0.1` 代表 10% visual 分数 + 90% metadata 分数。注意这不是纯视觉索引结果，而是面向 PDF-RAG 的实用检索优化。

检索层结果如下，均为 HomeworkQA 200 题、`top_k=5`、候选深度取全量页后按 PDF 过滤：

| 配置 | R@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| pure visual | 0.3650 | 0.6700 | 0.8600 | 0.5333 | 0.6113 |
| visual + metadata rerank, alpha=0.1 | 0.8300 | 0.9800 | 0.9950 | 0.8996 | 0.9227 |
| hybrid baseline | 0.8050 | 0.9650 | 1.0000 | 0.8821 | 0.9101 |
| fusion old, h=5.0/v=0.05 | 0.8050 | 0.9650 | 1.0000 | 0.8821 | 0.9101 |
| fusion rerank, h=5.0/v=5.0/alpha=0.1 | 0.8200 | 0.9900 | 1.0000 | 0.8983 | 0.9225 |

推荐消融顺序：

1. 纯 visual：`--visual-rerank-mode none`，作为下限。
2. visual rerank：`--visual-rerank-mode metadata --visual-rerank-alpha 0.1`，观察 visual 候选是否能被 OCR/summary 拉正。
3. fusion rerank：HomeworkQA 脚本现在默认使用 `--fusion-visual-weight 5.0`；如需复现实验旧结果，可显式改回 `0.05`。

复现命令见仓库根目录 `ins` 的 “HomeworkQA visual 索引调优” 段。

## 问题格式

`questions.jsonl` 每行一个 JSON object，主要字段如下：

```json
{
  "id": "homeworkqa_数据结构_001",
  "benchmark": "homeworkqa",
  "doc_id": "数据结构",
  "doc_path": "/abs/path/to/rendered/数据结构/manifest.json",
  "source_pdf": "/abs/path/to/pdf/数据结构.pdf",
  "question": "数据结构实验报告的课程名称是什么？",
  "answer": "Data structures and algorithms",
  "evidence_pages": [1],
  "type": "fact",
  "input_scope": "full_pdf",
  "question_generation": "manual_from_pdf_text_and_vlm_page_summaries"
}
```

字段说明：

- `doc_path`：完整 PDF 的 manifest；评估入口使用它，而不是单页图片。
- `source_pdf`：原始 PDF 路径，便于追溯。
- `evidence_pages`：1-based 页码，用于检索指标。
- `type`：问题类型，用于后续分桶分析。
- `input_scope=full_pdf`：明确问题面向完整 PDF。

## QA 多样性

每个 PDF 的 20 题尽量覆盖多类问题，而不是只做标题/日期查询。当前包含的问题类型包括：

- `fact`：事实查询，如课程名、作者、日期、数值。
- `entity`：人名、机构、模型、算法、工具名。
- `table` / `chart`：表格和图表读取。
- `formula` / `calculation`：公式、数值计算、推导结果。
- `procedure`：步骤、流程、算法执行过程。
- `concept`：概念解释。
- `comparison` / `judgement`：比较和判断。
- `visual`：手写图、逻辑图、物理图示等视觉信息。
- `summary`：整页或整份 PDF 的概括性问题。

后续追加问题时，每份 PDF 建议继续保持 20 题或按 20 的倍数扩展，并记录题型分布，避免单一事实题过多。

## 中间产物

以下目录是生成 benchmark 时的中间产物，已隐藏到 `.work/` 下，先保留，便于后续继续追加或修订问题：

- `data/bench_full/homeworkqa/.work/rendered/`：每个 PDF 渲染后的页面图和 manifest。
- `data/bench_full/homeworkqa/.work/text/`：`pdftotext` 抽取文本。
- `data/bench_full/homeworkqa/.work/homeworkqa_context.md`：文本和 VLM 页面摘要汇总。

这些目录不是主要阅读入口，但当前不能删除；`questions.jsonl` 和 `batch_config.json` 已引用 `.work/rendered/*/manifest.json`。后续如果再次移动 `.work/rendered/`，必须同步重写：

- `questions.jsonl` 中所有 `doc_path`；
- `questions_by_pdf/*.jsonl` 中所有 `doc_path`；
- `batch_config.json` 中所有 `manifest`。

## 已知限制

- 部分扫描/手写 PDF 无法用 `pdftotext` 抽出文本，已使用 VLM 页面摘要辅助生成问题。
- OCR 环境当前存在问题：PaddleOCR 子进程失败，fallback 到 pytesseract 时找不到 tesseract binary。
- 部分表格/手写题的精细字段来自 VLM 摘要，后续可人工复核并补充更细的答案。
- 当前是第一版自建 benchmark，适合作为 PDF-RAG 检索/fusion 的内部评估集，不宜直接作为公开标准数据集。

## 维护建议

1. 追加 PDF 后，先渲染成 per-PDF manifest。
2. 对可抽文本 PDF 用 `pdftotext` 建上下文。
3. 对扫描/手写 PDF 用 VLM 页面摘要补充上下文。
4. 每份 PDF 生成或维护 20 题，确保题型多样。
5. 题目不泄露页码，`evidence_pages` 仅作评估标签。
6. 更新 `questions.jsonl`、`questions_by_pdf/` 和 `batch_config.json`。
