# Text-Aware and Answer-Level Fusion

> Detailed metrics live in [EXPERIMENTS.md](EXPERIMENTS.md). This document only explains the method choices.

## Background

DocVisRAG has two kinds of signals:

- OCR/text signals are strong for names, dates, formulas, table text, and copied text.
- Visual/page signals are strong for layout, forms, tables, charts, handwritten content, and page-level context.

The early fusion problem was that page-level fusion could dilute strong OCR hits, while pure text could miss visual evidence. The current design keeps both but separates **retrieval fusion** from **answer-level fusion**.

## Retrieval Modes

| Mode | Unit | Signal | Use |
|---|---|---|---|
| text | OCR block/page/window | dense text embedding | lexical baseline and fine-grained evidence |
| hybrid | page or OCR block | OCR + page summaries + lexical rerank | general document retrieval |
| visual | page image | ColQwen/Byaldi visual embedding | visual/layout/table evidence |
| page fusion | page | weighted RRF over text/hybrid/visual | retrieval ablation |
| visual-dominant fusion | page | high visual weight | ablation for visual-heavy setting |
| snippet fusion | page + snippets | preserve visual pages and attach text snippets | ablation for evidence injection |

## Weighted RRF

The page-level fusion family uses weighted reciprocal rank fusion:

```text
score(page) = sum_i weight_i / (k + rank_i(page))
```

Typical settings used in HomeworkQA ablations:

```text
default page/block fusion:
  text=0.2, hybrid=5, visual=2

visual-dominant fusion:
  text=0, hybrid=0.5, visual=10
```

Empirically, these page-level variants are useful ablations but are not the best short-answer strategy.

## Answer-Level Fusion

For HomeworkQA short-answer, the main experiment uses answer-level fusion.

### hybrid-answer

```text
question
  -> text retriever + VLM reader -> text answer
  -> raw hybrid retriever + VLM reader -> hybrid answer
  -> local text LLM selector/synthesizer -> final hybrid-answer
```

This is used as the main `hybrid` result for short-answer HomeworkQA.

### fusion-answer

```text
question
  -> text answer
  -> raw hybrid answer
  -> visual answer
  -> raw page-fusion answer
  -> local text LLM selector/synthesizer -> final fusion-answer
```

This is used as the main `fusion` result for short-answer HomeworkQA.

The selector does not see the gold answer. It only sees the question and candidate answers, then chooses or synthesizes the final response.

## Why Answer-Level Fusion Works Better

Page-level fusion decides which evidence pages to show the reader. It can fail when:

- a visually correct page is ranked lower by OCR;
- OCR text is correct but lacks visual layout;
- multiple pages each contain partial evidence;
- the reader gives slightly different answers from different evidence routes.

Answer-level fusion lets each route produce its own answer first. The final selector can then compare semantic agreement across candidates. This keeps strong visual answers intact instead of weakening them by lowering visual retrieval weight.

## Current Recommendation

For HomeworkQA short-answer:

```text
main text:
  raw text reader

main hybrid:
  hybrid-answer = answer fusion(text, raw_hybrid)

main visual:
  raw visual reader

main fusion:
  fusion-answer = answer fusion(text, raw_hybrid, visual, raw_page_fusion)
```

For HomeworkQA-MC:

```text
main MC:
  mode-level text / hybrid / visual / fusion

future improvement:
  option-level verifier or MC answer-level fusion
```

## What Stays as Ablation

Keep these variants for ablation tables:

- page-level hybrid
- block-level hybrid
- page-level fusion
- block-level fusion
- visual-dominant fusion
- visual-snippet fusion
- block-snippet fusion

They explain why the main experiment uses answer-level fusion instead of only tuning retrieval weights.
