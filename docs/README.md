# Documentation Map

This directory keeps only the active project documents. Historical progress notes and duplicated experiment reports have been consolidated.

## Recommended Reading Order

1. [../README.md](../README.md)  
   Project motivation, main pipeline, current headline results, and quick-start commands.

2. [EXPERIMENTS.md](EXPERIMENTS.md)  
   The single source of truth for experiment results, including public sanity checks, HomeworkQA, HomeworkQA-MC, ablations, and reproduction commands.

3. [TEXT_AWARE_FUSION.md](TEXT_AWARE_FUSION.md)  
   Method note for text-aware retrieval, page/block fusion, visual-dominant fusion, and answer-level fusion.

4. [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md)  
   Server environment, deployment, and remote execution notes.

## Project Narrative

The project should be presented as:

```text
Motivation
  Public VQA datasets are mostly question-image pairs.
  DocVisRAG targets multi-page document RAG.

System
  Offline document indexing + online retrieval + VLM answering.

Benchmark
  Public datasets are sanity checks.
  HomeworkQA / HomeworkQA-MC are the main evaluation sets.

Main result
  HomeworkQA short-answer:
    hybrid-answer > text
    fusion-answer > visual

  HomeworkQA-MC after rebalance:
    hybrid > text
    fusion > visual

Application
  Offline-indexed enterprise/personal document assistants.
```

## Deprecated Documents

The following documents were merged into [EXPERIMENTS.md](EXPERIMENTS.md) and removed from the active docs set:

- `PROGRESS.md`
- `实验完成.md`
- `HOMEWORKQA_RESULTS.md`

If old versions are needed, recover them from git history.
