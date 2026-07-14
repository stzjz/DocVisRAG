# 文档地图

本目录只保留当前仍然活跃的项目文档。历史进展记录和重复实验结果已经合并或移入 `docs/archive/`。

## 推荐阅读顺序

1. [../README.md](../README.md)  
   项目动机、系统主线、当前主结果、UI 启动方式和常用实验指令。

2. [EXPERIMENTS.md](EXPERIMENTS.md)  
   实验结果的统一入口，包括公开数据集 sanity check、HomeworkQA、HomeworkQA-MC、消融实验和复现指令。

3. [TEXT_AWARE_FUSION.md](TEXT_AWARE_FUSION.md)  
   text-aware 检索、页面级融合、块级融合、visual-dominant fusion 和答案级 fusion 的方法说明。

4. [archive/SERVER_DEPLOYMENT.md](archive/SERVER_DEPLOYMENT.md)  
   服务器环境、远程运行和部署记录。该文档偏运行记录性质，因此放在 archive 中保留。

## 项目叙事

对外介绍项目时建议按下面主线展开：

```text
动机
  公开 VQA 数据集多为 question-image pair。
  DocVisRAG 关注多页文档集合上的 RAG 流程。

系统
  离线文档索引 + 在线检索 + VLM 基于证据页回答。

评估
  公开数据集用于 sanity check。
  HomeworkQA / HomeworkQA-MC 是主要评估集。

主结果
  HomeworkQA 简答题：
    hybrid-answer > text
    fusion-answer > visual

  HomeworkQA-MC 重平衡后：
    hybrid > text
    fusion > visual

应用
  面向企业/个人文档集合的离线索引式文档助手。
```

## 已合并或归档的文档

以下内容已经合并进 [EXPERIMENTS.md](EXPERIMENTS.md) 或移入 `docs/archive/`：

- `PROGRESS.md`
- `实验完成.md`
- `HOMEWORKQA_RESULTS.md`
- 旧版 server deployment / benchmark 运行记录

如果需要查旧版本，可以从 Git 历史或 `docs/archive/` 中恢复。
