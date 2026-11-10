# 练习：E13 A/B 试验

对应课：{doc}`../cases/13_ab_testing` · 覆盖矩阵：{doc}`../reference/analysis-coverage`

## 本课强制 API / 概念

| 项 | 你要能独立说清 |
| --- | --- |
| ready / DataAnalyzer | 课内至少调用或对照一次 |
| 核心纪律 | 两比例检验、功效、效应量；包缺口 ExperimentDesign |

## 自测

1. 本课在 E01–E22 体系里补上了哪类「常见分析」缺口？
2. 若把本课方法错误套到竞赛提交基线，最可能踩什么坑？
3. 根据课内「包侧优化启示」，你会给 wanglaoshi 提哪一条 P0/P1 API？

## 动手

打开 `examples/advanced/13_ab_testing.ipynb`，改一个超参或数据生成种子，确认 `outputs/13_ab_testing/` 摘要文件仍能写出。
