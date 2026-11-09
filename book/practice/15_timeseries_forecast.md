# 练习：时序分解预测

对应课：{doc}`../cases/15_timeseries_forecast` · 覆盖矩阵：{doc}`../reference/analysis-coverage`

## 本课强制 API / 概念

| 项 | 你要能独立说清 |
| --- | --- |
| extract_datetime_features / detect_time_leakage | 课内至少调用或对照一次 |
| 核心纪律 | 分解、ADF、季节朴素；禁止 shuffle |

## 自测

1. 本课相对 E01–E12 补上了哪类「常见分析」缺口？
2. 若把本课方法错误套到竞赛提交基线，最可能踩什么坑？
3. 根据课内「包侧优化启示」，你会给 wanglaoshi 提哪一条 P0/P1 API？

## 动手

打开 `examples/advanced/15_timeseries_forecast.ipynb`，改一个超参或数据生成种子，确认 `outputs/15_timeseries_forecast/` 摘要文件仍能写出。
