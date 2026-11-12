# 练习：E21 三表对账

对应课：{doc}`../cases/21_order_refund_reconciliation` · 覆盖矩阵：{doc}`../reference/analysis-coverage`

## 本课强制 API / 概念

| 项 | 你要能独立说清 |
| --- | --- |
| DataAnalyzer + pandas 对账 | 课内至少调用或对照一次 |
| 核心纪律 | 一对多连接防金额翻倍；销售额/退款与源表对平 |

## 自测

1. 本课补上了覆盖矩阵（E01–E22）里的哪一格？
2. 若跳过「对账 / 同一候选集」约束，结论会怎样被高估？

## 动手

打开 `examples/advanced/21_order_refund_reconciliation.ipynb`，确认 `outputs/21_order_refund_reconciliation/` 摘要文件能写出。
