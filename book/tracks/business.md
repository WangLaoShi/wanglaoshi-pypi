# 主线：业务流水线

`wanglaoshi.business` 把分析 Skill 固化成可追溯阶段：每个阶段返回 `StageResult`（`passed` / `warning` / `blocked` / `skipped`）。

## 最小用法

```python
from wanglaoshi import BusinessPipeline

pipe = BusinessPipeline(
    "archive/",
    required_columns=["order_id", "amount"],
    date_col="order_date",
    value_col="amount",
    dim_cols=["channel"],
    period="2024-Q1",
)
record = pipe.run("orders.csv")
print(record.summary())
```

## 阶段心智模型

| 阶段 | 做什么 |
| --- | --- |
| D01 接收 | 归档 + sha256 台账 |
| D02–D03 | 列名规整、可配置清洗（带影响日志） |
| D12 EDA | 画像 + 统计方法适用性提示 |
| D04–D06 | 衍生、四方向计划、指标与占比校验 |
| D07–D08 | 六视角分析、带元数据的图表归档 |
| D09–D10 | 结论/建议（收益无依据→`pending`）、报告与交付索引 |

上游 `blocked` 时，正式交付应为不可交付（`deliverable=False`）。

## 映射表

见 {doc}`../reference/skill-mapping`。

## 本地 Notebook

仓库：`examples/business_intake_demo.ipynb`（合成订单，可离线）。
