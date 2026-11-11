# 使用手册 06 · `business` 业务分析

## 两种用法

1. **总控**：`BusinessPipeline(...).run(path)` —— 适合端到端交付  
2. **阶段函数**：`intake_file` / `clean_dataframe` / `calculate_metrics` … —— 适合课堂拆解与单测

状态一律看 `StageResult.status ∈ {passed, warning, blocked, skipped}`。

## 案例：合成订单 + 退款对账

```python
from pathlib import Path
import tempfile
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.business import (
    BusinessPipeline, StageResult, RunRecord,
    intake_file, structure_dataframe, structure_file, clean_dataframe,
    run_eda_profile, STAT_METHOD_CATALOG,
    derive_metrics, add_value_tiers, build_analysis_plan,
    calculate_metrics, DEFAULT_METRIC_DICT, run_multidimensional_analysis,
    archive_charts, generate_insights, archive_delivery,
    check_join_inflation, reconcile_to_source,
    complete_period_index, yoy_with_rules, yoy_stage,
    convert_units, DEFAULT_FACTORS,
    validate_metric_dict, default_metric_registry,
)

ready(seed=42, quiet=True, show_env=False)
orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6],
    "order_date": pd.date_range("2024-01-01", periods=6, freq="MS"),
    "amount": [10.0, 20.0, 7.0, 80.0, 15.0, 12.0],
    "channel": ["web", "app", "web", "app", "web", "app"],
    "user_tier": ["A", "B", "A", "A", "B", "B"],
})
refunds = pd.DataFrame({"order_id": [2, 2, 4], "refund": [5.0, 3.0, 10.0]})

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    src = td / "orders.csv"
    orders.to_csv(src, index=False)

    # A. 总控
    pipe = BusinessPipeline(
        td / "archive_pipe",
        required_columns=["order_id", "amount", "channel"],
        date_col="order_date", value_col="amount",
        dim_cols=["channel"], segment_cols=["user_tier"], period="2024",
    )
    record = pipe.run(src)
    print(record.summary())

    # B. 阶段拆解（同一份订单）
    assert isinstance(intake_file(src, td / "arch", required_columns=["order_id", "amount"]), StageResult)
    structure_file(src, td / "structured.csv")
    structure_dataframe(orders)
    clean_dataframe(orders, drop_duplicates=True, subset=["order_id"])
    run_eda_profile(orders, field_roles={"segment_cols": ["channel"]})
    assert STAT_METHOD_CATALOG
    derive_metrics(orders, date_col="order_date", value_col="amount", add_tiers=True)
    add_value_tiers(orders, "amount")
    plan = build_analysis_plan(orders, date_col="order_date", value_col="amount", dim_cols=["channel"])
    met = calculate_metrics(orders, metric_dict=DEFAULT_METRIC_DICT, group_cols=["channel"], share_of="gmv")
    an = run_multidimensional_analysis(orders, date_col="order_date", value_col="amount", dim_col="channel", segment_col="user_tier")
    archive_charts(orders, td / "charts", date_col="order_date", value_col="amount", dim_col="channel")
    ins = generate_insights(metrics_data=met.data, analysis_data=an.data, planning_data=plan.data)
    rr = RunRecord(run_id="hb06")
    rr.add_stage(met); rr.add_stage(ins)
    archive_delivery(rr, td / "delivery", insights_data=ins.data, metrics_data=met.data)

    print(check_join_inflation(orders, refunds, on="order_id", measure_col="amount").status)
    print(reconcile_to_source(orders[["order_id", "amount"]], orders[["order_id", "amount"]], key="order_id", value_col="amount").status)
    print(len(complete_period_index(orders, "order_date", value_col="amount")))
    print(yoy_with_rules(orders, "order_date", "amount").head())
    print(yoy_stage(orders, "order_date", "amount").status)
    print(convert_units(orders, "amount", from_unit="yuan", to_unit="wan", factors=DEFAULT_FACTORS).status)
    print(validate_metric_dict(DEFAULT_METRIC_DICT, version="v1").status)
    print(default_metric_registry().status)
```

配套脚本：[`examples/handbook/06_business_guide.py`](../../examples/handbook/06_business_guide.py)

## 使用建议

1. 缺关键列应 `blocked`，不要静默补口径  
2. 一对多 join 先 `check_join_inflation`  
3. 同比先 `complete_period_index` 再 `yoy_with_rules`  
4. 结论必须能指回指标底稿 / 图表元数据

## 相关文档

- 短教程：05 / 16  
- 专章：`examples/business_pipeline_course.ipynb`  
- 精讲：`book/lessons/business_traceability.md`
