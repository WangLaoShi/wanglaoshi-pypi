```{note}
本页从仓库 `docs/tutorials/05_business_pipeline.md` 同步生成；请以仓库文件为编辑源。
```

# 05 · business 流水线（合成订单）

## 目标

15 分钟内：用合成订单 CSV 跑通 `BusinessPipeline`，看到各阶段状态与交付归档。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `BusinessPipeline` | `wanglaoshi.business` | D01–D12 总控 |
| `StageResult` / `RunRecord` | `business` | 阶段状态与整次运行记录 |
| `ready` | `runtime` | 环境就绪 |

## 数据集

合成订单表（`order_id` / `order_date` / `amount` / `channel` / `user_tier`）。  
与 `examples/business_intake_demo.ipynb` 同构，可离线。

## 完整示例

```python
from pathlib import Path
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.business import BusinessPipeline

ready(seed=42)
root = Path("outputs/tutorial_business")
root.mkdir(parents=True, exist_ok=True)
src = root / "orders.csv"
pd.DataFrame(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "order_date": [
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-04", "2024-01-05", "2024-01-06",
        ],
        "amount": [10.5, 20.0, 7.2, 80.0, 15.0, 12.0],
        "channel": ["web", "app", "web", "app", "web", "app"],
        "user_tier": ["A", "B", "A", "A", "B", "B"],
    }
).to_csv(src, index=False)

pipe = BusinessPipeline(
    root / "archive",
    required_columns=["order_id", "amount", "channel"],
    date_col="order_date",
    value_col="amount",
    dim_cols=["channel"],
    segment_cols=["user_tier"],
    period="2024-Q1",
)
record = pipe.run(src)
print(record.summary())
print("blocked?", record.blocked())
for s in record.stages:
    print(s.stage_id, s.status)
```

配套脚本：[`examples/tutorials/05_business_pipeline.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/05_business_pipeline.py)

## 常见坑

- 缺关键列应 `blocked`，不要静默瞎猜口径
- 结论必须带证据；无对照时 uplift 写 `pending`
- 归档目录是交付物，不只是 Notebook 内存结果

## 下一步

- ML 预处理 → [06_ml_preprocess.md](06_ml_preprocess.md)
- 业务专章 → `examples/business_pipeline_course.ipynb`
- 精讲：可追溯性 → `book/lessons/business_traceability.md`
