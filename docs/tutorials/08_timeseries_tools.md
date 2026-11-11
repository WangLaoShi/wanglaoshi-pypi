# 08 · 时序工具（合成季节序列）

## 目标

10 分钟内：用 `TimeSeriesTools` 做安全滞后、滚动均值、ADF 与季节分解摘要。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `TimeSeriesTools` | `analytics` | lag / rolling / ADF / 分解 |
| `ready` | `runtime` | 环境就绪 |

## 数据集

合成带趋势 + 周期=7 的序列（离线）。真实共享单车见 `examples/advanced/08_bike_sharing.ipynb`。

## 完整示例

```python
import numpy as np
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import TimeSeriesTools

ready(seed=42)
t = np.arange(56)
y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
df = pd.DataFrame({"y": y})

lagged = TimeSeriesTools.safe_lag(df, "y", lags=(1, 2))
rolled = TimeSeriesTools.safe_rolling_mean(lagged, "y", window=3)
adf = TimeSeriesTools.adf_report(df["y"])
dec = TimeSeriesTools.decompose_summary(df["y"], period=7)
print(lagged.head())
print(adf)
print({k: dec[k] for k in ("period", "resid_std") if k in dec})
```

配套脚本：[`examples/tutorials/08_timeseries_tools.py`](../../examples/tutorials/08_timeseries_tools.py)

## 常见坑

- `safe_lag` 禁止负滞后（前瞻）；不要手写 `shift(-1)` 当特征
- 滚动窗口只使用「当时可得」历史
- ADF 显著 ≠ 业务可预测；还要看季节与结构突变

## 下一步

- 校准与公平 → [09_calibration_fairness.md](09_calibration_fairness.md)
- 深课 → `examples/advanced/08_bike_sharing.ipynb`、`15_timeseries_forecast.ipynb`
