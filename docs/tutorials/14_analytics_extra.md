# 14 · analytics 其余（异常 / 排序 / 生存）

## 目标

补齐 `AnomalyDetector`、`RankingMetrics`、`SurvivalAnalyzer`。

## 用到的门面 API

上述三类及 `fit_predict_summary` / `ndcg_at_k` / `hit_rate_at_k` / `dcg_at_k` / `km` / `km_by_group`

## 数据集

合成特征、排序列表、删失时间。

## 完整示例

```python
import numpy as np
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import AnomalyDetector, RankingMetrics, SurvivalAnalyzer

ready(seed=42, quiet=True, show_env=False)
rng = np.random.default_rng(1)
X = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(5, 1, (10, 3))])
y_inj = np.array([0] * 80 + [1] * 10)

det = AnomalyDetector(contamination=10 / 90, random_state=1)
summary = det.fit_predict_summary(X, y_injected=y_inj)
print(summary)
_ = det.score_samples(X[:5])
_ = det.predict(X[:5])

assert RankingMetrics.ndcg_at_k([1, 2, 3], {1}, k=3) == 1.0
assert RankingMetrics.hit_rate_at_k([9, 1], {1}, k=2) == 1.0
print("dcg", RankingMetrics.dcg_at_k([1.0, 1.0, 0.0], k=3))

df_surv = pd.DataFrame(
    {
        "t": [1, 2, 3, 4, 5, 6, 7, 8],
        "e": [1, 0, 1, 1, 0, 1, 0, 1],
        "g": ["A", "A", "A", "A", "B", "B", "B", "B"],
    }
)
km = SurvivalAnalyzer.km(df_surv["t"], df_surv["e"], name="all")
print(km["median_survival"], km["event_rate"])
by = SurvivalAnalyzer.km_by_group(df_surv, "t", "e", "g")
print(list(by["groups"]))
```

配套脚本：[`examples/tutorials/14_analytics_extra.py`](../../examples/tutorials/14_analytics_extra.py)

## 常见坑

- 异常检测 `contamination` 应接近预期异常比例
- KM 中 `events=0` 表示删失，不是「没数据」

## 下一步

- ml 其余 → [15_ml_toolkit.md](15_ml_toolkit.md)
