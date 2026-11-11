# 使用手册 04 · `analytics` 工具箱

## 模块里有什么

| 符号 | 场景 |
| --- | --- |
| `StatsToolkit` | t / χ² / ANOVA |
| `ExperimentDesign` | 两比例检验、样本量、贝叶斯门槛 |
| `TimeSeriesTools` | 安全 lag / rolling、ADF、分解 |
| `CalibrationReport` / `expected_calibration_error` | 概率校准 |
| `GroupMetricsReport` | 分组 TPR/FPR |
| `AnomalyDetector` | IsolationForest 风格离群 |
| `RankingMetrics` | NDCG / HitRate / DCG |
| `SurvivalAnalyzer` | Kaplan–Meier |

## 案例拼盘（均可离线）

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import (
    StatsToolkit, ExperimentDesign, TimeSeriesTools,
    CalibrationReport, expected_calibration_error, GroupMetricsReport,
    AnomalyDetector, RankingMetrics, SurvivalAnalyzer,
)

ready(seed=42, quiet=True, show_env=False)
rng = np.random.default_rng(0)

# 1) 统计
a, b = rng.normal(0, 1, 60), rng.normal(0.3, 1, 60)
print(StatsToolkit.ttest_ind(a, b))
print(StatsToolkit.chi2_independence([[10, 5], [6, 12]]))
print(StatsToolkit.anova_oneway(a, b, rng.normal(0.1, 1, 60)))
print(ExperimentDesign.sample_size_two_proportion(0.1, 0.03))
print(ExperimentDesign.two_proportion_test(20, 200, 30, 200))

# 2) 时序
t = np.arange(56)
s = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
df = pd.DataFrame({"y": s})
print(TimeSeriesTools.safe_lag(df, "y", lags=(1, 2), assume_row_order=True).head(2))
print(TimeSeriesTools.safe_rolling_mean(df, "y", window=3, assume_row_order=True).tail(2))
print(TimeSeriesTools.adf_report(df["y"])["pvalue"])
print(TimeSeriesTools.decompose_summary(df["y"], period=7)["resid_std"])

# 3) 校准 + 公平
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
print(CalibrationReport.summarize(yte, proba))
print("ece", expected_calibration_error(yte, proba))
pred = (proba >= 0.5).astype(int)
sex = rng.choice(["F", "M"], size=len(yte))
print(GroupMetricsReport.audit(yte, pred, {"sex": sex}).head())

# 4) 异常 / 排序 / 生存
Xab = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(6, 1, (8, 3))])
print(AnomalyDetector(contamination=0.1, random_state=1).fit_predict_summary(Xab))
print(RankingMetrics.ndcg_at_k([1, 9, 2], {1, 2}, k=3))
surv = pd.DataFrame({"t": np.arange(1, 9), "e": [1, 0, 1, 1, 0, 1, 0, 1], "g": list("AAAABBBB")})
print(SurvivalAnalyzer.km(surv["t"], surv["e"])["median_survival"])
print(list(SurvivalAnalyzer.km_by_group(surv, "t", "e", "g")["groups"]))
```

配套脚本：[`examples/handbook/04_analytics_toolkit.py`](../../examples/handbook/04_analytics_toolkit.py)

## 使用建议

1. 统计结论写「效应量 + 适用条件」，不要只贴 p 值  
2. 时序特征只用 `safe_lag` / `safe_rolling_*`（须 `time_col` 或 `assume_row_order=True`），禁止未来函数  
3. 校准差时先别调业务阈值；公平指标要结合样本量解读

## 相关文档

- 短教程：04 / 08 / 09 / 14  
- 深课：E13–E16、E18、E22
