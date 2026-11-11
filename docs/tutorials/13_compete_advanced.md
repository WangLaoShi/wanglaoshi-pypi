# 13 · compete 进阶（集成 / 重要性 / 追踪 / 调参）

## 目标

覆盖 `ModelEnsemble`、`FeatureImportanceAnalyzer`、`ExperimentTracker`、`HyperparameterOptimizer`。

## 用到的门面 API

上述四个类及其方法：`voting` / `stacking` / `blending` · `calculate_importance` · `log_experiment` / `get_experiment_history` / `compare_experiments` · `bayesian_optimize` / `auto_tune`

## 数据集

合成二分类表（离线）。

## 完整示例

```python
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from wanglaoshi.runtime import ready
from wanglaoshi.compete import (
    ModelEnsemble,
    FeatureImportanceAnalyzer,
    ExperimentTracker,
    HyperparameterOptimizer,
)

ready(seed=42, quiet=True, show_env=False)
rng = np.random.default_rng(0)
X = pd.DataFrame({"f1": rng.normal(size=80), "f2": rng.normal(size=80), "f3": rng.integers(0, 3, 80)})
y = pd.Series((X["f1"] + 0.4 * X["f2"] > 0).astype(int))

models = [
    LogisticRegression(max_iter=400, random_state=42),
    DecisionTreeClassifier(max_depth=3, random_state=42),
]
ens = ModelEnsemble()
voter = ens.voting(models, X, y, voting="hard")
print("vote", voter.predict(X.iloc[:3]))

base = [
    LogisticRegression(max_iter=400, random_state=1),
    DecisionTreeClassifier(max_depth=2, random_state=1),
]
meta = LogisticRegression(max_iter=400, random_state=2)
stack_bundle = ens.stacking(base, meta, X, y, cv=3)
assert "meta_model" in stack_bundle

Xtr, Xva = X.iloc[:60], X.iloc[60:]
ytr, yva = y.iloc[:60], y.iloc[60:]
blender = ens.blending(
    [
        LogisticRegression(max_iter=400, random_state=3),
        DecisionTreeClassifier(max_depth=2, random_state=3),
    ],
    Xtr, ytr, Xva, yva,
)
print("blend", blender["predict"](X.iloc[:3]))

rf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)
imp = FeatureImportanceAnalyzer().calculate_importance(rf, X, y, method="builtin")
print(imp.head())

with tempfile.TemporaryDirectory() as td:
    tracker = ExperimentTracker(log_file=str(Path(td) / "experiments.json"))
    tracker.log_experiment(
        experiment_name="tut13",
        model_name="rf",
        features=list(X.columns),
        params={"n_estimators": 30},
        score=float(rf.score(X, y)),
        metrics={"acc": float(rf.score(X, y))},
    )
    hist = tracker.get_experiment_history()
    print("history", len(hist), tracker.compare_experiments())

# Optuna 可选：未安装时会 ImportError
opt = HyperparameterOptimizer()
try:
    best = opt.bayesian_optimize(
        LogisticRegression,
        {"C": {"type": "float", "low": 0.1, "high": 10.0, "log": True}},
        X, y, n_trials=3, cv=3,
    )
    print("best", best["best_params"])
    _ = opt.auto_tune  # 入口存在；完整调用同样依赖 optuna
except ImportError:
    print("skip HyperparameterOptimizer run: pip install optuna / wanglaoshi[compete]")
```

配套脚本：[`examples/tutorials/13_compete_advanced.py`](../../examples/tutorials/13_compete_advanced.py)

## 常见坑

- `blending` 需要独立验证集 `X_val` / `y_val`
- 贝叶斯调参依赖 **Optuna**（`pip install "wanglaoshi[compete]"`）

## 下一步

- analytics 其余 → [14_analytics_extra.md](14_analytics_extra.md)
