# 使用手册 03 · `compete` 工具箱

## 模块里有什么

| 类 | 职责 |
| --- | --- |
| `QuickBaseline` | 多模型快训快比 |
| `SubmissionGenerator` | 按模板生成 / 校验 / 融合提交 |
| `LeakageDetector` | 目标 / 时间 / 分布泄漏提示 |
| `CrossValidator` | 分层 / 分组 / 时序 CV |
| `ModelEnsemble` | voting / stacking / blending |
| `FeatureImportanceAnalyzer` | 内置 / 排列重要性 |
| `ExperimentTracker` | 实验 JSON 台账 |
| `HyperparameterOptimizer` | Optuna 调参（可选依赖） |

```python
from wanglaoshi.compete import (
    QuickBaseline, SubmissionGenerator, LeakageDetector, CrossValidator,
    ModelEnsemble, FeatureImportanceAnalyzer, ExperimentTracker,
)
```

## 案例：Titanic 一条龙

**数据：** `seaborn.load_dataset("titanic")`（去掉泄漏列 `alive`）。

```python
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from wanglaoshi.runtime import ready
from wanglaoshi.compete import (
    QuickBaseline, SubmissionGenerator, LeakageDetector, CrossValidator,
    ModelEnsemble, FeatureImportanceAnalyzer, ExperimentTracker,
)

ready(seed=42, quiet=True, show_env=False)
raw = sns.load_dataset("titanic").copy()
y = raw["survived"]

# 1) 泄漏提示：故意保留 alive
leaky_X = pd.get_dummies(
    raw.drop(columns=["survived", "deck", "embark_town", "class", "who", "adult_male"], errors="ignore"),
    drop_first=True,
).fillna(0)
hits = LeakageDetector().detect_target_leakage(leaky_X, y, threshold=0.85)
print("leak hits", hits)

# 2) 清洗特征
X = pd.get_dummies(
    raw.drop(
        columns=["survived", "alive", "deck", "embark_town", "class", "who", "adult_male"],
        errors="ignore",
    ),
    drop_first=True,
).fillna(0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3) 基线对比
qb = QuickBaseline()
results = qb.run_all_models(Xtr, ytr, Xte, yte, task_type="classification")
print(qb.compare_models(results, top_n=3))

# 4) CV
cv = CrossValidator().stratified_cv(LogisticRegression(max_iter=500, random_state=42), X, y, n_splits=5)
print("cv", cv["mean"], cv["std"])

# 5) 集成
voter = ModelEnsemble().voting(
    [LogisticRegression(max_iter=400, random_state=0), DecisionTreeClassifier(max_depth=3, random_state=0)],
    Xtr, ytr, voting="hard",
)

# 6) 重要性
rf = RandomForestClassifier(n_estimators=40, random_state=42).fit(Xtr, ytr)
imp = FeatureImportanceAnalyzer().calculate_importance(rf, Xtr, ytr, method="builtin")
print(imp.head())

# 7) 提交 + 融合
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    sample = pd.DataFrame({"Id": np.arange(len(yte)), "Survived": yte.to_numpy()})
    sample_path = td / "sample.csv"
    sample.to_csv(sample_path, index=False)
    gen = SubmissionGenerator()
    pred = results[qb.compare_models(results, top_n=1).iloc[0]["Model"]]["predictions"]
    p1 = td / "s1.csv"
    p2 = td / "s2.csv"
    gen.create_submission(pred, str(sample_path), str(p1), prediction_type="label")
    gen.create_submission(voter.predict(Xte), str(sample_path), str(p2), prediction_type="label")
    print(gen.validate_submission(str(p1), str(sample_path)))
    ens = gen.ensemble_submissions([str(p1), str(p2)], method="simple_average", output_path=str(td / "ens.csv"))
    print(ens.head())

    tracker = ExperimentTracker(log_file=str(td / "exp.json"))
    tracker.log_experiment("titanic-hb", "rf", list(X.columns), {"n": 40}, float(rf.score(Xte, yte)))
    print(tracker.get_experiment_history().iloc[0]["score"])
```

配套脚本：[`examples/handbook/03_compete_toolkit.py`](../../examples/handbook/03_compete_toolkit.py)

## 使用建议

1. 先泄漏检测，再基线，再 CV，最后才提交  
2. `prediction_type` 必须显式，禁止静默 argmax  
3. `ensemble_submissions` 适合概率/数值列融合；标签融合要小心语义  
4. Optuna 调参见 tutorials/13（需 `[compete]` extras）

## 相关文档

- 短教程：02 / 07 / 13  
- 精讲：`book/lessons/leakage.md`、`submission_contracts.md`
