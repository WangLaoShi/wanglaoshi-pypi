# 09 · 校准 + 分组公平（Breast cancer）

## 目标

15 分钟内：对比概率校准（ECE / Brier），并按合成敏感属性做分组 TPR/FPR 审计。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `CalibrationReport` | `analytics` | Brier / ECE / 对比表 |
| `expected_calibration_error` | `analytics` | 单模型 ECE |
| `GroupMetricsReport` | `analytics` | 分组 n / TPR / FPR |
| `ready` | `runtime` | 环境就绪 |

## 数据集

sklearn Breast cancer（可离线）。分组属性用合成 `sex` 演示 API（非真实医疗敏感字段）。

## 完整示例

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import CalibrationReport, GroupMetricsReport

ready(seed=42)
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=2000, random_state=42)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)

print(CalibrationReport.summarize(yte, proba, name="logistic"))
flat = np.full(len(yte), 0.5)
print(CalibrationReport.compare(yte, {"logistic": proba, "flat": flat}))

rng = np.random.default_rng(0)
sex = rng.choice(["F", "M"], size=len(yte))
audit = GroupMetricsReport.audit(yte, pred, {"sex": sex})
print(audit)
```

配套脚本：[`examples/tutorials/09_calibration_fairness.py`](../../examples/tutorials/09_calibration_fairness.py)

## 常见坑

- 准确率高也可能校准差；部署阈值前先看 ECE / 可靠性图
- 分组差异可能来自样本量与基率，不能直接当「歧视定论」
- 真实公平审计需业务定义的敏感属性与合规流程

## 下一步

- 测试治理 → [10_testing_repro.md](10_testing_repro.md)
- 深课 → `examples/advanced/04_breast_cancer.ipynb`、`10_adult_census.ipynb`
