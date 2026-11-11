# 02 · compete 基线与提交（Titanic）

## 目标

15 分钟内：用 seaborn Titanic 跑 `QuickBaseline`，再用 `SubmissionGenerator` 写出并校验提交文件。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `ready` | `runtime` | 环境就绪 |
| `QuickBaseline` | `compete` | 多模型快速对比 |
| `SubmissionGenerator` | `compete` | 按模板生成/校验提交 |

## 数据集

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| Titanic | `seaborn.load_dataset("titanic")` | 公开内置表；**须删除泄漏列 `alive`** |

## 完整示例

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from wanglaoshi.runtime import ready
from wanglaoshi.compete import QuickBaseline, SubmissionGenerator

ready(seed=42)

raw = sns.load_dataset("titanic").copy()
# alive ≈ survived，直接当特征即目标泄漏
drop_cols = [c for c in ("alive", "deck", "embark_town", "class", "who", "adult_male") if c in raw.columns]
df = raw.drop(columns=drop_cols)
y = df.pop("survived")
X = pd.get_dummies(df, drop_first=True).fillna(0)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
qb = QuickBaseline()
results = qb.run_all_models(Xtr, ytr, Xte, yte, task_type="classification")
print(qb.compare_models(results, top_n=3))

pred = results[qb.compare_models(results, top_n=1).iloc[0]["Model"]]["predictions"]
sample = pd.DataFrame({"Id": np.arange(len(pred)), "Survived": yte.to_numpy()})
sample.to_csv("sample_submission.csv", index=False)
gen = SubmissionGenerator()
gen.create_submission(pred, "sample_submission.csv", "submission.csv", prediction_type="label")
print(gen.validate_submission("submission.csv", "sample_submission.csv"))
```

配套脚本：[`examples/tutorials/02_compete_baseline.py`](../../examples/tutorials/02_compete_baseline.py)

## 常见坑

- 必须显式 `prediction_type`（`label` / 概率约定），禁止静默 `argmax`
- 泄漏列（如 `alive`）进模型会虚高离线分
- 基线表是参照，不是拿验证集刷到过拟合的许可证

## 下一步

- 特征折内编码 → [03_features_encode.md](03_features_encode.md)
- 完整 Titanic 课 → `examples/advanced/02_titanic.ipynb`
- 精讲：提交契约 / 泄漏 → `book/lessons/`
