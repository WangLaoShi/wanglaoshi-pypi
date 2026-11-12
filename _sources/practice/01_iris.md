# 练习：Iris（绑定 wanglaoshi API）

对应课：{doc}`../cases/01_iris` · 精讲：{doc}`../lessons/submission_contracts` · {doc}`../lessons/cv_and_splits`

## 本课强制 API 清单

| API | 你要能独立写出来 |
| --- | --- |
| `wanglaoshi.ready` | 一次调用完成环境就绪 |
| `DataAnalyzer.get_basic_info` | 说出 shape / 列类型 |
| `CT.QuickBaseline.run_all_models` + `compare_models` | 同划分下比模型 |
| `CT.SubmissionGenerator.create_submission` | `prediction_type='label'` |
| `CT.SubmissionGenerator.validate_submission` | 读懂 `is_valid` / `errors` |

## 学习目标

- 解释小样本多分类为何要用 **分层划分**  
- 区分整数标签与字符串标签的 submission 形态  
- 指出：验证集 Accuracy **不能**写进提交文件  

## 自测

1. `train_test_split(..., stratify=None)` 在 Iris 上最可能出什么问题？  
2. sample 目标列叫 `Species`（字符串）时，`prediction_type` 与 dtype 应如何设？  
3. `prediction_type='probability'` 为什么是错的？（正确枚举见说明书）  

## 动手（最小可运行，须本地跑通）

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from wanglaoshi.runtime import ready, DataAnalyzer
from wanglaoshi import compete as CT

ready(show_env=False)
df = load_iris(as_frame=True).frame
print(DataAnalyzer(df).get_basic_info())

X, y = df.drop(columns=["target"]), df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

base = CT.QuickBaseline()
results = base.run_all_models(X_tr, y_tr, X_te, y_te, task_type="classification")
print(base.compare_models(results, top_n=3))

best = base.compare_models(results, top_n=1).iloc[0]["Model"]
pred = results[best]["predictions"]
sample = pd.DataFrame({"Id": np.arange(len(pred)), "target": y_te.to_numpy()})
sample.to_csv("sample_iris.csv", index=False)
gen = CT.SubmissionGenerator()
gen.create_submission(pred, "sample_iris.csv", "sub_iris.csv", prediction_type="label")
print(gen.validate_submission("sub_iris.csv", "sample_iris.csv"))
```

## 加难（交作业加分项）

1. 故意打乱提交 ID 顺序，确认 `validate_submission` 失败。  
2. 把 `stratify=y` 去掉，打印验证集各类计数，写进 Markdown。  
3. 对照仓库 `tests/test_submission_generator.py`：你的坏提交对应哪类断言？  

## 参考思路

分层保证每份数据都见到三类；提交只对测试 ID 负责；`prediction_type` 合法值是 `auto|label|proba|multioutput`。
