# 第一小时：最小闭环

目标：在本地跑通「读数据 → 简单模型 → 生成提交文件」。

## 1. Iris 基线（可离线）

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from wanglaoshi.runtime import ready
from wanglaoshi.CompetitionTools import SubmissionGenerator

ready()  # 关警告 + 打印环境（不要再叠 JE.no_warning / filterwarnings）

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)
print("valid acc:", model.score(X_valid, y_valid))

# 构造假 sample_submission，练习本地生成
sample = pd.DataFrame({"Id": np.arange(len(X_valid)), "target": y_valid.to_numpy()})
sample_path = "sample_submission.csv"
sample.to_csv(sample_path, index=False)

pred = model.predict(X_valid)
gen = SubmissionGenerator()
sub = gen.create_submission(pred, sample_path, "submission.csv")
print(gen.validate_submission("submission.csv", sample_path))
```

## 2. 下一步

- 完整课：{doc}`../cases/01_iris`  
- 要做 EDA 报告：{doc}`../tracks/analyzer`  
- 要做业务交付：{doc}`../tracks/business`  
