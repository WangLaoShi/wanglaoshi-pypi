# 精讲：提交文件契约

Kaggle / 内赛提交不是「把预测存成 CSV」那么简单：**列名、ID 顺序、标签类型、概率列数** 都必须与 `sample_submission` 一致。`SubmissionGenerator` 的价值是把这些契约写成可重复的生成与校验。

## 1. 契约清单（交作业前勾）

| 检查项 | 正确行为 | 错误后果 |
| --- | --- | --- |
| ID 列 | 与 sample **同名、同顺序、同集合** | 错位提交、静默低分 |
| 目标列名 | 跟 sample，不擅自改名 | 平台拒收或读错列 |
| 标签 dtype | 字符串标签保持 categorical；整数保持整数 | 校验误判、统计无意义 |
| 概率矩阵 | 多列概率需显式 `class_labels`；禁止静默 `argmax` | 类别错位 |
| 缺失/Inf | 生成前清洗或按规则拒绝 | 平台失败 |
| 线下分数 | **不要**写进提交文件 | 违反竞赛纪律 |

## 2. 完整小实验：整数标签

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wanglaoshi import ready
from wanglaoshi.CompetitionTools import SubmissionGenerator

ready()

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=500, random_state=42).fit(X_tr, y_tr)
pred = model.predict(X_te)

sample = pd.DataFrame({"Id": np.arange(len(X_te)), "target": y_te.to_numpy()})
sample.to_csv("sample_submission.csv", index=False)

gen = SubmissionGenerator()
gen.create_submission(pred, "sample_submission.csv", "submission.csv")
print(gen.validate_submission("submission.csv", "sample_submission.csv"))
```

## 3. 字符串标签：不要先转成 0/1/2 再猜映射

```python
# 假设比赛要求 Species 为字符串
sample = pd.DataFrame({
    "Id": [0, 1, 2],
    "Species": ["setosa", "versicolor", "virginica"],
})
sample.to_csv("sample_str.csv", index=False)

# 预测必须是同类字符串（或你在文档中声明的映射，并在校验中一致）
pred_str = np.array(["setosa", "versicolor", "setosa"])
gen.create_submission(pred_str, "sample_str.csv", "sub_str.csv", target_column="Species")
print(gen.validate_submission("sub_str.csv", "sample_str.csv"))
```

```{important}
B 阶段修复过的坑：模板目标列名未推断、字符串标签写出后统计失败、概率静默 argmax。你的代码应以 **显式参数** 消除歧义；歧义时应报错。
```

## 4. 故意写坏，看校验是否拦住

建议自己改一版「坏提交」并确认 `validate_submission` 失败：

1. 打乱 ID 顺序  
2. 少一行  
3. 目标列改名 `Target`  
4. 混入 `np.nan`  

能被拦住 = 你理解了契约；拦不住 = 提 Issue / 补测试（对应路线图 C06 方向）。

## 5. 与真实案例的关系

- Iris / Digits：标签提交 — {doc}`../cases/01_iris`、{doc}`../cases/12_digit_recognizer`  
- House Prices：数值目标 — {doc}`../cases/11_house_prices`  
- 无标签测试集：**只生成，不本地「刷测试分」**（E15）

## 6. 自测

1. `predict_proba` 得到 `(n, 3)`，sample 只有一列 `target`，正确做法是什么？  
2. 为什么「验证集 Accuracy」不能当作提交文件的一列？  
3. sample 有 1000 行，预测只有 999，校验应返回什么性质的结果？
