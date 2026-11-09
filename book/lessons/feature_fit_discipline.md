# 精讲：特征工程的拟合纪律

特征变换若在**全数据**上 `fit`，再划分训练/验证，等于把验证信息掺进训练。目标编码、缺失填充、PCA、特征选择全都适用同一条纪律：**只在训练折上 fit，再 transform 到验证/测试**。

## 1. 一句话规矩

```text
fit 只用训练折可见的行；
transform 可以作用于验证/测试，但不得用它们更新统计量。
```

## 2. 错误 vs 正确：缺失填充

```python
# BAD
df["x"] = df["x"].fillna(df["x"].median())
train, valid = split(df)  # 中位数已含 valid

# GOOD
train, valid = split(df)
med = train["x"].median()
train = train.copy(); valid = valid.copy()
train["x"] = train["x"].fillna(med)
valid["x"] = valid["x"].fillna(med)
```

## 3. 目标编码必须折内完成

Adult 课（{doc}`../cases/10_adult_census`）的核心不是「会不会调用 API」，而是：

```python
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def oof_target_encode(train_df, col, target, n_splits=5, seed=42):
    """生成仅用其他折标签算出的编码，避免本行标签泄漏。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = pd.Series(index=train_df.index, dtype=float)
    global_mean = train_df[target].mean()
    for tr_idx, va_idx in kf.split(train_df):
        means = train_df.iloc[tr_idx].groupby(col)[target].mean()
        out.iloc[va_idx] = train_df.iloc[va_idx][col].map(means).fillna(global_mean)
    return out
```

对**真正的测试集**：用**全部训练集**的 `groupby.mean()` 再 `map`（此时没有测试标签可漏）。

## 4. 特征选择同样不能偷看

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("select", SelectKBest(f_classif, k=10)),
    ("clf", LogisticRegression(max_iter=500)),
])
# 交叉验证时对 Pipeline 整体 score → SelectKBest 每折重新 fit
```

```{warning}
先在全表 `SelectKBest.fit(X, y)` 选出 10 列，再 CV —— 这是经典泄露，论文和作业里都常见。
```

## 5. 结合 `AdvancedFeatureEngineer`

```python
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer

fe = AdvancedFeatureEngineer()
# 模式：在训练折上 fit / 生成映射，再应用到验证折
# 具体方法名见 API 盘点；原则不变：禁止「全表 fit 一次打天下」后划分
```

选特征 API 的入参以当前版本为准；若签名需要 `X, y`，确保 `X,y` 来自**当前训练折**。

## 6. 自测

1. 用全表相关系数选出 Top20 特征再做 `train_test_split`，问题在哪？  
2. 测试集目标编码时，为什么可以用「全部训练集」的均值，却不能在 CV 里对训练折用「含自己」的均值？  
3. Pipeline 相对手写 `fit/transform` 的最大教学价值是什么？

## 7. 下一步

- 泄露总览：{doc}`leakage`  
- 案例：{doc}`../cases/10_adult_census`、{doc}`../cases/02_titanic`
