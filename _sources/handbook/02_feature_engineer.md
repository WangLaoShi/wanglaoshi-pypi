```{note}
本页从仓库 `docs/handbook/02_feature_engineer.md` 同步生成；请以仓库文件为编辑源。
```

# 使用手册 02 · `AdvancedFeatureEngineer`

## 这个类是干什么的

`wanglaoshi.features.AdvancedFeatureEngineer` 覆盖竞赛常用特征套路：时间拆解、文本浅特征、目标/频率/独热编码、数值变换、特征选择。

```python
from wanglaoshi.features import AdvancedFeatureEngineer
eng = AdvancedFeatureEngineer()
```

## 方法地图

| 类别 | 方法 | 注意 |
| --- | --- | --- |
| 时间 | `extract_datetime_features` | 只拆「当时可得」字段 |
| 文本 | `extract_text_features` | `basic` / `tfidf` / `count` |
| 类别 | `target_encode` / `frequency_encode` / `one_hot_encode` | **TE 必须折内** |
| 数值 | `log_transform` / `boxcox_transform` / `bin_features` | Box-Cox 需正值 |
| 选择 | `select_features_by_importance` / `_correlation` / `_mutual_info` | 选择也要在训练折拟合 |

## 案例：合成电商小表

一张表同时含时间、文本备注、高基类别、数值金额，走一遍主要方法。

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from wanglaoshi.runtime import ready
from wanglaoshi.features import AdvancedFeatureEngineer

ready(seed=42, quiet=True, show_env=False)
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame(
    {
        "order_time": pd.date_range("2024-01-01", periods=n, freq="D"),
        "note": rng.choice(["快", "普通备注 hello", "差评 物流慢", "好评"], size=n),
        "city": rng.choice([f"c{i}" for i in range(25)], size=n),
        "amount": rng.lognormal(mean=3.0, sigma=0.5, size=n),
    }
)
# 弱相关标签
df["y"] = (df["amount"] > df["amount"].median()).astype(int)

eng = AdvancedFeatureEngineer()
df = eng.extract_datetime_features(df, "order_time")
df = eng.extract_text_features(df, "note", method="basic")
df = eng.frequency_encode(df, "city")
# 低基数演示 one-hot（高基数生产慎用）
df_small = eng.one_hot_encode(df.head(30).copy(), ["city"])
train = df.iloc[:150].copy()
valid = df.iloc[150:].copy()
train = eng.target_encode(train, "city", "y", smoothing=5.0)
te_col = "city_target_encoded"
te_map = train.groupby("city")[te_col].first()
valid[te_col] = valid["city"].map(te_map)

df_num = eng.log_transform(df, ["amount"])
df_num = eng.boxcox_transform(df_num, ["amount"])
df_num = eng.bin_features(df_num, ["amount"], n_bins=4)

feat_cols = [
    c
    for c in train.columns
    if c not in {"y", "note", "order_time", "city"} and pd.api.types.is_numeric_dtype(train[c])
]
X = train[feat_cols].fillna(0)
y = train["y"]
sel_imp = eng.select_features_by_importance(X, y, n_features=8, model_type="random_forest")
sel_corr = eng.select_features_by_correlation(X, y, threshold=0.05)
sel_mi = eng.select_features_by_mutual_info(X, y, n_features=8)
print("imp", sel_imp)
print("corr", sel_corr[:5], "ohe_cols", [c for c in df_small.columns if c.startswith("city")][:5])
print("mi", sel_mi)
```

配套脚本：[`examples/handbook/02_feature_engineer.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/handbook/02_feature_engineer.py)

## 使用建议

1. **目标编码**：禁止 `concat(train,test)` 后 `target_encode`  
2. 高基数优先 `frequency_encode` / 折内 TE，慎用巨型 one-hot  
3. `boxcox_transform` 前检查正值与零/负值  
4. 特征选择与标准化一样，必须在训练折内完成

## 相关文档

- 短教程：[tutorials/03](../tutorials/03_features_encode)  
- 精讲：`book/lessons/feature_fit_discipline.md`  
- 深课：`examples/advanced/10_adult_census.ipynb`
