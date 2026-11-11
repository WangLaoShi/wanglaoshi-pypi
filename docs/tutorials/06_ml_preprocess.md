# 06 · ml 预处理与建模（Breast cancer）

## 目标

10 分钟内：用 `DataPreprocessor` 处理缺失/缩放，用 `MLModel` 训练并评估。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `DataPreprocessor` | `wanglaoshi.ml` | 缺失、类别编码、缩放 |
| `MLModel` | `wanglaoshi.ml` | 统一接口的经典 ML 模型 |
| `TORCH_AVAILABLE` | `wanglaoshi.ml` | 是否安装了 torch（可选） |
| `ready` | `runtime` | 环境就绪 |

## 数据集

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| Breast cancer | `sklearn.datasets.load_breast_cancer` | 二分类医学特征表，可离线 |

## 完整示例

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from wanglaoshi.runtime import ready
from wanglaoshi.ml import DataPreprocessor, MLModel, TORCH_AVAILABLE

ready(seed=42)
print("torch available:", TORCH_AVAILABLE)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# 人为制造少量缺失，演示预处理
rng = np.random.default_rng(0)
mask = rng.random(len(df)) < 0.02
df.loc[mask, df.columns[0]] = np.nan

y = df.pop("target")
pre = DataPreprocessor()
df2 = pre.handle_missing_values(df, method="mean")
num_cols = list(df2.columns)
df2 = pre.scale_features(df2, num_cols)

model = MLModel("logistic")
model.train(df2, y, test_size=0.25)
print(model.evaluate())
```

配套脚本：[`examples/tutorials/06_ml_preprocess.py`](../../examples/tutorials/06_ml_preprocess.py)

## 常见坑

- `MLModel("logistic")` 必须传入 `model_type`，不能空构造
- torch **不是**默认依赖；未安装时 sklearn 部分仍可用
- 缩放/编码同样要在训练折内拟合（生产请进 Pipeline）

## 下一步

- 泄漏 + CV → [07_leakage_and_cv.md](07_leakage_and_cv.md)
- 校准/阈值深课 → `examples/advanced/04_breast_cancer.ipynb`
- 回到教程索引 → [README.md](README.md)
- API → [`docs/API_REFERENCE.md`](../API_REFERENCE.md) 中 `ml` 节
