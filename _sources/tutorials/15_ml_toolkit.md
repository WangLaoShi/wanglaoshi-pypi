```{note}
本页从仓库 `docs/tutorials/15_ml_toolkit.md` 同步生成；请以仓库文件为编辑源。
```

# 15 · ml 工具箱（特征 / 评估 / DL 入口）

## 目标

覆盖 `FeatureEngineer`、`ModelEvaluator`、`DLModel`、`NeuralNetwork`、`example_usage`。

## 本节公开接口

上述符号 + `DataPreprocessor` / `MLModel` / `TORCH_AVAILABLE`

## 数据集

小合成表。

## 完整示例

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from wanglaoshi.runtime import ready
from wanglaoshi.ml import (
    FeatureEngineer,
    ModelEvaluator,
    DataPreprocessor,
    MLModel,
    DLModel,
    NeuralNetwork,
    TORCH_AVAILABLE,
    example_usage,
)

ready(seed=42, quiet=True, show_env=False)

df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0], "B": ["a", "b", "a", "c", "b"]})
pre = DataPreprocessor()
df_num = pre.handle_missing_values(df[["A"]], method="mean")
df = df.copy()
df["A"] = df_num["A"]
df = pre.encode_categorical(df, ["B"])

fe = FeatureEngineer()
df = fe.create_polynomial_features(df, ["A"], degree=2)
df = fe.create_interaction_features(df, ["A", "A_power_2"])

X = df[["A", "A_power_2"]]
y = df["B"]
model = MLModel("logistic")
model.train(X, y)
pred = model.predict(model.X_test)

ev = ModelEvaluator()
ev.plot_confusion_matrix(model.y_test, pred)
ev.plot_learning_curve(
    np.array([2, 3, 4]),
    np.array([[0.7, 0.72], [0.75, 0.76], [0.8, 0.81]]),
    np.array([[0.65, 0.66], [0.7, 0.71], [0.74, 0.75]]),
)

print("torch", TORCH_AVAILABLE)
assert DLModel is not None and NeuralNetwork is not None
if TORCH_AVAILABLE:
    _ = NeuralNetwork(2, 4, 2)
    _ = DLModel(2, 4, 2)

example_usage()
```

配套脚本：[`examples/tutorials/15_ml_toolkit.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/15_ml_toolkit.py)

## 常见坑

- `DLModel` / `NeuralNetwork` 无 torch 时可导入，构造会 `ImportError`
- 评估绘图请设 `MPLBACKEND=Agg`

## 下一步

- business 阶段函数 → [16_business_stages.md](16_business_stages.md)
