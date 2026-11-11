# 使用手册 05 · `ml` 门面

## 模块里有什么

| 符号 | 作用 |
| --- | --- |
| `DataPreprocessor` | 缺失 / 类别编码 / 缩放 |
| `FeatureEngineer`（ml 内） | 多项式 / 交互（轻量） |
| `MLModel` | 统一经典模型接口 |
| `ModelEvaluator` | 混淆矩阵 / 学习曲线 |
| `DLModel` / `NeuralNetwork` | torch 可选 |
| `TORCH_AVAILABLE` / `example_usage` | 环境探测与官方串联示例 |

> 竞赛向复杂特征请用 `wanglaoshi.features.AdvancedFeatureEngineer`。

## 案例：Breast cancer 预处理 → 训练 → 评估

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.datasets import load_breast_cancer

from wanglaoshi.runtime import ready
from wanglaoshi.ml import (
    DataPreprocessor, FeatureEngineer, MLModel, ModelEvaluator,
    DLModel, NeuralNetwork, TORCH_AVAILABLE, example_usage,
)

ready(seed=42, quiet=True, show_env=False)
df = load_breast_cancer(as_frame=True).frame.copy()
rng = np.random.default_rng(0)
df.loc[rng.random(len(df)) < 0.02, df.columns[0]] = np.nan
y = df.pop("target")

pre = DataPreprocessor()
df = pre.handle_missing_values(df, method="mean")
df = pre.scale_features(df, list(df.columns))

# 可选：多项式增强（演示用前 2 列）
fe = FeatureEngineer()
df2 = fe.create_polynomial_features(df.iloc[:, :2].copy(), list(df.columns[:2]), degree=2)
df2 = fe.create_interaction_features(df2, list(df.columns[:2]))

model = MLModel("logistic")
model.train(df, y, test_size=0.25)
print(model.evaluate())
pred = model.predict(model.X_test)

ModelEvaluator().plot_confusion_matrix(model.y_test, pred)
ModelEvaluator().plot_learning_curve(
    np.array([50, 100, 150]),
    np.array([[0.9, 0.91], [0.93, 0.94], [0.95, 0.96]]),
    np.array([[0.88, 0.89], [0.91, 0.92], [0.93, 0.94]]),
)

print("torch", TORCH_AVAILABLE)
assert DLModel is not None and NeuralNetwork is not None
example_usage()
```

配套脚本：[`examples/handbook/05_ml_guide.py`](../../examples/handbook/05_ml_guide.py)

## 使用建议

1. 缩放 / 编码放入训练折或 Pipeline，避免泄漏  
2. 无 torch 时仍可导入 `DLModel`，但训练会报错提示安装  
3. 需要树模型 + 提交链路时，优先 `compete.QuickBaseline`

## 相关文档

- 短教程：06 / 15  
- API：`API_REFERENCE` 中 `ml` 节
