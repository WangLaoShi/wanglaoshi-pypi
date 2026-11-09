# 主线：竞赛工具链

## 模块地图

| 类 | 干什么 |
| --- | --- |
| `SubmissionGenerator` | 按 sample 模板生成/校验提交 |
| `QuickBaseline` | 快速多模型对比 |
| `LeakageDetector` | 目标/时间/分布泄露提示 |
| `ModelEnsemble` | voting / stacking / blending |
| `CrossValidator` | 分层 / 时间 / 分组 CV |
| `FeatureImportanceAnalyzer` | 内置/排列重要性 |
| `ExperimentTracker` | 实验记录 JSON |
| `HyperparameterOptimizer` | Optuna（需 `[optuna]`） |

```python
from wanglaoshi.CompetitionTools import (
    SubmissionGenerator,
    QuickBaseline,
    LeakageDetector,
)
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer
```

## 关键纪律

- 概率矩阵不要静默 `argmax`（除非显式给 `class_labels`）  
- 目标编码、选特征必须在 **训练折内** 拟合  
- 本地无标签测试集只用于生成提交，不用于「刷分」

## 推荐案例

从 {doc}`../cases/01_iris` 开始，按 {doc}`../start/learning-path` 路径 B 走完。
