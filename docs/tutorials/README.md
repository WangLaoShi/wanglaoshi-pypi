# 公开接口教程（1.15.1+）

按各子包的 **稳定公开接口**（软件设计中的 Facade：对外统一入口，隐藏内部实现）拆开的短教程：公开/内置数据，10–20 分钟跑通调用方式。

- 详尽签名：[`docs/API_REFERENCE.md`](../API_REFERENCE.md)
- 类与方法手册：[`docs/handbook/`](../handbook/)
- 架构分层：[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
- 深案例：[`examples/advanced/`](../../examples/advanced/)
- 可执行烟测脚本：[`examples/tutorials/`](../../examples/tutorials/)

## 学习顺序

```text
01 ready+EDA → 02 compete 基线 → 03 features → 06 ml
        │              └→ 07 泄漏+CV → 13 集成/重要性/追踪
        ├→ 04 统计 → 08 时序 → 09 校准/公平 → 14 异常/排序/生存
        ├→ 05 / 16 business
        ├→ 10 / 17 _testing
        ├→ 11 runtime 工具箱
        └→ 12 analyze 辅助
```

### 第一批（上手）

| 课 | 文档 | 接口模块 |
| --- | --- | --- |
| 01 | [01_ready_and_eda.md](01_ready_and_eda.md) | `ready` · `DataAnalyzer` |
| 02 | [02_compete_baseline.md](02_compete_baseline.md) | `QuickBaseline` · `SubmissionGenerator` |
| 03 | [03_features_encode.md](03_features_encode.md) | `AdvancedFeatureEngineer` |
| 04 | [04_analytics_stats.md](04_analytics_stats.md) | `StatsToolkit` · `ExperimentDesign` |
| 05 | [05_business_pipeline.md](05_business_pipeline.md) | `BusinessPipeline` |
| 06 | [06_ml_preprocess.md](06_ml_preprocess.md) | `DataPreprocessor` · `MLModel` |

### 第二批（进阶公开接口）

| 课 | 文档 | 接口模块 |
| --- | --- | --- |
| 07 | [07_leakage_and_cv.md](07_leakage_and_cv.md) | `LeakageDetector` · `CrossValidator` |
| 08 | [08_timeseries_tools.md](08_timeseries_tools.md) | `TimeSeriesTools` |
| 09 | [09_calibration_fairness.md](09_calibration_fairness.md) | `CalibrationReport` · `GroupMetricsReport` |
| 10 | [10_testing_repro.md](10_testing_repro.md) | `DataRegistry` · `ExperimentContext` · 分数下限 |

### 第三批（公开符号补齐）

| 课 | 文档 | 接口模块 |
| --- | --- | --- |
| 11 | [11_runtime_toolkit.md](11_runtime_toolkit.md) | `VERSIONS` · `JupyterEnv` · `JupyterFont` · `Useful` · `WebGetter` |
| 12 | [12_analyze_helpers.md](12_analyze_helpers.md) | `load_data` · `analyze_*` · `DataProcessor` · `Visualizer` · `ReportGenerator` |
| 13 | [13_compete_advanced.md](13_compete_advanced.md) | `ModelEnsemble` · `FeatureImportanceAnalyzer` · `ExperimentTracker` · `HyperparameterOptimizer` |
| 14 | [14_analytics_extra.md](14_analytics_extra.md) | `AnomalyDetector` · `RankingMetrics` · `SurvivalAnalyzer` |
| 15 | [15_ml_toolkit.md](15_ml_toolkit.md) | `FeatureEngineer` · `ModelEvaluator` · `DLModel` · `NeuralNetwork` · `example_usage` |
| 16 | [16_business_stages.md](16_business_stages.md) | business 阶段函数与常量 |
| 17 | [17_testing_extra.md](17_testing_extra.md) | `_testing` 其余公开入口 |

## 一键烟测

```bash
PYTHONPATH=. python scripts/smoke_docs_examples.py
```

## 约定

- 只教公开接口导入
- 优先离线可跑；Optuna / torch 为可选增强
- 深水区仍以 E01–E22 为准
