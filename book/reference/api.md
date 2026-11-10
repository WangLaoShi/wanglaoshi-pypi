```{note}
短盘点来自仓库 `docs/API_INVENTORY.md`（构建时也会同步到 {doc}`api-inventory`）。  
**详尽 API 说明书（含已校验可运行示例）** 请打开仓库：  
[docs/API_REFERENCE.md](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/docs/API_REFERENCE.md)  
架构分层：[docs/ARCHITECTURE.md](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/docs/ARCHITECTURE.md)  
本地校验：`PYTHONPATH=. python scripts/verify_api_reference.py`
```

# 公开 API（教材入口）

本站教的就是包 **`wanglaoshi`**（仓库 `wanglaoshi-pypi`）。自 **1.15.1** 起只教**门面导入**。查签名请以说明书为准。

## 门面速查

| 门面 | 用途 | 常见入口 |
| --- | --- | --- |
| `wanglaoshi.runtime` | Notebook 就绪 / 环境 / 下载 | `ready`, `VERSIONS`, `JupyterEnv`, `WebGetter` |
| `wanglaoshi.analyze` | EDA 与报告 | `DataAnalyzer`, `analyze_data`, `analyze_notebook` |
| `wanglaoshi.compete` | 提交、基线、泄漏、集成、CV | `SubmissionGenerator`, `QuickBaseline`, `LeakageDetector` |
| `wanglaoshi.features` | 特征工程 | `AdvancedFeatureEngineer` |
| `wanglaoshi.analytics` | 统计 / 时序 / 校准 / 公平等 | `StatsToolkit`, `TimeSeriesTools` |
| `wanglaoshi.ml` | ML/DL（torch 可选） | `DataPreprocessor`, `MLModel`, `TORCH_AVAILABLE` |
| `wanglaoshi.business` | 业务 Skill 主链 | `BusinessPipeline`, `StageResult` |
| `wanglaoshi._testing` | 测试与复现治理 | `assert_close`, `DataRegistry`, `ExperimentContext` |

```python
from wanglaoshi.runtime import ready
from wanglaoshi.analyze import DataAnalyzer
from wanglaoshi.compete import QuickBaseline, SubmissionGenerator
from wanglaoshi.features import AdvancedFeatureEngineer
```

## 关键契约

- `prediction_type`：按提交模板约定标签或概率；禁止静默 argmax
- `ModelEnsemble.blending` 需要 `X_val` / `y_val`
- `CrossValidator.group_cv` 需要 `groups`
- 业务阶段：`passed` / `warning` / `blocked` / `skipped`
- 旧顶层模块名（`CompetitionTools` / `FeatureEngineering` / `MLDL` 等）仅为过渡 shim，教材不再使用

## 相关页

- 术语：{doc}`glossary`
- 短盘点：{doc}`api-inventory`
- 进度：{doc}`project-status`
- Skill 映射：{doc}`skill-mapping`
