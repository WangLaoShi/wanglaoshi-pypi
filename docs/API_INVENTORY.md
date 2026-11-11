# A01 公开 API 盘点

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 A01。  
Skill 映射见 [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md)。

> **详尽签名 + 已校验可运行示例** 见 **[API_REFERENCE.md](API_REFERENCE.md)**（`python scripts/verify_api_reference.py`）。  
> 本文件只做短盘点；有冲突以 `API_REFERENCE.md` 与源码为准。

## 推荐公开接口（稳定公开）

见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。新代码优先：

| 接口模块 | 示例 |
| --- | --- |
| `wanglaoshi.compete` | `SubmissionGenerator`, `QuickBaseline` |
| `wanglaoshi.analyze` | `DataAnalyzer`, `analyze_data` |
| `wanglaoshi.features` | `AdvancedFeatureEngineer` |
| `wanglaoshi.analytics` | `StatsToolkit`, `TimeSeriesTools` |
| `wanglaoshi.ml` | `DataPreprocessor`, `MLModel`, `TORCH_AVAILABLE` |
| `wanglaoshi.runtime` | `ready`, `VERSIONS`, `JupyterEnv` |
| `wanglaoshi._testing` | `assert_close`, `ExperimentContext` |
| `wanglaoshi.business` | `BusinessPipeline` |

## 顶层导出说明（1.15.1+）

`__all__` 仍分 `_STABLE_PUBLIC` / `_COMPAT_EXPORTS`，但 **文档与示例只教公开接口**。  
`_COMPAT_EXPORTS` 与根 shim 为过渡保留，**新代码请勿依赖**。

| 名称 | 类型 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `compete` / `analyze` / `features` / `analytics` / `ml` / `runtime` / `_testing` | 子包公开接口 | — | **稳定公开**；见 `docs/ARCHITECTURE.md` |
| `business` / `BusinessPipeline` / … | 模块/类/函数 | pandas | **稳定**；`from wanglaoshi.business import …` |
| `ready`（捷径） | 函数 | 标准库 (+ 可选 numpy) | 等价 `runtime.ready`；示例请写 `runtime` |
| `DataAnalyzer` / `analyze_*`（捷径） | 类/函数 | pandas/sklearn/matplotlib | 等价 `analyze`；示例请写公开接口 |
| `StatsToolkit` / `TimeSeriesTools` 等（捷径） | 类 | 视功能而定 | 等价 `analytics` |
| `assert_close` / `DataRegistry` / …（捷径） | 函数/类 | 视功能而定 | 等价 `_testing` |
| `Analyzer` / `CompetitionTools` / `FeatureEngineering` / `MLDL` / runtime 旧模块 | shim | — | **已放弃推荐**；仅过渡 |
| `Analyzer_Plain` / `wqe` | legacy | 同上 | soft deprecate；`WANGLAOSHI_SILENCE_LEGACY=1` 可关 |
| `validate_metric_dict` / `default_metric_registry` | 函数 | — | S5 指标口径版本登记 |

## CompetitionTools 关键契约

- `SubmissionGenerator.create_submission(...)`
  - 从 sample_submission 推断目标列
  - `prediction_type`: `label` / `probability`（禁止静默 argmax）
  - 字符串标签走类别统计
- `QuickBaseline.run_all_models` / `compare_models`
- 其他：ensemble、leakage、CV 工具（详见源码）

## business 阶段状态

统一 `StageResult.status ∈ {passed, warning, blocked, skipped}`。

| 阶段 | 模块 | 状态 |
| --- | --- | --- |
| D01 | `intake.py` | 已实现 |
| D02 | `structuring.py` + `reconciliation.py` | MVP；**P0 对账/膨胀已加** |
| D03 | `cleaning.py` | MVP |
| D12 | `eda.py` | MVP（复用 DataAnalyzer） |
| D04 | `derived.py` + `calendar.py` | MVP；同比 + **头腰尾 `add_value_tiers`** |
| D05 | `planning.py` | MVP |
| D06 | `metrics.py` | MVP |
| D07 | `analysis.py` | MVP |
| D08 | `visualization.py` | MVP |
| D09 | `insights.py` | MVP |
| D10 | `archive.py` | MVP |
| D11 | `pipeline.py` | 总控已串联 |

## 验证

```bash
python scripts/smoke_docs_examples.py
bash scripts/run_tests.sh quick
pytest tests/ -q
```
