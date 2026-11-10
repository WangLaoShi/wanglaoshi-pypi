# A01 公开 API 盘点

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 A01。  
Skill 映射见 [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md)。

> **详尽签名 + 已校验可运行示例** 见 **[API_REFERENCE.md](API_REFERENCE.md)**（`python scripts/verify_api_reference.py`）。  
> 本文件只做短盘点；有冲突以 `API_REFERENCE.md` 与源码为准。

## 推荐门面（稳定公开）

见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。新代码优先：

| 门面 | 示例 |
| --- | --- |
| `wanglaoshi.compete` | `SubmissionGenerator`, `QuickBaseline` |
| `wanglaoshi.analyze` | `DataAnalyzer`, `analyze_data` |
| `wanglaoshi.features` | `AdvancedFeatureEngineer` |
| `wanglaoshi.analytics` | `StatsToolkit`, `TimeSeriesTools` |
| `wanglaoshi.runtime` | `ready`, `VERSIONS`, `JupyterEnv` |
| `wanglaoshi._testing` | `assert_close`, `ExperimentContext` |
| `wanglaoshi.business` | `BusinessPipeline` |

## 顶层导出（`from wanglaoshi import ...`）

`__all__` 分两层：`_STABLE_PUBLIC`（门面 + 捷径）与 `_COMPAT_EXPORTS`（课堂兼容）。

| 名称 | 类型 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `compete` / `analyze` / `features` / `analytics` / `runtime` / `_testing` | 子包（实现已迁入） | — | **稳定**；根上旧模块为 shim；见 `docs/ARCHITECTURE.md` |
| `ready` | 函数 | 标准库 (+ 可选 numpy) | **稳定** Notebook 一键就绪 |
| `DataAnalyzer` / `analyze_*` | 类/函数 | pandas/sklearn/matplotlib | **稳定**捷径（等价 `analyze` 门面） |
| `business` / `BusinessPipeline` / … | 模块/类/函数 | pandas | **稳定**；见 `wanglaoshi.business` |
| `stats` / `StatsToolkit` / `TimeSeriesTools` / … | 模块/类 | 视功能而定 | **稳定**捷径（等价 `analytics`） |
| `assert_close` / `DataRegistry` / `ExperimentContext` / … | 函数/类 | 视功能而定 | **稳定**捷径（等价 `_testing`） |
| `JupyterEnv` / `VERSIONS` / `JupyterFont` / `Useful` / `WebGetter` | 模块 | 视功能而定 | **兼容**；推荐 `runtime` 门面 |
| `Analyzer` / `CompetitionTools` / `FeatureEngineering` | 模块 | sklearn/pandas | **兼容**；推荐对应门面 |
| `Analyzer_Plain` | 模块 | 同上 | **Legacy** soft deprecate；`WANGLAOSHI_SILENCE_LEGACY=1` 可关警告 |
| `wqe`（非顶层 `__all__`） | 模块 | 同上 | **Legacy** soft deprecate；请优先 `wanglaoshi.analyze` |
| `MLDL` | 可选模块 | **torch** | **兼容**；未装 torch 时可能为 `None` |
| `JupyterEnv.no_warning` | 函数 | warnings | 仅关警告；课堂请优先 `ready()` |
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
