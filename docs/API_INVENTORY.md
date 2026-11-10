# A01 公开 API 盘点

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 A01。  
Skill 映射见 [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md)。

> **详尽签名 + 已校验可运行示例** 见 **[API_REFERENCE.md](API_REFERENCE.md)**（`python scripts/verify_api_reference.py`）。  
> 本文件只做短盘点；有冲突以 `API_REFERENCE.md` 与源码为准。

## 顶层导出（`from wanglaoshi import ...`）

| 名称 | 类型 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `JupyterEnv` | 模块 | 标准库/Jupyter | `no_warning` 等 |
| `VERSIONS` | 模块 | rich / requests | `show_env_info` / `check_all_versions` |
| `ready` | 函数 | 标准库 (+ 可选 numpy) | **推荐** Notebook 一键就绪；合并关警告+环境信息，避免与 `JE.no_warning` 重复 |
| `Analyzer` / `DataAnalyzer` | 模块/类 | pandas/sklearn/matplotlib | EDA 主入口 |
| `Analyzer_Plain` | 模块 | 同上 | 精简版 |
| `CompetitionTools` | 模块 | sklearn | 基线、提交、泄漏检查等 |
| `FeatureEngineering` | 模块 | pandas/sklearn | 编码/变换 |
| `MLDL` | 可选模块 | **torch** | 未装 torch 时顶层导入不失败；`MLDL` 可能为 `None` |
| `JupyterFont` / `Useful` / `WebGetter` | 模块 | 视功能而定 | 辅助 |
| `JupyterEnv.no_warning` | 函数 | warnings | 仅关警告；与 `warnings.filterwarnings` 同义，课堂请优先 `ready()` |
| `business` | 模块 | pandas/matplotlib | Skill 固化流水线 |
| `BusinessPipeline` / `intake_file` / `generate_insights` / `archive_delivery` / `archive_charts` / `run_eda_profile` … | 函数/类 | pandas | 见 `wanglaoshi.business` |
| `check_join_inflation` / `reconcile_to_source` | 函数 | pandas | 连接膨胀 / 源表对账（P0） |
| `complete_period_index` / `yoy_with_rules` | 函数 | pandas | 缺期骨架 / 同比规则（P0） |
| `stats` / `StatsToolkit` / `ExperimentDesign` | 模块/类 | scipy/statsmodels | 假设检验与 A/B（P0） |
| `TimeSeriesTools` | 类 | statsmodels/pandas | 分解 / ADF / 安全滞后（P1） |
| `CalibrationReport` | 类 | sklearn | Brier / ECE（P1） |
| `GroupMetricsReport` | 类 | sklearn | 分组公平性审计（P1） |
| `AnomalyDetector` | 类 | sklearn | IsolationForest（P1） |
| `RankingMetrics` | 类 | numpy | NDCG@K（P1） |

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
| D04 | `derived.py` + `calendar.py` | MVP；**P0 同比规则已加** |
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
