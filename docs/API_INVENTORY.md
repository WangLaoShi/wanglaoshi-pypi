# A01 公开 API 盘点

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 A01。  
Skill 映射见 [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md)。

## 顶层导出（`from wanglaoshi import ...`）

| 名称 | 类型 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `JupyterEnv` | 模块 | 标准库/Jupyter | `no_warning` 等 |
| `VERSIONS` | 模块 | rich / requests | `show_env_info` / `check_all_versions` |
| `Analyzer` / `DataAnalyzer` | 模块/类 | pandas/sklearn/matplotlib | EDA 主入口 |
| `Analyzer_Plain` | 模块 | 同上 | 精简版 |
| `CompetitionTools` | 模块 | sklearn | 基线、提交、泄漏检查等 |
| `FeatureEngineering` | 模块 | pandas/sklearn | 编码/变换 |
| `MLDL` | 可选模块 | **torch** | 未装 torch 时顶层导入不失败；`MLDL` 可能为 `None` |
| `JupyterFont` / `Useful` / `WebGetter` | 模块 | 视功能而定 | 辅助 |
| `business` | 模块 | pandas/matplotlib | Skill 固化流水线 |
| `BusinessPipeline` / `intake_file` / `generate_insights` / `archive_delivery` / `archive_charts` / `run_eda_profile` … | 函数/类 | pandas | 见 `wanglaoshi.business` |

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
| D02 | `structuring.py` | MVP |
| D03 | `cleaning.py` | MVP |
| D12 | `eda.py` | MVP（复用 DataAnalyzer） |
| D04 | `derived.py` | MVP |
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
