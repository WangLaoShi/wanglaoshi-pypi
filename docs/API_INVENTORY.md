# A01 公开 API 盘点（初版）

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 A01。

## 顶层导出（`from wanglaoshi import ...`）

| 名称 | 类型 | 依赖 | 备注 |
| --- | --- | --- | --- |
| `JupyterEnv` | 模块 | 标准库/Jupyter | 环境与告警控制 |
| `VERSIONS` | 模块 | packaging 等 | `show_env_info` / `check_all_versions` |
| `Analyzer` / `DataAnalyzer` | 模块/类 | pandas/sklearn/matplotlib | EDA 主入口 |
| `Analyzer_Plain` | 模块 | 同上 | 精简版 |
| `CompetitionTools` | 模块 | sklearn | 基线、提交、泄漏检查等 |
| `FeatureEngineering` | 模块 | pandas/sklearn | 编码/变换 |
| `MLDL` | 可选模块 | **torch** | 未装 torch 时顶层导入不失败 |
| `JupyterFont` / `Useful` / `WebGetter` | 模块 | 视功能而定 | 辅助 |
| `business` / `BusinessPipeline` / `intake_file` 等 | 模块/函数 | pandas | Skill 固化流水线 |

## CompetitionTools 关键契约（B 相关）

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
| D04 | `derived.py` | MVP |
| D05+ | — | pipeline 中显式 skipped |

## 验证

```bash
pytest tests/test_submission_generator.py tests/test_business_intake.py -q
```
