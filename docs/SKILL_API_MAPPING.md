# A02 Skill → API / 配置 / 校验 / 测试映射

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11)。

| Skill | API | 配置键 | 校验失败行为 | 测试 |
| --- | --- | --- | --- | --- |
| data-intake | `intake_file` | `required_columns`, `period` | 缺文件 `blocked`；缺列 `warning` | `test_intake_*` |
| data-structuring | `structure_dataframe` / `structure_file`；P0：`check_join_inflation` / `reconcile_to_source` | `column_map`, `dtype_map`；对账 `on`/`measure_col` | 缺文件 `blocked`；类型失败 `warning`；膨胀 → `warning` | `test_structure_*` / `test_p0_*` |
| data-cleaning | `clean_dataframe` | `fillna`, `clip_bounds`, `drop_duplicates` | 默认不因异常删行；写影响日志 | `test_clean_*` |
| derived-metrics | `derive_metrics`；P0：`complete_period_index` / `yoy_with_rules` / `yoy_stage` | `date_col`, `value_col`；同比 `as_of`/`freq` | 缺关键列 `blocked`；零基数/未完月 → reason | `test_derive_*` / `test_p0_*` |
| analysis-decomposition | `build_analysis_plan` | `field_roles.*` | 无可用方向 `warning` | `test_planning_*` |
| metrics-calculation | `calculate_metrics` | `metrics`, `group_cols`, `share_of` | 零分母/缺列 `warning`；占比不一致 `warning` | `test_metrics_*` |
| multidimensional-analysis | `run_multidimensional_analysis` | `date_col`, `dim_col`, `segment_col` | 视角不足 `warning`；pareto 非因果 | `test_analysis_*` |
| data-visualization | `archive_charts` | `period`, `source`, `unit` | 无图 `warning`；元数据 sidecar | `test_archive_charts_*` |
| exploratory-data-analysis | `run_eda_profile` | `field_roles`, `sample_n` | Analyzer 失败则轻量画像 | `test_eda_*` |
| insight-conclusion / recommendations | `generate_insights` | 阈值来自分析/指标 | 无证据→观察；收益无依据→pending | `test_insights_*` |
| analysis-report / delivery-archive | `archive_delivery` | `redact_fields`, `required` | 缺交付项或上游 blocked → `blocked` | `test_archive_*` |
| data-analysis 总控 | `BusinessPipeline.run` | `pipeline.stop_on_block` | 上游 `blocked` → 下游 `skipped`；无正式交付 | `test_pipeline_*` |
| exploratory-data-analysis | `DataAnalyzer`（复用） | 抽样阈值 | 不阻断流水线 | 既有 Analyzer 用法 |
| 竞赛提交 | `CompetitionTools.SubmissionGenerator` | `prediction_type`, `target_column` | 歧义/非法形状报错 | `tests/test_submission_generator.py` |

## A03 编号约定

| 对象 | 前缀 | 示例 |
| --- | --- | --- |
| 阶段 | `Dxx_name` | `D06_metrics` |
| 证据 | `evidence` / `sha256:` / `metric:` / `plan:` | `sha256:abcd…` |
| 指标 | 配置字典键 | `gmv`, `aov` |
| 运行 | `run_id` | `uuid[:12]` |

统一载体：`StageResult` / `RunRecord`（`wanglaoshi.business.contracts`）。
