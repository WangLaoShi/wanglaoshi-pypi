```{note}
短盘点来自仓库 `docs/API_INVENTORY.md`（构建时也会同步到 {doc}`api-inventory`）。  
**详尽 API 说明书（含已校验可运行示例）** 请打开仓库：  
[docs/API_REFERENCE.md](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/docs/API_REFERENCE.md)  
本地校验：`PYTHONPATH=. python scripts/verify_api_reference.py`
```

# 公开 API（教材入口）

本站教的就是包 **`wanglaoshi`**（仓库 `wanglaoshi-pypi`）。查签名请以说明书为准。

## 顶层导出速查

| 名称 | 用途 |
| --- | --- |
| `ready` | Notebook 一键就绪 |
| `DataAnalyzer` / `analyze_data` / `analyze_notebook` | EDA；Notebook 用**模块函数** `analyze_notebook(df)` |
| `CompetitionTools` | 提交、基线、泄露、集成、CV、实验 |
| `FeatureEngineering` | `AdvancedFeatureEngineer` |
| `BusinessPipeline` | 业务 Skill 主链 |
| `WebGetter.Wget` | `from wanglaoshi.WebGetter import Wget` |
| `MLDL` | 可选；未装 torch 时可能为 `None` |

## 关键契约

- `prediction_type` ∈ `{auto, label, proba, multioutput}`（不是 `probability`）
- `ModelEnsemble.blending` 需要 `X_val` / `y_val`
- `CrossValidator.group_cv` 需要 `groups`
- 业务阶段：`passed` / `warning` / `blocked` / `skipped`

## 相关页

- 短盘点：{doc}`api-inventory`
- 进度：{doc}`project-status`
- Skill 映射：{doc}`skill-mapping`
