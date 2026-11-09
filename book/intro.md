# 基线工坊 · 从 EDA 到提交

这是 **Python 包 `wanglaoshi`**（仓库 [WangLaoShi/wanglaoshi-pypi](https://github.com/WangLaoShi/wanglaoshi-pypi)）的公开教材站，不是另一套玩具教程。

> 安装：`pip install wanglaoshi` · 开发：`pip install -e ".[dev]"` · API 真源：仓库 `docs/API_REFERENCE.md`

## 本站 ↔ 仓库怎么对应

| 你在本站看到的 | 仓库里真实存在的 |
| --- | --- |
| Notebook 案例课 | `examples/advanced/*.ipynb` |
| 业务专章 | `examples/business_pipeline_course.ipynb` + `wanglaoshi/business/` |
| 三条主线文稿 | 直接讲 `DataAnalyzer` / `CompetitionTools` / `BusinessPipeline` |
| 精讲课 | 讲清泄露、划分、提交契约——对应包里的检测器与生成器 |
| API / 进度 | `docs/API_REFERENCE.md`、`PROJECT_STATUS.md` |
| 发版检查 | `./scripts/release_check.sh` |

学完一课，你应该能指出：**用了哪个类、哪个方法、对应仓库哪段测试或说明书**。

## 建议读法（别只看 Notebook）

1. {doc}`start/install` → 装上**同一个** `wanglaoshi`  
2. {doc}`lessons/overview` → 精讲（比案例更深）  
3. {doc}`tracks/competition` / {doc}`tracks/analyzer` / {doc}`tracks/business` → 选主线  
4. {doc}`start/first-hour` → 一小时最小闭环（全是包 API）  
5. {doc}`cases/overview` → 12 课；每课练习必须点名 API  
6. {doc}`reference/api` → 查签名；冲突以仓库说明书为准  

```{warning}
只「跑通 Notebook」不算学会。练习页要求你说出方法名（例如 `SubmissionGenerator.validate_submission`），并知道失败时错误从哪来。
```

## 你能练到的包能力

- `ready()`：环境就绪（别叠三次 `filterwarnings`）  
- `DataAnalyzer` / `analyze_data` / `analyze_notebook`：可交付 EDA  
- `CompetitionTools`：基线、提交、泄露、集成、CV、实验记录  
- `FeatureEngineering.AdvancedFeatureEngineer`：时间/文本/编码/选特征  
- `BusinessPipeline`：Skill 固化主链（接收→归档）  

开始：{doc}`start/learning-path`
