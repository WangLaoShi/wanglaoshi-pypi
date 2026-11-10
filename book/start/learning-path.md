# 学习路径

先选主线，但**所有路径都先过精讲课**——否则案例会显得「又短又浅」。

## 共同前置（约 90–120 分钟）

1. {doc}`install`  
2. {doc}`../lessons/overview` → 至少读完泄露 / 划分 / 提交三课  
3. {doc}`first-hour`（全是 `wanglaoshi` API）  

## 路径 A：数据分析入门（EDA）

1. {doc}`../tracks/analyzer`  
2. {doc}`../cases/01_iris` → {doc}`../cases/02_titanic` → {doc}`../cases/04_breast_cancer`  
3. 每课练习必须点名 `DataAnalyzer` 方法  

## 路径 B：竞赛选手（Kaggle 向）

1. {doc}`../tracks/competition`  
2. {doc}`../lessons/submission_contracts` + {doc}`../lessons/feature_fit_discipline`  
3. 按顺序啃案例（与仓库 `examples/advanced/README` 一致）：

| 课号 | 课 | 必绑 API / 纪律 |
| --- | --- | --- |
| E01 | {doc}`../cases/01_iris` | 提交契约 + 分层 |
| E02 | {doc}`../cases/02_titanic` | 折内填充 |
| E04 | {doc}`../cases/04_breast_cancer` | 重要性 / 召回 / 校准 |
| E12 | {doc}`../cases/12_digit_recognizer` | 高维 EDA |
| E06 | {doc}`../cases/06_california_housing` | 分组划分 |
| E03 | {doc}`../cases/03_wine_quality` | 回归指标 |
| E07 | {doc}`../cases/07_abalone` | 混合编码 |
| E08 | {doc}`../cases/08_bike_sharing` | 时间 / 滞后 |
| E09 | {doc}`../cases/09_mall_customers` | 聚类稳定性 |
| E10 | {doc}`../cases/10_adult_census` | 折内目标编码 / 公平性 |
| E05 | {doc}`../cases/05_credit_card_fraud` | 不均衡指标 |
| E11 | {doc}`../cases/11_house_prices` | 集成 + 提交 |

## 路径 D：统计 / 试验 / 时序 / 文本 / 对账 / 推荐（E13–E22）

补齐常见分析方法；见 {doc}`../reference/analysis-coverage`。课号一律 **Exx**。

| 课号 | 课 | 焦点 |
| --- | --- | --- |
| E13 | {doc}`../cases/13_ab_testing` | A/B、功效、贝叶斯门槛概率 |
| E14 | {doc}`../cases/14_hypothesis_testing` | t / χ² / ANOVA |
| E15 | {doc}`../cases/15_timeseries_forecast` | 分解 + 同比缺月/零基数 |
| E16 | {doc}`../cases/16_survival_churn` | KM 与删失 |
| E17 | {doc}`../cases/17_text_eda` | 文本特征 |
| E18 | {doc}`../cases/18_anomaly_detection` | 无标签异常 |
| E19 | {doc}`../cases/19_association_rules` | 购物篮规则 |
| E20 | {doc}`../cases/20_causal_did` | DiD 入门 |
| E21 | {doc}`../cases/21_order_refund_reconciliation` | 三表对账防翻倍 |
| E22 | {doc}`../cases/22_movielens_ndcg` | NDCG@10 |

## 路径 C：业务分析流水线

1. {doc}`../lessons/business_traceability`  
2. {doc}`../tracks/business`  
3. {doc}`../cases/business_pipeline` + 练习  

## 和仓库的关系（背下来）

- 包名 / 导入：`wanglaoshi`  
- 说明书：`docs/API_REFERENCE.md`  
- 发版：`./scripts/release_check.sh`  
- 进度：`PROJECT_STATUS.md`  
