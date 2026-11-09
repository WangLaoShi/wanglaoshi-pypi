# 进阶案例总览

每课 = 仓库里的真实 Notebook + 课下练习。  
源码目录：`examples/advanced/`（构建时复制进 `book/cases/`）。  
方法覆盖与包缺口：{doc}`../reference/analysis-coverage`。

## 和 `wanglaoshi` 的强制对应

| 课 | Notebook | 必用 / 必懂的 API 或概念 | 练的纪律 |
| --- | --- | --- | --- |
| 01 Iris | `01_iris.ipynb` | `ready`, `DataAnalyzer`, `QuickBaseline`, `SubmissionGenerator` | 分层；标签提交 |
| 02 Titanic | `02_titanic.ipynb` | `DataAnalyzer`, `AdvancedFeatureEngineer`, `LeakageDetector` | 缺失只在训练折 |
| 04 Breast Cancer | `04_breast_cancer.ipynb` | `QuickBaseline`, `FeatureImportanceAnalyzer` | 缩放进 Pipeline |
| 12 Digits | `12_digit_recognizer.ipynb` | `DataAnalyzer`, `SubmissionGenerator` | 高维 EDA；错分 |
| 06 California | `06_california_housing.ipynb` | `CrossValidator.group_cv` | 同区块不进两侧 |
| 03 Wine Quality | `03_wine_quality.ipynb` | `QuickBaseline`（回归） | 有序/不均衡指标 |
| 07 Abalone | `07_abalone.ipynb` | `one_hot_encode` / 目标变换 | 混合类型 |
| 08 Bike | `08_bike_sharing.ipynb` | `extract_datetime_features`, `detect_time_leakage` | 滞后不得见未来 |
| 09 Mall | `09_mall_customers.ipynb` | EDA + 稳定性 | 聚类非提交赛 |
| 10 Adult | `10_adult_census.ipynb` | `target_encode`（折内） | 目标编码防泄露 |
| 05 Fraud | `05_credit_card_fraud.ipynb` | 指标 + `QuickBaseline` | 别只看 Accuracy |
| 11 House | `11_house_prices.ipynb` | `ModelEnsemble`, `SubmissionGenerator` | 集成与校验 |
| 13 A/B | `13_ab_testing.ipynb` | `DataAnalyzer` + 功效/效应量 | 预注册主指标 |
| 14 假设检验 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA | 效应量与适用性 |
| 15 时序预测 | `15_timeseries_forecast.ipynb` | 时间特征 + 泄漏检测 | 禁止 shuffle 评估 |
| 16 生存流失 | `16_survival_churn.ipynb` | KM / 删失 | 勿当普通分类 |
| 17 文本 | `17_text_eda.ipynb` | `extract_text_features` | 矢量器折内 fit |
| 18 异常 | `18_anomaly_detection.ipynb` | IsolationForest | 无标签 vs E05 |
| 19 关联规则 | `19_association_rules.ipynb` | support/confidence/lift | lift ≠ 因果 |
| 20 因果 DiD | `20_causal_did.ipynb` | 合成 DiD | 平行趋势假设 |
| 21 三表对账 | `21_order_refund_reconciliation.ipynb` | 膨胀倍数 + 对平 | 一对多防翻倍 |
| 22 MovieLens | `22_movielens_ndcg.ipynb` | NDCG@10 | 热门榜 vs 共现；时间切分 |
| 04↑ 校准 | 并入 Breast | Brier / ECE | 同测试集校准前后 |
| 10↑ 公平性 | 并入 Adult | 交叉群体 TPR/FPR | 逐组样本量 |
| 13↑ 贝叶斯 A/B | 并入 A/B | Beta-Binomial | 业务门槛概率 |
| 15↑ 同比边缘 | 并入时序 | 缺月/零基数/未完月 | 先补日历骨架 |

业务专章另计：{doc}`business_pipeline` → `BusinessPipeline`。

## 怎么学

1. 先读精讲 {doc}`../lessons/overview`  
2. 对照 {doc}`../reference/api` 与 {doc}`../reference/glossary`  
3. 合上 Notebook 做练习  
4. 用案例反推包缺口（见覆盖矩阵 P0/P1）

## 数据可得性

- **可直接跑**：Iris、Titanic、Breast、California、Digits，以及 **13–22（合成 / DEMO 回退）**  
- **可能需 `data/`**：Wine、Fraud、Abalone、Bike、Mall、Adult、House（常有 DEMO）

## 批量证据

{doc}`../reference/evidence`
