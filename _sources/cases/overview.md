# 进阶案例总览（E01–E22）

每课 = 仓库里的真实 Notebook + 课下练习。  
**课号唯一写法：E01–E22**（与 `examples/advanced/01_*.ipynb` 前缀一致）。  
源码目录：`examples/advanced/`（构建时复制进 `book/cases/`）。  
方法覆盖与包缺口：{doc}`../reference/analysis-coverage`。

## 和 `wanglaoshi` 的强制对应

| 课号 | Notebook | 必用 / 必懂的 API 或概念 | 练的纪律 |
| --- | --- | --- | --- |
| E01 | `01_iris.ipynb` | `ready`, `DataAnalyzer`, `QuickBaseline`, `SubmissionGenerator` | 分层；标签提交 |
| E02 | `02_titanic.ipynb` | `DataAnalyzer`, `AdvancedFeatureEngineer`, `LeakageDetector` | 缺失只在训练折 |
| E03 | `03_wine_quality.ipynb` | `QuickBaseline`（回归） | 有序/不均衡指标 |
| E04 | `04_breast_cancer.ipynb` | `QuickBaseline`, `FeatureImportanceAnalyzer`；**校准/ECE** | 缩放进 Pipeline |
| E05 | `05_credit_card_fraud.ipynb` | 指标 + `QuickBaseline` | 别只看 Accuracy |
| E06 | `06_california_housing.ipynb` | `CrossValidator.group_cv` | 同区块不进两侧 |
| E07 | `07_abalone.ipynb` | `one_hot_encode` / 目标变换 | 混合类型 |
| E08 | `08_bike_sharing.ipynb` | `extract_datetime_features`, `detect_time_leakage` | 滞后不得见未来 |
| E09 | `09_mall_customers.ipynb` | EDA + 稳定性 | 聚类非提交赛 |
| E10 | `10_adult_census.ipynb` | `target_encode`；**公平性交叉群体** | 折内目标编码 |
| E11 | `11_house_prices.ipynb` | `ModelEnsemble`, `SubmissionGenerator` | 集成与校验 |
| E12 | `12_digit_recognizer.ipynb` | `DataAnalyzer`, `SubmissionGenerator` | 高维 EDA；错分 |
| E13 | `13_ab_testing.ipynb` | `DataAnalyzer` + 功效；**贝叶斯门槛** | 预注册主指标 |
| E14 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA | 效应量与适用性 |
| E15 | `15_timeseries_forecast.ipynb` | 时间特征；**同比边缘** | 禁止 shuffle 评估 |
| E16 | `16_survival_churn.ipynb` | KM / 删失 | 勿当普通分类 |
| E17 | `17_text_eda.ipynb` | `extract_text_features` | 矢量器折内 fit |
| E18 | `18_anomaly_detection.ipynb` | IsolationForest | 无标签 vs E05 |
| E19 | `19_association_rules.ipynb` | support/confidence/lift | lift ≠ 因果 |
| E20 | `20_causal_did.ipynb` | 合成 DiD | 平行趋势假设 |
| E21 | `21_order_refund_reconciliation.ipynb` | 膨胀倍数 + 对平 | 一对多防翻倍 |
| E22 | `22_movielens_ndcg.ipynb` | NDCG@10 | 热门榜 vs 共现；时间切分 |

业务专章另计：{doc}`business_pipeline` → `BusinessPipeline`。

## 怎么学

1. 先读精讲 {doc}`../lessons/overview`  
2. 对照 {doc}`../reference/api` 与 {doc}`../reference/glossary`  
3. 合上 Notebook 做练习  
4. 用案例反推包缺口（见覆盖矩阵 P0/P1）

## 数据可得性

- **可直接跑**：E01、E02、E04、E06、E12，以及 **E13–E22（合成 / DEMO 回退）**  
- **可能需 `data/`**：E03、E05、E07、E08、E09、E10、E11（常有 DEMO）

## 批量证据

{doc}`../reference/evidence`
