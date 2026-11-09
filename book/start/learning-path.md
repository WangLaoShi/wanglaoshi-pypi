# 学习路径

按基础选一条主线；进阶案例可以穿插，不必一次读完 12 课。

## 路径 A：数据分析入门（EDA）

适合：第一次接触 pandas / 报告。

1. {doc}`../start/install`  
2. {doc}`../tracks/analyzer`  
3. 案例：{doc}`../cases/01_iris` → {doc}`../cases/02_titanic` → {doc}`../cases/04_breast_cancer`

## 路径 B：竞赛选手（Kaggle 向）

适合：要本地生成 submission、比基线、防泄露。

1. {doc}`../tracks/competition`  
2. 案例顺序（与仓库 `examples/advanced/README` 一致）：

| 顺序 | 课 | 难点 |
| --- | --- | --- |
| 1 | {doc}`../cases/01_iris` | 小样本分层、标签提交 |
| 2 | {doc}`../cases/02_titanic` | 缺失/文本、训练验证隔离 |
| 3 | {doc}`../cases/04_breast_cancer` | 缩放/PCA、召回权衡 |
| 4 | {doc}`../cases/12_digit_recognizer` | 高维 EDA、错分 |
| 5 | {doc}`../cases/06_california_housing` | 空间分组 vs 随机划分 |
| 6 | {doc}`../cases/03_wine_quality` | 回归/有序、不均衡 |
| 7 | {doc}`../cases/07_abalone` | 混合编码、目标变换 |
| 8 | {doc}`../cases/08_bike_sharing` | 时间验证、滞后 |
| 9 | {doc}`../cases/09_mall_customers` | 聚类稳定性 |
| 10 | {doc}`../cases/10_adult_census` | 折内目标编码 |
| 11 | {doc}`../cases/05_credit_card_fraud` | 极端不均衡、PR-AUC |
| 12 | {doc}`../cases/11_house_prices` | 高维混合、集成 |

## 路径 C：业务分析流水线

适合：销售/运营表，要可追溯交付，而不是只拟合模型。

1. {doc}`../tracks/business`  
2. 对照 {doc}`../reference/skill-mapping`  
3. 本地打开 `examples/business_intake_demo.ipynb`

## 每课 Notebook 固定结构

1. 环境检查（`VERSIONS`）  
2. 数据加载  
3. `DataAnalyzer` 快速 EDA  
4. 正确的划分（分层 / 时间 / 分组）  
5. `QuickBaseline` 对比  
6. `FeatureEngineering` 提示  
7. `SubmissionGenerator` 本地生成与校验  
8. 误差分析与检查清单  
