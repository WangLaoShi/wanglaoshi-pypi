# 进阶演示（E01–E22）

对应 GitHub Issue：[#9](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/9)  
总跟踪：[#6](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/6)  
**方法覆盖矩阵：** [docs/ANALYSIS_COVERAGE.md](../../docs/ANALYSIS_COVERAGE.md)

**课号约定：** 一律 **E01–E22**（文件 `01_*.ipynb` … `22_*.ipynb`）。不要混用「案例 21」等写法。

本目录验证 `wanglaoshi` 的 EDA / 特征 / 基线 / 提交能力，并覆盖常见统计分析、对账、推荐排序等。  
教材源：`book/`（`python scripts/build_book.py`）。  
批量证据：`docs/evidence/ADVANCED_EVIDENCE.md`（最近 **22/22 passed**）。

## 学习顺序

| 课号 | Notebook | 核心难点 |
| --- | --- | --- |
| E01 | `01_iris.ipynb` | 小样本分层、标签提交 |
| E02 | `02_titanic.ipynb` | 缺失/文本、训练验证隔离 |
| E03 | `03_wine_quality.ipynb` | 回归/有序、不均衡 |
| E04 | `04_breast_cancer.ipynb` | 缩放/PCA、召回；**校准/ECE** |
| E05 | `05_credit_card_fraud.ipynb` | 不均衡、PR-AUC |
| E06 | `06_california_housing.ipynb` | 空间分组 CV |
| E07 | `07_abalone.ipynb` | 混合编码、目标变换 |
| E08 | `08_bike_sharing.ipynb` | 时间验证、滞后 |
| E09 | `09_mall_customers.ipynb` | 聚类稳定性 |
| E10 | `10_adult_census.ipynb` | 折内目标编码；**公平性** |
| E11 | `11_house_prices.ipynb` | 高维混合、集成 |
| E12 | `12_digit_recognizer.ipynb` | 高维 EDA、错分 |
| E13 | `13_ab_testing.ipynb` | A/B、功效；**贝叶斯门槛** |
| E14 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA |
| E15 | `15_timeseries_forecast.ipynb` | 分解；**同比边缘** |
| E16 | `16_survival_churn.ipynb` | KM、删失 |
| E17 | `17_text_eda.ipynb` | 文本浅特征 / TF-IDF |
| E18 | `18_anomaly_detection.ipynb` | IsolationForest |
| E19 | `19_association_rules.ipynb` | 支持度/置信度/提升度 |
| E20 | `20_causal_did.ipynb` | 合成 DiD |
| E21 | `21_order_refund_reconciliation.ipynb` | 三表对账防翻倍 |
| E22 | `22_movielens_ndcg.ipynb` | NDCG@10（热门榜基线） |

## 生成脚本

```bash
python examples/advanced/_add_curriculum_wave2.py   # E13–E20
python examples/advanced/_add_curriculum_wave3.py   # E21–E22 + E04/E10/E13/E15 增补
```

## 跑通证据

```bash
PYTHONPATH=. python scripts/run_advanced_evidence.py
```
