# 进阶演示（22 个案例）

对应 GitHub Issue：[#9](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/9)  
总跟踪：[#6](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/6)  
**方法覆盖矩阵：** [docs/ANALYSIS_COVERAGE.md](../../docs/ANALYSIS_COVERAGE.md)

本目录验证 `wanglaoshi` 的 EDA / 特征 / 基线 / 提交能力，并覆盖常见统计分析、对账、推荐排序等。  
教材源：`book/`（`python scripts/build_book.py`）。  
批量证据：`docs/evidence/ADVANCED_EVIDENCE.md`（最近 **22/22 passed**）。

## 学习顺序

| 顺序 | Notebook | 核心难点 |
| --- | --- | --- |
| 1–12 | `01`…`12`（Iris→House） | 见既有竞赛/表格基线课 |
| 13 | `13_ab_testing.ipynb` | A/B、功效；**贝叶斯门槛概率** |
| 14 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA |
| 15 | `15_timeseries_forecast.ipynb` | 分解；**同比缺月/零基数/未完月** |
| 16 | `16_survival_churn.ipynb` | KM、删失 |
| 17 | `17_text_eda.ipynb` | 文本浅特征 / TF-IDF |
| 18 | `18_anomaly_detection.ipynb` | IsolationForest |
| 19 | `19_association_rules.ipynb` | 支持度/置信度/提升度 |
| 20 | `20_causal_did.ipynb` | 合成 DiD |
| 21 | `21_order_refund_reconciliation.ipynb` | **三表对账防翻倍** |
| 22 | `22_movielens_ndcg.ipynb` | **NDCG@10**（热门榜基线） |

并入增补（不另开编号）：**04** 校准/ECE；**10** 公平性交叉群体。

## 生成脚本

```bash
python examples/advanced/_add_curriculum_wave2.py   # 13–20
python examples/advanced/_add_curriculum_wave3.py   # 21–22 + 并入增补
```

## 跑通证据

```bash
PYTHONPATH=. python scripts/run_advanced_evidence.py
```
