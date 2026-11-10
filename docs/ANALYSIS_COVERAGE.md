# 数据分析能力覆盖矩阵

**用途：** 对照「常见数据分析方法」与 `examples/advanced` 案例，标出已覆盖 / 缺口，并列出 **wanglaoshi 包侧可优化点**。  
**更新日期：** 2026-11-09  

**课号约定（唯一）：** 进阶 Notebook 一律写作 **E01–E22**（与文件名前缀 `01_`…`22_` 对齐）。  
不要用「案例 21」等混称。路线图里的导航/证据配套勾选写作 **EM01–EM05**（不是课号）。

## 1. 方法 × 案例

| 常见分析能力 | 状态 | 主要案例 | 包内支撑 |
| --- | --- | --- | --- |
| 描述统计 / 缺失 / 异常 EDA | ✅ | 几乎全部；业务 D12 | `DataAnalyzer` |
| 相关 / 共线 / PCA | ✅ | E01、E04、E12、E03 | `correlation_analysis` / `pca_analysis` |
| 分层分类 + 提交契约 | ✅ | E01、E12、E02 | `QuickBaseline` / `SubmissionGenerator` |
| 不均衡 / PR-AUC | ✅ | E05、E04 | 无专用 API（自写 sklearn） |
| 回归 + 目标变换 | ✅ | E06、E07、E11、E03 | `QuickBaseline` |
| 空间 / 分组 CV | ✅ | E06 | `CrossValidator.group_cv` |
| 时间划分 + 滞后 | ✅ | E08；E15 | `extract_datetime_features` / `detect_time_leakage` |
| 聚类稳定性 | ✅ | E09 | 无专用 API |
| 目标编码防泄漏 | ✅ | E10 | `target_encode`（须折内） |
| 集成 | ✅ | E11 | `ModelEnsemble` |
| 业务可追溯流水线 | ✅ | business_pipeline | `BusinessPipeline` |
| 假设检验（t / χ² / ANOVA） | ✅ | E14 | ✅ `StatsToolkit` |
| A/B 试验 / 功效 / 效应量 | ✅ | E13 | ✅ `ExperimentDesign` |
| 贝叶斯 A/B / 先验敏感性 | ✅ | **并入 E13** | ✅ `bayes_prob_beat`（共轭；pymc 可选） |
| 时序分解 / 平稳性 / 简单预测 | ✅ | E15 | ❌ 无分解封装 |
| 缺期对齐 / 同比（零基数/未完月） | ✅ | **并入 E15** | ✅ `yoy_with_rules` / `complete_period_index` |
| 生存 / 流失（KM） | ✅ | E16 | ❌ 无生存模块 |
| 文本 EDA / 词袋特征 | ✅ | E17 | `extract_text_features`（浅） |
| 异常检测（无标签） | ✅ | E18 | ❌ 无 IsolationForest 封装 |
| 关联规则 / 购物篮 | ✅ | E19 | ❌ 宜放 extras |
| 因果推断 / DiD 入门 | ✅ | E20 | ❌ 无因果模块（刻意） |
| 多表关联膨胀 / 三表对账 | ✅ | **E21** | ✅ `check_join_inflation` / `reconcile_to_source` |
| 推荐 / 排序指标（NDCG） | ✅ | **E22** | ❌ 无 RankingMetrics |
| 校准曲线 / Brier / ECE | ✅ | **并入 E04** | ❌ 无校准 API |
| 公平性系统化审计 | ✅ | **并入 E10** | ❌ 无 GroupMetricsReport |

## 2. 由案例反推的包优化优先级

| 优先级 | 优化项 | 依据案例 | 建议 |
| --- | --- | --- | --- |
| P0 | `StatsToolkit` / `ExperimentDesign` | E14 / E13 | ✅ 已落地最小版 |
| P0 | 业务 D02 多表关联 + 膨胀/对账日志 | **E21** | ✅ `reconciliation.py` |
| P0 | 业务 D04 缺期对齐 / 同比规则 | **E15 增补** | ✅ `calendar.py` |
| P1 | `TimeSeriesTools` / `SurvivalAnalyzer` / `AnomalyDetector` | E15 / E16 / E18 | |
| P1 | `RankingMetrics.ndcg_at_k` | **E22** | |
| P1 | 概率校准 + ECE 报表 | **E04 增补** | |
| P1 | `GroupMetricsReport` 公平性 | **E10 增补** | |
| P2 | 文本中文分词、关联规则 extras | E17 / E19 | |

## 3. 课号一览（E01–E22）

| 课号 | 文件 | 主题 |
| --- | --- | --- |
| E01–E12 | `01_iris` … `12_digit_recognizer` | 竞赛/表格基线（见 advanced README） |
| E13 | `13_ab_testing.ipynb` | A/B + 贝叶斯增补 |
| E14 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA |
| E15 | `15_timeseries_forecast.ipynb` | 分解预测 + 同比边缘 |
| E16 | `16_survival_churn.ipynb` | KM / 删失 |
| E17 | `17_text_eda.ipynb` | 文本 EDA |
| E18 | `18_anomaly_detection.ipynb` | 无标签异常 |
| E19 | `19_association_rules.ipynb` | 关联规则 |
| E20 | `20_causal_did.ipynb` | DiD 入门 |
| E21 | `21_order_refund_reconciliation.ipynb` | 三表对账防翻倍 |
| E22 | `22_movielens_ndcg.ipynb` | NDCG@10 |
| — | E04 / E10 正文增补 | 校准 ECE；公平性交叉群体 |

## 4. 验收

```bash
PYTHONPATH=. python scripts/run_advanced_evidence.py
# 期望 passed=22（E01–E22）
```
