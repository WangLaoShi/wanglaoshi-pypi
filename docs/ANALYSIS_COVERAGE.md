# 数据分析能力覆盖矩阵

**用途：** 对照「常见数据分析方法」与 `examples/advanced` 案例，标出已覆盖 / 缺口，并列出 **wanglaoshi 包侧可优化点**。  
**更新日期：** 2026-11-09（wave3：六大缺口收口）

## 1. 方法 × 案例

| 常见分析能力 | 状态 | 主要案例 | 包内支撑 |
| --- | --- | --- | --- |
| 描述统计 / 缺失 / 异常 EDA | ✅ | 几乎全部；业务 D12 | `DataAnalyzer` |
| 相关 / 共线 / PCA | ✅ | Iris、Breast、Digits、Wine | `correlation_analysis` / `pca_analysis` |
| 分层分类 + 提交契约 | ✅ | Iris、Digits、Titanic | `QuickBaseline` / `SubmissionGenerator` |
| 不均衡 / PR-AUC | ✅ | Fraud、Breast | 无专用 API（自写 sklearn） |
| 回归 + 目标变换 | ✅ | California、Abalone、House、Wine | `QuickBaseline` |
| 空间 / 分组 CV | ✅ | California | `CrossValidator.group_cv` |
| 时间划分 + 滞后 | ✅ | Bike；案例 15 | `extract_datetime_features` / `detect_time_leakage` |
| 聚类稳定性 | ✅ | Mall | 无专用 API |
| 目标编码防泄漏 | ✅ | Adult | `target_encode`（须折内） |
| 集成 | ✅ | House | `ModelEnsemble` |
| 业务可追溯流水线 | ✅ | business_pipeline | `BusinessPipeline` |
| 假设检验（t / χ² / ANOVA） | ✅ | 案例 14 | ❌ 无统一接口 |
| A/B 试验 / 功效 / 效应量 | ✅ | 案例 13 | ❌ 无试验设计 API |
| 贝叶斯 A/B / 先验敏感性 | ✅ | **并入案例 13** | ❌（共轭解析；pymc 可选） |
| 时序分解 / 平稳性 / 简单预测 | ✅ | 案例 15 | ❌ 无分解封装 |
| 缺期对齐 / 同比（零基数/未完月） | ✅ | **并入案例 15** | ❌ 业务 D04 深水区 |
| 生存 / 流失（KM） | ✅ | 案例 16 | ❌ 无生存模块 |
| 文本 EDA / 词袋特征 | ✅ | 案例 17 | `extract_text_features`（浅） |
| 异常检测（无标签） | ✅ | 案例 18 | ❌ 无 IsolationForest 封装 |
| 关联规则 / 购物篮 | ✅ | 案例 19 | ❌ 宜放 extras |
| 因果推断 / DiD 入门 | ✅ | 案例 20 | ❌ 无因果模块（刻意） |
| 多表关联膨胀 / 三表对账 | ✅ | **案例 21** | ❌ D02 深水区 S2 |
| 推荐 / 排序指标（NDCG） | ✅ | **案例 22** | ❌ 无 RankingMetrics |
| 校准曲线 / Brier / ECE | ✅ | **并入案例 04** | ❌ 无校准 API |
| 公平性系统化审计 | ✅ | **并入案例 10** | ❌ 无 GroupMetricsReport |

> 说明：上表「常见分析」教学覆盖已收口。仍可能扩展的专题（非阻塞）：贝叶斯层级模型全文、IV/断点回归、多臂 bandit、在线学习等。

## 2. 由案例反推的包优化优先级

| 优先级 | 优化项 | 依据案例 | 建议 |
| --- | --- | --- | --- |
| P0 | `StatsToolkit` / `ExperimentDesign` | 14 / 13 | 含贝叶斯门槛概率可选 |
| P0 | 业务 D02 多表关联 + 膨胀/对账日志 | **21** | 路线图 S2 |
| P0 | 业务 D04 缺期对齐 / 同比规则 | **15 增补** | 路线图 S3 |
| P1 | `TimeSeriesTools` / `SurvivalAnalyzer` / `AnomalyDetector` | 15 / 16 / 18 | |
| P1 | `RankingMetrics.ndcg_at_k` | **22** | |
| P1 | 概率校准 + ECE 报表 | **04 增补** | |
| P1 | `GroupMetricsReport` 公平性 | **10 增补** | |
| P2 | 文本中文分词、关联规则 extras | 17 / 19 | |

## 3. 案例编号（进阶目录）

> 文件名 `13_*.ipynb` 等与路线图勾选框「E13–E17」（导航/证据配套）**不是同一套编号**。

| ID | 文件 | 主题 |
| --- | --- | --- |
| 01–12 | 既有竞赛/表格基线 | 见 `examples/advanced/README.md` |
| 13 | `13_ab_testing.ipynb` | A/B + **贝叶斯增补** |
| 14 | `14_hypothesis_testing.ipynb` | t / χ² / ANOVA |
| 15 | `15_timeseries_forecast.ipynb` | 分解预测 + **同比边缘** |
| 16–20 | 生存 / 文本 / 异常 / 关联 / DiD | |
| 21 | `21_order_refund_reconciliation.ipynb` | 三表对账防翻倍 |
| 22 | `22_movielens_ndcg.ipynb` | NDCG@10 排序评估 |
| 04↑ | Breast 增补 | 校准 / ECE |
| 10↑ | Adult 增补 | 公平性交叉群体 |

## 4. 验收

```bash
PYTHONPATH=. python scripts/run_advanced_evidence.py
# 期望 passed=22
```
