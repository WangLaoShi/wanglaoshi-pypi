# 中文术语表

检索或复习时可用本页。Jupyter Book 搜索框也支持这些中文关键词。

```{glossary}
分层划分
  按类别比例划分训练/验证，避免某类在验证集中消失。小样本多分类常用。

数据泄露
  模型训练时用到了「预测时不可用」的信息，导致线下分数虚高、线上崩盘。

目标泄露
  特征中直接或间接含有目标信息（如用结果字段当特征）。

时间泄露
  用了未来信息（错误的时间划分、错误的滞后特征）。

分布泄露 / 分布漂移
  训练与测试特征分布显著不一致，泛化假设被破坏。

交叉验证（CV）
  多次划分评估泛化；时间/分组场景要用对应的 CV，而非盲目 KFold。

目标编码
  用目标均值等统计量编码类别；必须在训练折内拟合，并常需平滑。

提交文件（submission）
  按竞赛 sample 模板生成的预测表；本地校验 ID/列/类型，不应用测试标签刷分。

StageResult
  业务流水线单阶段结果：含 status、data、warnings、block_reasons 等。

passed / warning / blocked / skipped
  业务阶段状态：成功；可继续但有问题；阻断；因上游阻断等显式跳过。

deliverable
  交付索引中的「是否可正式交付」。上游 blocked 时应为 false。

PR-AUC
  Precision-Recall 曲线下面积；极端不均衡时往往比 ROC-AUC 更贴业务。

PCA
  主成分分析；降维前通常先标准化，且只能在训练集上拟合。

滞后特征
  用历史窗口构造特征；窗口只能包含预测时刻已发生的信息。

脱敏（redact）
  归档时把敏感字段替换为占位符，避免交付包泄漏隐私。

evidence / 证据编号
  结论与建议必须能指回的指标、图表或哈希等依据。

pending uplift
  行动建议里收益目标暂无依据，标记待定，避免拍脑袋承诺。
```

## 英文对照（便于读源码）

| 中文 | 常见英文 |
| --- | --- |
| 分层 | stratified |
| 泄露 | leakage |
| 基线 | baseline |
| 集成 | ensemble / stacking / blending |
| 特征工程 | feature engineering |
| 业务流水线 | business pipeline |
