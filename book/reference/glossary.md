# 中文术语表

进阶课号一律 **E01–E22**（勿与配套勾选 EM01–EM05 混淆）。

按主题分组；与 [API 参考](api.md)、[分析覆盖矩阵](analysis-coverage.md) 对照。包内符号用反引号标出。

## 工作流与交付

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 基线 | baseline | 可复现、可提交的第一版预测流水线 |
| 快速基线 | `QuickBaseline` | 一键切分、训练、评估、可选提交文件 |
| 就绪检查 | `ready` | 训练前结构体检（泄漏、目标、时间等） |
| 阶段结果 | `StageResult` | 流水线每步的结构化产出（指标、工件、日志） |
| 提交契约 | submission contract | 行序/ID/列名与评测机约定一致 |
| 提交生成器 | `SubmissionGenerator` | 按模板写出提交 CSV |
| 业务流水线 | `BusinessPipeline` | 可追溯多阶段业务分析管线 |
| 证据回放 | evidence replay | CI/脚本重跑 notebook 并落盘摘要 |

## 探索与质量

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 探索性分析 | EDA | 描述、缺失、分布、相关、可视化 |
| 数据剖析器 | `DataAnalyzer` | 一站式 EDA / 报告辅助 |
| 缺失机制 | MCAR/MAR/MNAR | 完全随机 / 依赖观测 / 依赖未观测 |
| 异常值 | outlier | 偏离主体分布的点；未必是错误 |
| 重复行 | duplicate | 完全或键重复；需业务判定去留 |
| 口径 | definition | 指标分子分母、时间窗、归属规则 |
| 一致性校验 | consistency check | 主键、外键、汇总勾稽 |

## 特征与泄漏

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 特征工程 | FE / `FeatureEngineer` | 衍生、编码、缩放、文本/时间特征 |
| 目标编码 | target encoding / `target_encode` | 用目标统计编码类别；**必须折内** |
| 时间特征 | datetime features | 从时间戳拆出的周期与滞后候选 |
| 滞后特征 | lag | 用历史值预测未来；禁止用「未来」 |
| 数据泄漏 | leakage | 训练时用不该在预测时可得的信息 |
| 时间泄漏 | time leakage | 用了预测时刻之后的信息 |
| 目标泄漏 | target leakage | 特征直接或间接含目标 |

## 验证与模型

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 交叉验证 | CV / `CrossValidator` | K 折等；估计泛化 |
| 分组交叉验证 | GroupKFold | 同组样本同进同出，防泄漏 |
| 时间序列切分 | TimeSeriesSplit | 训练窗严格早于验证窗 |
| 分层抽样 | stratified | 保持类别比例 |
| 过拟合 | overfitting | 训练集好、泛化差 |
| 欠拟合 | underfitting | 模型容量不足或特征不足 |
| 集成 | ensemble / `ModelEnsemble` | 多模型融合（平均、加权、堆叠） |
| 校准 | calibration | 预测概率与真实频率对齐 |

## 分类与不均衡

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 准确率 | accuracy | 正确数 / 总数；类别不平衡时易误导 |
| 精确率 | precision | 预测为正中真正为正的比例 |
| 召回率 | recall / TPR | 真实为正中被找出的比例 |
| F1 | F1-score | 精确率与召回的调和平均 |
| ROC-AUC | ROC AUC | 排序能力；不关心阈值 |
| PR-AUC | average precision | 不均衡下更敏感的排序/检出指标 |
| 混淆矩阵 | confusion matrix | TP/FP/FN/TN |
| 类别不平衡 | class imbalance | 正负样本比例悬殊 |
| `class_weight` | class_weight | 提高少数类误分代价 |

## 回归与变换

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| MAE | mean absolute error | 绝对误差均值；稳健、可解释 |
| RMSE | root mean squared error | 对大误差更敏感 |
| R² | coefficient of determination | 相对基线方差的解释比例 |
| 目标变换 | target transform | 如 log1p；逆变换后才可比业务单位 |
| 残差 | residual | 真实 − 预测；用于诊断异方差/结构 |

## 降维与相关

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 相关矩阵 | correlation | 线性相关强度；非因果 |
| 共线性 | multicollinearity | 特征高度相关，系数不稳 |
| PCA | `pca_analysis` | 主成分；方差方向，非监督 |
| 方差解释比 | explained variance ratio | 各主成分解释的方差比例 |

## 聚类

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| K-Means | k-means | 距离型划分聚类 |
| 轮廓系数 | silhouette | 簇内紧、簇间离的综合分 |
| 稳定性 | stability | 种子/子样扰动下簇是否可复现 |
| 业务可读标签 | persona | 用统计画像给簇命名，避免只报簇号 |

## 假设检验与试验

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 零假设 | H₀ | 默认「无效应 / 无差异」 |
| 备择假设 | H₁ | 研究者关心的差异或效应 |
| p 值 | p-value | 在 H₀ 下看到当前或更极端结果的概率 |
| 效应量 | effect size | 差异「有多大」（如 Cohen's d、提升百分点） |
| 功效 | power | 真实有效应时能检出的概率 |
| 一类错误 | Type I / α | 误拒真 H₀（假阳性） |
| 二类错误 | Type II / β | 未拒假 H₀（假阴性） |
| t 检验 | t-test | 均值差（正态/大样本近似） |
| 卡方检验 | χ² | 分类表独立性 / 拟合优度 |
| 方差分析 | ANOVA | 多组均值是否有差异 |
| A/B 试验 | A/B test | 随机分流比较两（多）版本 |
| 样本量 | sample size | 达目标功效所需观测数 |
| 多重比较 | multiple testing | 多次检验抬高假阳性；需校正 |

## 时间序列

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 趋势 | trend | 长期方向性变化 |
| 季节性 | seasonality | 固定周期波动 |
| 残差分量 | remainder | 分解后剩余项 |
| 平稳性 | stationarity | 均值/方差不随时间系统性漂移 |
| ADF | Augmented Dickey–Fuller | 单位根/平稳性检验之一 |
| 季节分解 | STL / classical decomp | 拆成趋势+季节+残差 |
| 前瞻偏差 | look-ahead bias | 用未来信息训练/评估 |

## 生存与流失

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 删失 | censoring | 观测结束时事件未发生 |
| 生存函数 | S(t) | t 时刻仍未事件的概率 |
| Kaplan–Meier | KM | 非参数生存曲线估计 |
| 风险比 | hazard ratio | 相对风险（模型语境） |
| 流失 | churn | 客户停止使用等事件 |

## 文本与异常

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 词袋 | bag-of-words | 词频向量，忽略语序 |
| TF-IDF | tf-idf | 词频 × 逆文档频率 |
| 文本特征 | `extract_text_features` | 长度、标点、简单统计等浅特征 |
| 异常检测 | anomaly detection | 无标签或弱标签下找离群模式 |
| Isolation Forest | iForest | 随机划分隔离异常点 |

## 因果与进阶（矩阵中多为缺口）

| 术语 | 英文 / 符号 | 含义 |
| --- | --- | --- |
| 相关 ≠ 因果 | association ≠ causation | 观测相关不能直接当干预效应 |
| 混杂 | confounding | 同时影响处理与结果的变量 |
| 双重差分 | DiD | 处理组与对照组的趋势差之差 |
| 工具变量 | IV | 外生变异识别因果 |
| 关联规则 | association rules | 购物篮「若买 A 则买 B」 |
| NDCG | NDCG | 排序质量指标（归一化折损累计增益） |
| 对称聚合 | symmetric aggregate | 一对多连接后防重复计数的度量语义 |
| 对账 | reconciliation | 汇总结果与源表勾稽一致 |
| 可靠性图 | reliability diagram | 预测概率分箱 vs 真实频率 |
| ECE | expected calibration error | 分箱加权的校准误差 |
| Brier 分数 | Brier score | 概率预测均方误差 |
| 同比 | YoY | 与去年同期对比；需处理缺期/零基数/未完月 |
| 先验敏感性 | prior sensitivity | 贝叶斯结论随先验变化的程度 |

## 竞赛与业务常用缩写

| 缩写 | 含义 |
| --- | --- | --- |
| OOF | Out-of-fold，折外预测 |
| LB | Leaderboard，排行榜分 |
| CV | Cross-validation |
| SMOTE 等 | 过采样；竞赛慎用，优先 `class_weight` |
| KPI | 关键业务指标 |
| ETL | 抽取-转换-加载 |

---

**维护：** 新增进阶课时同步补词；包 API 变更以 `docs/API_REFERENCE.md` 为准。
