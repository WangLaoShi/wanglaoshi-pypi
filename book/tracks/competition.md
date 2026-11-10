# 主线：竞赛工具链

目标：在本地完成 **「可信验证 → 合法提交」**，而不是只追求 Notebook 最后一行很高的分数。

## 1. 模块地图

| 类 | 干什么 | 精讲课 |
| --- | --- | --- |
| `SubmissionGenerator` | 按 sample 生成/校验提交 | {doc}`../lessons/submission_contracts` |
| `QuickBaseline` | 多模型快速对比 | 本页 §3 |
| `LeakageDetector` | 目标/时间/分布泄露提示 | {doc}`../lessons/leakage` |
| `CrossValidator` | 分层 / 时间 / 分组 CV | {doc}`../lessons/cv_and_splits` |
| `ModelEnsemble` | voting / stacking / blending | 进阶案例 House Prices |
| `FeatureImportanceAnalyzer` | 内置/排列重要性 | 解释须谨慎 |
| `ExperimentTracker` | 实验记录 JSON | 复现 |
| `HyperparameterOptimizer` | Optuna（需 `[optuna]`） | 预算写死 |
| `AdvancedFeatureEngineer` | 时间/文本/编码等 | {doc}`../lessons/feature_fit_discipline` |

```python
from wanglaoshi import ready
from wanglaoshi.CompetitionTools import (
    SubmissionGenerator,
    QuickBaseline,
    LeakageDetector,
    CrossValidator,
)
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer

ready()
```

## 2. 一条竞赛作业的标准剧本（90 分钟）

1. **读规则与 sample_submission**：目标列类型？指标？  
2. **EDA**（短）：缺失、类别不平衡、时间列？ → Analyzer 主线  
3. **定划分**：分层 / 时间 / 分组 → 写进报告第一节  
4. **泄露快检**：可疑特征先拿掉或改造法  
5. **QuickBaseline**：只比较「同划分」下的模型  
6. **特征**：折内 fit；记录种子与步骤  
7. **生成 + validate_submission**  
8. **误差分析**：谁总错？是否阈值问题？  

## 3. QuickBaseline 课堂片段

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from wanglaoshi.CompetitionTools import QuickBaseline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

qb = QuickBaseline(random_state=42)
# 具体方法名以当前版本为准，常见模式：
# result = qb.compare(X_tr, y_tr, X_va, y_va)
# print(result)
```

**教学重点：** 基线表是为了「先有可复现参照」，不是为了选一个去过拟合验证集。

## 4. 关键纪律（背下来）

1. 概率矩阵不要静默 `argmax`（除非显式 `class_labels`）  
2. 目标编码、选特征、标准化必须在 **训练折内** 拟合  
3. 无标签测试集只用于生成提交，不用于本地刷分（配套 EM03）  
4. 时间滞后特征不得引用「未来」目标（配套 EM04 / 课 E08）  
5. 校验失败禁止「手动改 CSV 对付平台」而不修生成逻辑  

## 5. 推荐学习顺序（路径 B 摘要）

| 课号 | 课 | 你必须能讲清的一句话 |
| --- | --- | --- |
| E01 | {doc}`../cases/01_iris` | 分层 + 标签提交 |
| E02 | {doc}`../cases/02_titanic` | 缺失处理与隔离 |
| E04 | {doc}`../cases/04_breast_cancer` | 缩放与阈值权衡 |
| E12 | {doc}`../cases/12_digit_recognizer` | 高维与错分 |
| E06 | {doc}`../cases/06_california_housing` | 分组 vs 随机 |
| … | 其余按 {doc}`../start/learning-path`（E01–E22） | — |

精讲建议先于案例：{doc}`../lessons/overview`。

## 6. 课后作业（可评分）

1. 用 Iris 或 Breast Cancer：三份提交（正确 / ID 打乱 / 少行），只保留校验通过的那份。  
2. 写 300 字：你的划分为何匹配「部署假设」。  
3. 用错误墙 {doc}`../lessons/common_mistakes` 自检，列出改动的 commit 或 diff 要点。  
