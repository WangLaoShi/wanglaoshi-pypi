"""第三轮课程加深：05 / 09 / 10 / 11 扩写至约 3 倍篇幅。

用法（在仓库根或 examples/advanced 下）：
  python examples/advanced/_deepen_curriculum_batch3.py

会覆盖同名 .ipynb，并在 stdout 打印扩写前后字符数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _deepen_all import code, env_cells, md, write  # noqa: E402

SLUGS = [
    "05_credit_card_fraud",
    "09_mall_customers",
    "10_adult_census",
    "11_house_prices",
]

# 教师导读：加长篇幅且不改可执行主线
TEACHER_E05 = """
## 教师导读：欺诈检测课堂节奏

**先打脸 Accuracy**：用 Dummy 全负类，让学生背出 fraud rate 与 dummy accuracy。  
**再上 PR-AUC**：同一验证集，比较模型表与 `average_precision_score` 排序差异。  
**时间切分**：画 train/valid 时间区间，禁止 `shuffle=True` 的 hold-out。  
**class_weight**：对照 None vs balanced 的 fraud recall。  
**SMOTE**：只展示折内代码，用一句话封住「全表 SMOTE 再划分」错误。  
**提交**：只演示 label；刻意不提 probability 类型。

**与 sklearn 文档差异**：我们强调 PR 主 ROC 辅，与 credit card 赛题社区习惯一致。  
**DEMO 数据**：无 CSV 时流程不变，_metrics 数值仅供演示 pipeline。
"""

TEACHER_E09 = """
## 教师导读：mall customers 课堂 90 分钟脚本

**0–10 min**：展示未缩放散点，问「哪个轴主导距离？」引导学生说 Income。  
**10–25 min**：`StandardScaler` + 重新散点；强调聚类是几何问题。  
**25–40 min**：肘部 + silhouette 双图；出现分歧时投票，不要求唯一 k。  
**40–55 min**：多 seed ARI；若 min ARI 低，现场调高 n_init 示范。  
**55–70 min**：簇画像 + 四象限叙事；讨论「簇不等于人格标签」的伦理。  
**70–85 min**：对比 Kaggle 分类赛：无 submission、无 PR-AUC。  
**85–90 min**：收口 wanglaoshi：`DataAnalyzer` 在 W1，聚类在 sklearn。

### 学生常问

- 「silhouette 为负是否不能用？」部分点本就处于簇边界，可接受。  
- 「能否用 DBSCAN？」可以，但要讲密度参数 ε；本案例用 KMeans 因可重复。  
- 「Genre 要 one-hot 吗？」可选侧车分析，不必进距离，除非业务明确要按性别分群。

### 与仓库其它案例链接

- E05：监督 + 极端不均衡 + PR-AUC + 提交  
- E10：监督 + 高基数 TE 泄漏  
- E11：回归 + log 目标 + ModelEnsemble  

本案例刻意**不**走提交流程，避免学生误认为所有数据集都要 `SubmissionGenerator`。
"""

TEACHER_E10 = """
## 教师导读：Adult 与泄漏「应激训练」

**教学目标**：学生结束后能用一句话说明「为什么全表 target_encode 是错的」。

**演示顺序**：先跑 OOF（第 5 节）建立正确直觉，再跑全量 TE（第 4 节）制造对比；若 leaky acc 未显著高于 safe，解释抽样与 QUICK 子样本方差，强调趋势而非绝对差。

**FE API 警示板**：`target_encode` 写在幻灯片上：`fit scope = 传入 df 行集合`。让学生口述：「我对 train_fold 调 encode，对 valid_fold map，对 test 只用 train 全量 map。」

**分组公平性**：性别切片 Accuracy 差异可能是样本量与先验不同；引入 TPR parity 概念但不展开公式，防课时爆炸。

**作业建议**：（1）手写 3 折 OOF TE 伪代码；（2）用 `FE.frequency_encode` 替换一列 TE 比较；（3）阅读 `SubmissionGenerator` 文档确认禁止 probability 提交。

**课时衔接**：承接 E05 的「指标不能糊涂」——这里是「特征不能泄漏」。
"""

TEACHER_E11 = """
## 教师导读：House Prices 集成最小闭环

**关键词**：缺失语义、log 目标、频率编码、blending。

**实验板**：左侧单模 RF，右侧 `ModelEnsemble.blending`；对比 hold-out RMSE(log) 与 MAE(price)。集成未必必胜，但要学会**报告两种尺度**。

**高基数课**：打印 Neighborhood 基数；让学生想象 One-Hot 列数。频率编码是 leaky-safe 入门，TE 留到 E10。

**Garage 缺失**：白板画「无车库 ⇒ NA 合法」；HasGarage_hint 列是思想实验，不追求 leaderboard 增益。

**stacking vs blending**：用 2 分钟口径——blending 快、易过拟合权重；stacking 折 OOF 更贵更稳。QUICK 课堂选 blending 保障 evidence。

**提交 ritual**：`expm1` → `prediction_type='label'` → `validate_submission`。强调竞赛列是美元，不是 log。
"""


def _chars(slug: str) -> int:
    p = ROOT / f"{slug}.ipynb"
    return len(p.read_text(encoding="utf-8")) if p.exists() else 0


def e05() -> None:
    write(
        "05_credit_card_fraud",
        [
            md(
                """# E05 Credit Card Fraud：时间划分、极端不均衡与 PR-AUC

> Issue #9 / Epic #6 / 路线图 E05

## 案例背景
信用卡欺诈检测是典型的**极度类别不均衡**二分类：正类（欺诈）往往 <1%。业务上漏掉一笔欺诈的代价远高于误杀正常交易，但模型的「准确率」在这种分布下几乎失去区分力——全预测为 0 也能得到 99%+ Accuracy。

## 你将学到
1. **评估指标**：PR-AUC（Average Precision）为何比 ROC-AUC 更贴欺诈场景；Accuracy 陷阱的可视化
2. **验证策略**：按 `Time` 排序切分，禁止随机打乱（避免用「未来」训练过去）
3. **不平衡处理**：`class_weight='balanced'` 作为可上线基线；**SMOTE 仅应在训练折内**使用的故事线
4. **wanglaoshi**：`ready` → `DataAnalyzer` → `QuickBaseline` → `SubmissionGenerator`（`prediction_type='label'`，禁止 `'probability'`）

## 理论速览：PR 曲线 vs ROC 曲线
- **ROC**：横轴 FPR、纵轴 TPR；当负类极多时，FPR 仍可能很低，曲线「看起来很好」
- **PR**：横轴 Recall、纵轴 Precision；正类稀少时，Precision 对阈值极敏感，更反映排序质量
- 本 Notebook **主指标：PR-AUC**；ROC-AUC 仅作对照，不作为选型依据

## 数据
- 本地：`data/creditcard.csv`（Kaggle Credit Card Fraud）
- 缺失：合成 DEMO 不均衡数据（流程等价，结论一致）
"""
            ),
            *env_cells(),
            md("## 0. 环境确认"),
            code(
                """
import pandas as pd
import numpy as np
print('pandas', pd.__version__, 'numpy', np.__version__)
print('QUICK =', QUICK, 'RANDOM_STATE =', RANDOM_STATE)
"""
            ),
            md("## 1. 加载真实数据或合成 DEMO"),
            code(
                """
from pathlib import Path
from sklearn.datasets import make_classification

path = Path('../../data/creditcard.csv')
DEMO = False
if path.exists():
    df = pd.read_csv(path)
    if QUICK and len(df) > 80000:
        pos = df[df['Class'] == 1]
        neg = df[df['Class'] == 0].sample(
            min(20000, (df['Class'] == 0).sum()), random_state=RANDOM_STATE
        )
        df = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_STATE)
    print('loaded', path.resolve(), df.shape)
    print(df['Class'].value_counts(normalize=True))
else:
    DEMO = True
    print('未找到 creditcard.csv，使用合成不均衡 DEMO 数据')
    X_syn, y_syn = make_classification(
        n_samples=8000 if QUICK else 30000,
        n_features=20,
        n_informative=8,
        weights=[0.985, 0.015],
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X_syn, columns=[f'V{i}' for i in range(X_syn.shape[1])])
    df['Time'] = np.arange(len(df))
    df['Amount'] = np.abs(np.random.RandomState(RANDOM_STATE).normal(50, 30, len(df)))
    df['Class'] = y_syn
    print(df['Class'].value_counts(normalize=True))
df.head(3)
"""
            ),
            md(
                """## 2. EDA：类别比例、金额分布与「准确率幻觉」

极端不均衡时，应先回答：**随机猜全为负，指标是多少？** 这给出任何模型的下限。
"""
            ),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.get_basic_info())
display(analyzer.basic_statistics().head(12))
fraud_rate = float(df['Class'].mean())
print('fraud rate =', fraud_rate)
print('dummy all-zero accuracy ≈', 1 - fraud_rate)
if 'Amount' in df.columns:
    print(df.groupby('Class')['Amount'].agg(['mean', 'median', 'max']))
"""
            ),
            md(
                """### 2.1 理论：为何 Accuracy 误导

设欺诈率 \(p \approx 0.002\)。分类器永远输出「正常」可得 Accuracy ≈ 99.8%，但 Recall=0，业务价值为零。因此本案例：
- **主看**：PR-AUC、`classification_report` 中的 precision/recall
- **辅看**：ROC-AUC（了解排序，但不作为唯一依据）
"""
            ),
            code(
                """
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

feature_cols = [c for c in df.columns if c not in ('Class', 'Time')]
work = df.sort_values('Time').reset_index(drop=True)
split = int(len(work) * 0.8)
train, valid = work.iloc[:split], work.iloc[split:]
Xtr, Xva = train[feature_cols], valid[feature_cols]
ytr, yva = train['Class'], valid['Class']

dummy = DummyClassifier(strategy='constant', constant=0)
dummy.fit(Xtr, ytr)
pred_d = dummy.predict(Xva)
print('Dummy Accuracy', accuracy_score(yva, pred_d))
print('Dummy PR-AUC   ', average_precision_score(yva, pred_d))
"""
            ),
            md(
                """## 3. 时间切分（禁止随机打乱）

欺诈数据按交易时间有序。随机 `train_test_split` 会让训练集混入验证时段之后的信息，离线指标虚高。  
**正确做法**：按 `Time` 排序，前 80% 训练、后 20% 验证（模拟「用历史预测未来」）。
"""
            ),
            code(
                """
print('train fraud rate', ytr.mean(), '| valid fraud rate', yva.mean())
print('time train', train['Time'].min(), '->', train['Time'].max())
print('time valid', valid['Time'].min(), '->', valid['Time'].max())
print('n_train', len(train), 'n_valid', len(valid))
"""
            ),
            md("## 4. 特征缩放（PCA 匿名特征 + Amount）"),
            code(
                """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xva_s = pd.DataFrame(scaler.transform(Xva), columns=Xva.columns, index=Xva.index)
Xtr_s.head(2)
"""
            ),
            md(
                """## 5. 基线：`class_weight='balanced'` 与 PR-AUC 选型

`class_weight` 在损失函数里加大正类权重，**不合成新样本**，适合作为生产默认可解释基线。  
`QuickBaseline` 跑多模型时，我们**用手工 PR-AUC 表**重排，而不是默认 Accuracy 排序。
"""
            ),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_s, ytr, Xva_s, yva, task_type='classification')
comparison = baseline.compare_models(results, top_n=8)
display(comparison)

rows = []
for name, pack in results.items():
    model = pack['model']
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(Xva_s)[:, 1]
    else:
        proba = np.asarray(pack['predictions'], dtype=float)
    rows.append({
        'Model': name,
        'PR-AUC': average_precision_score(yva, proba),
        'ROC-AUC': roc_auc_score(yva, proba),
    })
metric_df = pd.DataFrame(rows).sort_values('PR-AUC', ascending=False).reset_index(drop=True)
print('=== 按 PR-AUC 排序（主指标）===')
display(metric_df)
best = metric_df.iloc[0]['Model']
model = results[best]['model']
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(Xva_s)[:, 1]
else:
    proba = np.asarray(results[best]['predictions'], dtype=float)
pred = (proba >= 0.5).astype(int)
print(classification_report(yva, pred, zero_division=0))
print('选中模型', best, 'PR-AUC', average_precision_score(yva, proba))
"""
            ),
            md(
                """## 6. 显式对比：`class_weight` 与默认权重

下面只换 `LogisticRegression` 的 `class_weight`，观察 **PR-AUC / Recall** 变化（Accuracy 可能下降，这是预期现象）。
"""
            ),
            code(
                """
from sklearn.pipeline import Pipeline

for cw in [None, 'balanced']:
    pipe = Pipeline([
        ('clf', LogisticRegression(max_iter=800, random_state=RANDOM_STATE, class_weight=cw)),
    ])
    pipe.fit(Xtr_s, ytr)
    p = pipe.predict_proba(Xva_s)[:, 1]
    pred_pos = (p >= 0.5)
    fraud_recall = ((yva.to_numpy() == 1) & pred_pos).sum() / max(1, (yva == 1).sum())
    print('class_weight=', cw,
          'PR-AUC', round(average_precision_score(yva, p), 4),
          'ROC-AUC', round(roc_auc_score(yva, p), 4),
          'fraud recall@0.5', round(fraud_recall, 4))
"""
            ),
            md(
                """## 7. SMOTE 叙事：只在训练折内重采样

**错误**：对整表 SMOTE 后再划分 → 验证集与训练集通过合成样本「泄漏」相似点。  
**正确**：在每个 CV 折里，仅对 **训练折** `fit_resample`，验证折保持原始分布。

本段在**时间有序验证集不变**的前提下，用 `StratifiedKFold` 在训练部分演示折内 SMOTE 的 PR-AUC（教学用途；上线仍以 class_weight + 阈值为主）。
"""
            ),
            code(
                """
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

n_splits = 3 if QUICK else 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
oof_proba = np.zeros(len(Xtr_s))
clf = LogisticRegression(max_iter=600, random_state=RANDOM_STATE, class_weight='balanced')

for fold, (tr_idx, va_idx) in enumerate(cv.split(Xtr_s, ytr)):
    X_f_tr = Xtr_s.iloc[tr_idx]
    y_f_tr = ytr.iloc[tr_idx]
    X_f_va = Xtr_s.iloc[va_idx]
    smote = SMOTE(random_state=RANDOM_STATE + fold, k_neighbors=3)
    try:
        X_res, y_res = smote.fit_resample(X_f_tr, y_f_tr)
    except ValueError:
        X_res, y_res = X_f_tr, y_f_tr
        print(f'fold {fold}: SMOTE skip（正类过少）')
    clf.fit(X_res, y_res)
    oof_proba[va_idx] = clf.predict_proba(X_f_va)[:, 1]

print('OOF PR-AUC (train, fold-safe SMOTE)', average_precision_score(ytr, oof_proba))
print('Hold-out valid PR-AUC (no SMOTE on valid)', average_precision_score(yva, proba))
"""
            ),
            md(
                r"""## 7.1 可视化：PR 曲线 vs ROC 曲线（同一模型）

在同一张图上对比两种曲线，强化「Accuracy/ROC 好看但 PR 反映排序」的直觉。
- PR 曲线在正类稀少时通常**更陡、更敏感**
- 选型以 **Average Precision（PR-AUC）** 为准
"""
            ),
            code(
                """
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

fpr, tpr, _ = roc_curve(yva, proba)
prec_c, rec_c, _ = precision_recall_curve(yva, proba)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(rec_c, prec_c, label=f'AP={average_precision_score(yva, proba):.3f}')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('PR curve (主)')
axes[0].axhline(yva.mean(), color='gray', ls='--', label='base rate')
axes[0].legend()
axes[1].plot(fpr, tpr, label=f'AUC={roc_auc_score(yva, proba):.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].set_title('ROC curve (辅)')
axes[1].legend()
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """## 7.2 金额特征：log1p(Amount) 与欺诈

`Amount` 长尾分布常见。取对数后缩放，有时能提升 PR-AUC（非保证，需实验验证）。
"""
            ),
            code(
                """
if 'Amount' in train.columns:
    amt_tr = np.log1p(train['Amount'].astype(float))
    amt_va = np.log1p(valid['Amount'].astype(float))
    Xtr_a = Xtr_s.copy()
    Xva_a = Xva_s.copy()
    Xtr_a['Amount_log'] = StandardScaler().fit_transform(amt_tr.values.reshape(-1, 1))
    Xva_a['Amount_log'] = StandardScaler().fit(amt_tr.values.reshape(-1, 1)).transform(
        amt_va.values.reshape(-1, 1)
    )
    lr = LogisticRegression(max_iter=600, random_state=RANDOM_STATE, class_weight='balanced')
    lr.fit(Xtr_a, ytr)
    p_amt = lr.predict_proba(Xva_a)[:, 1]
    print('PR-AUC with Amount_log', average_precision_score(yva, p_amt))
else:
    print('无 Amount 列，跳过金额工程')
"""
            ),
            md(
                """### 7.3 理论补课：不均衡学习菜单

| 方法 | 作用位置 | 上线友好度 | 本 Notebook |
|------|----------|------------|-------------|
| `class_weight` | 损失函数 | 高 | ✅ 基线 |
| 阈值调整 | 推断 | 高 | ✅ 第 8 节 |
| SMOTE | 仅训练折 | 中（需严格 CV） | ✅ 第 7 节 |
| 欠采样 | 训练 | 低（丢信息） | 未展开 |
| 异常检测视角 | 特征空间 | 视场景 | 拓展阅读 |

欺诈系统常见组合：**时间验证 + class_weight + PR-AUC 选型 + 业务阈值**。
"""
            ),
            md("## 8. 阈值：固定 Precision 下最大化 Recall"),
            code(
                """
from sklearn.metrics import precision_recall_curve

prec, rec, ths = precision_recall_curve(yva, proba)
target_prec = 0.5
mask = prec[:-1] >= target_prec
if mask.any():
    idx = np.argmax(rec[:-1][mask])
    thr = ths[mask][idx]
    print('chosen thr', thr, 'precision', prec[:-1][mask][idx], 'recall', rec[:-1][mask][idx])
else:
    print('无 precision>=0.5 的阈值，请放宽约束或改盯 PR-AUC')
"""
            ),
            md("## 9. 模拟提交（标签，非概率）"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/05_credit_card_fraud')
out.mkdir(parents=True, exist_ok=True)
pred_label = (proba >= 0.5).astype(int)
sample = pd.DataFrame({'Id': np.arange(len(pred_label)), 'Class': yva.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_label,
    str(out / 'sample_submission.csv'),
    str(out / 'submission.csv'),
    prediction_type='label',
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
print('DEMO mode =', DEMO)
sub.head(3)
"""
            ),
            md(
                """## 10. 小结与检查清单
- [x] 时间切分而非随机划分
- [x] **主指标 PR-AUC**，对照 ROC-AUC / Dummy 下限
- [x] `class_weight='balanced'` 基线
- [x] SMOTE 仅在 CV 训练折内（叙事 + 代码）
- [x] 提交 `prediction_type='label'`
- [ ] 上线：监控 precision@流量、人工复核队列
- [ ] 成本敏感阈值（漏报 vs 误报货币化）
"""
            ),
            md(
                """## 附录 A：欺诈检测实验设计备忘

**离线实验叙事**：先固定时间切分，再在同一验证集上比较特征、模型、重采样。任何改动若重新划分时间窗，必须与历史实验标注版本号，否则 PR-AUC 不可比。

**为何不用 Accuracy 选型**：正类占比 \(p\) 小时，Accuracy 的 Hessian 对正类极不敏感；Optimizer 会滑向「全负类」鞍点。报告 Accuracy 可作为业务沟通，但不宜写入模型 gates。

**ROC vs PR 再强调**：ROC 积分对负类采样不敏感；PR 积分直接惩罚「把正类排在后面」的行为。Kaggle 类似赛题常见 leaderboard 用 AUC，但银行风控内部更常看 KS、PR-AUC、Lorenz 曲线。

**class_weight 机制**：相当于对正负样本的损失乘系数，不改变训练集行数。优点是流程与生产一致；缺点是若 \(p\) 极小，权重比值大，训练方差升高，应配合早停与正则。

**SMOTE 折内故事**：合成样本只在训练折生成，验证折永远是原始分布，这是唯一 honest 的 PR 曲线。切勿 `fit_resample` 在全训练集后再切验证——那会让验证点落在合成邻域里。

**wanglaoshi API 路径**：`ready` 统一环境 → `DataAnalyzer` 看 fraud rate → `QuickBaseline` 多模型初筛（记得按 PR-AUC 重排）→ `SubmissionGenerator` 输出离散 `Class` 标签。禁止 `prediction_type='probability'` 伪造概率提交。

**Amount / Time**：金额对数化、时间周期特征（小时/星期）是常见增益点；PCA 分量 V1–V28 已不可逆，解释性弱，可用 SHAP 在验证折上抽查。
"""
            ),
            md(
                """## 附录 B：阈值与运营成本（定性）

设：漏掉一笔欺诈损失 \(L_{fn}\)，误杀一笔正常交易成本 \(L_{fp}\)。  
最优点满足 \( \frac{p(\text{欺诈}|s)}{p(\text{正常}|s)} \) 与 \(\frac{L_{fp}}{L_{fn}}\) 的权衡。  
当 \(L_{fn} \gg L_{fp}\) 时，应降低阈值、抬高召回，即使 Precision 下降。  
本 Notebook 用「Precision ≥ 0.5 下最大 Recall」只是演示接口，上线阈值须与风控委员会签字版本一致。
"""
            ),
            md("## 附录 C：交叉验证 PR-AUC 代码模板（时间序 hold-out 已固定）"),
            code(
                """
from sklearn.model_selection import cross_val_score

clf_cv = LogisticRegression(max_iter=500, class_weight='balanced', random_state=RANDOM_STATE)
pr_scores = cross_val_score(
    clf_cv, Xtr_s, ytr, cv=3 if QUICK else 5,
    scoring='average_precision', n_jobs=-1,
)
print('CV PR-AUC mean±std', pr_scores.mean(), pr_scores.std())
"""
            ),
            md(TEACHER_E05),
        ],
    )


def e09() -> None:
    write(
        "09_mall_customers",
        [
            md(
                """# E09 Mall Customers：缩放、聚类数与稳定性

> Issue #9 / Epic #6 / 路线图 E09

## 案例定位
商场客户细分是**无监督聚类**经典题：没有「标准答案标签」，也没有 Kaggle 式提交排名。  
目标是：**用可重复的簇结构支撑运营动作**（高净值低消费 vs 低净值高消费等），而不是 chasing leaderboard。

## 你将学到
1. 聚类前必须做**特征缩放**（Income / Score 量纲不同）
2. **选 k**：肘部法（inertia）+ 轮廓系数（silhouette）互相印证
3. **稳定性**：换 `random_state` 时，标签置换不应剧烈变化（用 ARI 量化）
4. **wanglaoshi**：`DataAnalyzer` 做分布摸底；本案例**不做** `SubmissionGenerator`（非竞赛提交场景）

## 理论：K-Means 的假设与局限
- 假设：球形簇、近似等方差；对 Income-Score 散点通常尚可
- `n_init`：多次初始化取最优 inertia，减轻局部最优
- 稳定性：客户细分要可解释 → 若换种子簇大幅重排，业务难以落地

## 数据
- 本地：`data/Mall_Customers.csv`
- 缺失：合成客户画像 DEMO
"""
            ),
            *env_cells(),
            md("## 1. 加载或合成"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/Mall_Customers.csv')
DEMO = False
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    DEMO = True
    rng = np.random.RandomState(RANDOM_STATE)
    n = 400 if QUICK else 800
    df = pd.DataFrame({
        'CustomerID': np.arange(1, n + 1),
        'Genre': rng.choice(['Male', 'Female'], n),
        'Age': rng.randint(18, 70, n),
        'Annual Income (k$)': rng.normal(60, 25, n).clip(15, 140),
        'Spending Score (1-100)': rng.normal(50, 25, n).clip(1, 100),
    })
    print('DEMO customers', df.shape)
df.head()
"""
            ),
            md("## 2. EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

analyzer = DataAnalyzer(df)
display(analyzer.get_basic_info())
display(analyzer.basic_statistics())
"""
            ),
            md("## 3. 选数值特征与散点直觉"),
            code(
                """
import matplotlib.pyplot as plt

cols = []
for name in ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']:
    if name in df.columns:
        cols.append(name)
if len(cols) < 2:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if 'id' not in c.lower()][:3]

X = df[cols].astype(float)
print('cluster features:', cols)
if len(cols) >= 2:
    plt.figure(figsize=(5, 4))
    plt.scatter(X[cols[0]], X[cols[1]], alpha=0.5, s=20)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title('Income vs Spending（未缩放）')
    plt.tight_layout()
    plt.show()
"""
            ),
            md(
                """## 4. 缩放 + 搜 k（肘部 + Silhouette）

轮廓系数 \(s \in [-1,1]\)：越接近 1 表示点与同簇更紧、与他簇更疏。  
肘部法看 inertia 下降放缓的 k；两者不一致时，优先结合**业务可解释性**定 k。
"""
            ),
            code(
                """
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

ks = range(2, 10 if not QUICK else 8)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lab = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, lab))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(list(ks), inertias, marker='o')
ax[0].set_title('Elbow (inertia)')
ax[0].set_xlabel('k')
ax[1].plot(list(ks), sils, marker='o')
ax[1].set_title('Silhouette')
ax[1].set_xlabel('k')
plt.tight_layout()
plt.show()
best_k = list(ks)[int(np.argmax(sils))]
print('best_k by silhouette', best_k, 'score', round(max(sils), 4))
"""
            ),
            md(
                """## 5. 多随机种子稳定性（ARI）

**Adjusted Rand Index (ARI)** 衡量两套标签在忽略置换意义下的一致性，1 为完全一致。  
对固定 `best_k`，扫描多个 `random_state`；若 ARI 均值过低，说明解不稳定，不宜直接用于运营命名。
"""
            ),
            code(
                """
from sklearn.metrics import adjusted_rand_score

base = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
seeds = list(range(12 if QUICK else 30))
aris = []
for seed in seeds:
    lab = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit_predict(Xs)
    aris.append(adjusted_rand_score(base, lab))
print('ARI vs base seed: mean', round(np.mean(aris), 4), 'min', round(np.min(aris), 4))
print('sample ARI', aris[:8])
"""
            ),
            md("## 6. 最终簇标签与画像表"),
            code(
                """
labels = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
profile = df.copy()
profile['cluster'] = labels
centers = profile.groupby('cluster')[cols].mean()
display(centers)
if len(cols) >= 2:
    plt.figure(figsize=(5, 4))
    for c in range(best_k):
        m = labels == c
        plt.scatter(Xs[m, 0], Xs[m, 1], alpha=0.55, s=18, label=f'cluster {c}')
    plt.xlabel(cols[0] + ' (scaled)')
    plt.ylabel(cols[1] + ' (scaled)')
    plt.legend()
    plt.title(f'KMeans k={best_k}')
    plt.tight_layout()
    plt.show()
"""
            ),
            md("## 6.1 未缩放 vs 缩放：轮廓系数对照"),
            code(
                """
km_raw = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(X.values)
km_scl = labels
print('silhouette raw', round(silhouette_score(X.values, km_raw), 4))
print('silhouette scaled', round(silhouette_score(Xs, km_scl), 4))
"""
            ),
            md("## 6.2 类别变量 `Genre` 与簇的列联"),
            code(
                """
if 'Genre' in df.columns:
    ct = pd.crosstab(profile['cluster'], profile['Genre'], normalize='index')
    display(ct)
else:
    print('无 Genre 列')
"""
            ),
            md(
                """## 6.3 理论：ARI 与种子敏感性

ARI 接近 1 说明簇划分在标签置换意义下稳定。  
若 `min(ARI) < 0.7`，建议增大 `n_init`、尝试 GMM（软分配）或与业务专家共创 k。  
**本案例不是提交竞赛**：无需 `SubmissionGenerator`；交付物是「可解释的 `CustomerID → cluster` 映射表」。
"""
            ),
            md(
                """## 6.4 拓展阅读

- **Davies–Bouldin**：簇内散 / 簇间距，越小越好（与 silhouette 互补）
- **层次聚类**：小 n 时可画树状图辅助选 k
- **SCatter 经典四象限**：Income × Spending 平面上的「土豪 / 潜力 / 谨慎 / 普通」叙事
"""
            ),
            md(
                """## 7. 与竞赛流程的区别

| 维度 | 本案例 | 典型 Kaggle 分类 |
|------|--------|------------------|
| 标签 | 无真值簇标签 | 有监督 y |
| 指标 | 轮廓/ARI/业务解读 | AUC/LogLoss |
| 提交 | **不需要** | SubmissionGenerator |

若要将簇用于_campaign_，导出 `CustomerID, cluster` 即可，无需概率型预测文件。
"""
            ),
            md(
                """## 8. 检查清单
- [x] 缩放后聚类
- [x] elbow + silhouette 选 k
- [x] 多 seed ARI 稳定性
- [x] 簇均值画像
- [ ] 业务命名（如「理性消费者」「冲动型」）
- [ ] 与 A/B 活动对照验证
"""
            ),
            md(
                """## 附录：聚类实战 Q&A（中文）

**Q：k 越大越好吗？**  
A：否。k 增大 inertia 单调降，但 silhouette 常先升后降。要在可解释性与拟合间折中。

**Q：为何不用 Accuracy？**  
A：无标签场景根本没有 Accuracy。若用外部分群标签评估，那是另一套监督指标，不是本节重点。

**Q：K-Means 对离群点？**  
A：敏感。营销金额尖峰客户可单独做 anomaly 层，再对其子集聚类。

**Q：稳定性多重要？**  
A：运营要固定话术与 coupon 策略，ARI<0.8 的解难以年度复用。可多 seed 投票或选 medoid 代表点。

**Q：与 wanglaoshi 的关系？**  
A：`DataAnalyzer` 做分布与缺失；聚类本身用 sklearn。本案例 deliberately 不用 `SubmissionGenerator`，强调「洞察交付」而非排名。

**Q：Gender/年龄要进模型吗？**  
A：伦理与法规允许时可用于画像；若仅作切片报告，可不加入距离计算，避免敏感属性驱动分群。
"""
            ),
            md("## 附录 B：导出簇分配（非竞赛提交）"),
            code(
                """
from pathlib import Path
export = profile[['CustomerID', 'cluster'] + cols] if 'CustomerID' in profile.columns else profile[['cluster'] + cols]
out_path = Path('outputs/09_mall_customers')
out_path.mkdir(parents=True, exist_ok=True)
export.head(10).to_csv(out_path / 'cluster_assignments_preview.csv', index=False)
print('wrote', out_path / 'cluster_assignments_preview.csv', 'rows preview', len(export.head(10)))
"""
            ),
            md(
                """## 附录 C：课堂讨论题

1. 若 Marketing 坚持 k=6 但 silhouette 峰值在 k=4，你如何沟通？  
2. ARI 很高但簇中心业务难命名，是否仍上线？  
3. 新增线上特征（会员等级）后，旧簇映射如何迁移？  
4. 对比 K-Means 与 Gaussian Mixture 的软分配优劣。
"""
            ),
            md(
                """## 读本：客户细分项目里程碑（非竞赛）

| 阶段 | 产出 | 成功标准 |
|------|------|----------|
| W1 | 清洗 + EDA | 无重复 CustomerID，缺失率表 |
| W2 | 缩放 + 选 k | silhouette≥0.25 且业务认可 k |
| W3 | 稳定性 | ARI mean≥0.85 |
| W4 | 画像 + 命名 | 每簇一句话价值主张 |
| W5 | 试点 | 小规模 campaign 对照组 uplift |

聚类 ≠ 分类：没有唯一真值标签，因此**不要为了虚构 leaderboard 造 submission 文件**。wanglaoshi 的价值在 `DataAnalyzer` 帮你在 W1 快速发现量纲与缺失；后续距离与聚类仍用 sklearn 标准栈。

### 距离、尺度与特征权重

欧氏距离对量纲敏感：Income 以千美元计、Score 0–100，若不 `StandardScaler`，Income 将主导距离。  
若业务认为 Spending 更重要，可在缩放后对列乘权重（例如 `Xs[:,1] *= 1.5`），但须在文档写明，否则年度复现困难。

### 从簇到行动

常见四象限叙事（Income × Spending）：高收高花「喜爱型」、高收低花「谨慎型」、低收高花「冲动型」、低收低花「普通型」。数字标签需经 centroid 验证，不可生搬硬套。
"""
            ),
            md(TEACHER_E09),
        ],
    )


def e10() -> None:
    write(
        "10_adult_census",
        [
            md(
                """# E10 Adult Census：折内目标编码与分组表现

> Issue #9 / Epic #6 / 路线图 E10

## 案例背景
Adult 收入预测含大量**高基数类别**（职业、教育、籍贯等）。One-Hot 会稀疏爆炸；**目标编码（Target Encoding）** 用目标均值压缩类别，但必须 **fold-safe**，否则验证集信息经全表统计泄漏进特征，离线 AUC 虚高。

## 你将学到
1. **泄漏演示**：全量 `y` 做 TE vs **OOF（Out-of-Fold）** 正确做法
2. **wanglaoshi API**：`FeatureEngineering.AdvancedFeatureEngineer.target_encode` — 便捷但**默认在传入的整张表上拟合**，课堂演示后必须强调上线要折内/嵌 Pipeline
3. 分组指标：按 `sex` 等切片看 Accuracy 差异（公平性分析入口）
4. `SubmissionGenerator`：`prediction_type='label'`

## 理论：目标编码与平滑
对类别 \(c\)，编码 \(\hat{\mu}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}\)（\(m\) 为 smoothing）。  
**泄漏来源**：若用含验证行在内的全表估计 \(\bar{y}_c\)，验证行的标签已通过统计量进入特征。

## 数据
- 本地：`data/adult.csv`
- 回退：OpenML `adult`（version 2）
"""
            ),
            *env_cells(),
            md("## 1. 加载 Adult"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/adult.csv')
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    from sklearn.datasets import fetch_openml
    raw = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    df = raw.frame.copy()
    print('OpenML', df.shape)

target_col = None
for c in ['income', 'class', 'target', '>50K']:
    if c in df.columns:
        target_col = c
        break
assert target_col, list(df.columns)

if QUICK and len(df) > 4000:
    df = df.sample(4000, random_state=RANDOM_STATE).reset_index(drop=True)
    print('QUICK sampled', df.shape)

print('target', target_col, df[target_col].value_counts())
df.head(3)
"""
            ),
            md("## 2. EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.missing_value_analysis().head(15))
display(analyzer.basic_statistics().head(12))
"""
            ),
            md("## 3. 构造二值目标与特征列表"),
            code(
                """
from sklearn.preprocessing import LabelEncoder

work = df.copy()
y_raw = work[target_col].astype(str)
y = (y_raw.str.contains('>50K')).astype(int) if y_raw.str.contains('>50K').any() else LabelEncoder().fit_transform(y_raw)
work = work.drop(columns=[target_col])
cat_cols = work.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
print('n_cat', len(cat_cols), 'n_num', len(num_cols))
print('sample cat', cat_cols[:6])
"""
            ),
            md(
                """## 4. ❌ 错误示范：全量目标编码（泄漏）

下面故意用 **train+valid 全部行的 y** 估计类别均值，再划分。  
仅用于对比：你会看到「过于乐观」的近似指标倾向（本段用简单相关作泄漏直觉）。
"""
            ),
            code(
                """
from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(
    work, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
te_cols = cat_cols[:6]

def naive_full_te(frame, y_all, cols):
    out = frame.copy()
    for c in cols:
        means = y_all.groupby(frame[c].astype(str)).mean()
    return means

# 全表拼回去再编码（错误）
full = pd.concat([Xtr, Xte])
y_full = pd.concat([pd.Series(ytr, index=Xtr.index), pd.Series(yte, index=Xte.index)])
leak_tr, leak_te = Xtr.copy(), Xte.copy()
for c in te_cols:
    means = y_full.groupby(full[c].astype(str)).mean()
    gmean = y_full.mean()
    leak_tr[c + '_te'] = Xtr[c].astype(str).map(means).fillna(gmean)
    leak_te[c + '_te'] = Xte[c].astype(str).map(means).fillna(gmean)
feat_leak = num_cols + [c + '_te' for c in te_cols]
leak_tr_f = leak_tr[feat_leak].apply(pd.to_numeric, errors='coerce').fillna(0)
leak_te_f = leak_te[feat_leak].apply(pd.to_numeric, errors='coerce').fillna(0)
print('leakage feature shape', leak_tr_f.shape)
"""
            ),
            md(
                """## 5. ✅ OOF 目标编码（折内拟合）

在训练集内 `KFold`：每折只用训练子集的 y 估计映射，对验证子集 **transform**。  
这是对 Kaggle/CV 友好的标准 TE 流程。
"""
            ),
            code(
                """
from sklearn.model_selection import KFold

def oof_target_encode(X_train, y_train, cols, n_splits=5, seed=42):
    Xo = X_train.copy()
    y_s = pd.Series(y_train, index=X_train.index)
    gmean = y_s.mean()
    n_splits = 3 if QUICK else n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in cols:
        col_name = c + '_te_oof'
        Xo[col_name] = gmean
        for tr_i, va_i in kf.split(Xo):
            means = y_s.iloc[tr_i].groupby(X_train[c].astype(str).iloc[tr_i]).mean()
            Xo.iloc[va_i, Xo.columns.get_loc(col_name)] = (
                X_train[c].astype(str).iloc[va_i].map(means).fillna(gmean)
            )
    return Xo

def apply_te_from_train(train, valid, y_train, cols):
    tr = train.copy()
    va = valid.copy()
    y_s = pd.Series(y_train, index=train.index)
    gmean = y_s.mean()
    for c in cols:
        means = y_s.groupby(train[c].astype(str)).mean()
        tr[c + '_te'] = train[c].astype(str).map(means).fillna(gmean)
        va[c + '_te'] = valid[c].astype(str).map(means).fillna(gmean)
    return tr, va

Xtr_oof = oof_target_encode(Xtr, ytr, te_cols, seed=RANDOM_STATE)
Xtr_te, Xte_te = apply_te_from_train(Xtr, Xte, ytr, te_cols)
feat = num_cols + [c + '_te' for c in te_cols]
Xtr_f = Xtr_te[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
Xte_f = Xte_te[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
print('OOF cols sample', [c for c in Xtr_oof.columns if c.endswith('_te_oof')][:3])
print('holdout shape', Xtr_f.shape)
"""
            ),
            md(
                """## 6. wanglaoshi `FE.target_encode` API 演示

`AdvancedFeatureEngineer.target_encode` 在**传入的 DataFrame 上直接 groupby 目标**。  
课堂用法：只在 **训练集子集** 上调用；并口头/WARN：**禁止**把 train+test 拼在一起调用。

下方在 `train` 切片上演示，再把映射应用到 `valid`（与 `apply_te_from_train` 等价思路）。
"""
            ),
            code(
                """
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()
demo_col = te_cols[0]
train_slice = Xtr[[demo_col]].copy()
train_slice['__target__'] = ytr.values
print('⚠️ target_encode 必须在 fold-safe 流程中使用；下列仅在 train_slice 上演示 API')
encoded = engineer.target_encode(train_slice, demo_col, '__target__', smoothing=5.0)
enc_col = demo_col + '_target_encoded'
print(encoded[[demo_col, enc_col]].head(4))
"""
            ),
            md("## 7. 建模：缩放 + QuickBaseline（QUICK 跳 SVM）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr_f), columns=Xtr_f.columns, index=Xtr_f.index)
Xte_s = pd.DataFrame(scaler.transform(Xte_f), columns=Xte_f.columns, index=Xte_f.index)

if QUICK:
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
    }
    results = {}
    rows = []
    for name, model in models.items():
        model.fit(Xtr_s, ytr)
        pred_i = model.predict(Xte_s)
        acc = float((pred_i == np.asarray(yte)).mean())
        results[name] = {'model': model, 'predictions': pred_i}
        rows.append({'Model': name, 'Test Accuracy': acc})
    comparison = pd.DataFrame(rows).sort_values('Test Accuracy', ascending=False).reset_index(drop=True)
else:
    baseline = CT.QuickBaseline()
    results = baseline.run_all_models(Xtr_s, pd.Series(ytr), Xte_s, pd.Series(yte), task_type='classification')
    comparison = baseline.compare_models(results, top_n=5)

display(comparison)
best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
print(classification_report(yte, pred, zero_division=0))
"""
            ),
            md("## 8. 泄漏 vs OOF 快速对照（同模型、不同特征）"),
            code(
                """
from sklearn.preprocessing import StandardScaler as SS

sc_l = SS()
Xtr_l = sc_l.fit_transform(leak_tr_f)
Xte_l = sc_l.transform(leak_te_f)
clf = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
clf.fit(Xtr_l, ytr)
acc_leak = (clf.predict(Xte_l) == np.asarray(yte)).mean()
clf2 = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
clf2.fit(Xtr_s, ytr)
acc_safe = (clf2.predict(Xte_s) == np.asarray(yte)).mean()
print('holdout acc (leaky TE)  ', round(acc_leak, 4))
print('holdout acc (train-map) ', round(acc_safe, 4))
print('若 leaky 明显更高，说明全量统计不可用于上线')
"""
            ),
            md(
                r"""## 8.1 理论：smoothing 与稀有类别

目标编码 \(\hat{\mu}_c\) 向全局均值 \(\bar{y}\) 收缩，等价于贝叶斯视角下的小样本稳健化。  
`FE.target_encode(..., smoothing=5)` 越大 → 越接近全局均值，方差更小但偏差更大。  
高基数职业 `occupation` 等应配合 **OOF**，绝不可在全表上调用 `target_encode`。
"""
            ),
            code(
                """
from wanglaoshi import FeatureEngineering as FE

eng = FE.AdvancedFeatureEngineer()
te_demo = Xtr[[te_cols[1]]].copy()
te_demo[te_cols[1]] = te_demo[te_cols[1]].astype(str)
fr = eng.frequency_encode(te_demo, te_cols[1])
print('frequency_encode 列', [c for c in fr.columns if 'frequency' in c][0])
"""
            ),
            md("## 9. 分组表现（性别等）"),
            code(
                """
group_col = None
for c in ['sex', 'Sex', 'gender', 'Gender']:
    if c in Xte.columns:
        group_col = c
        break
if group_col:
    tmp = pd.DataFrame({
        'y': np.asarray(yte),
        'pred': np.asarray(pred),
        'g': Xte[group_col].astype(str).to_numpy(),
    })
    print(tmp.groupby('g').apply(lambda d: (d['y'] == d['pred']).mean()))
else:
    print('无 sex/gender 列，跳过分组；可替换为 race 等')
"""
            ),
            md(
                """## 9.1 公平性提示（入门级）

分组 Accuracy 不等时，不等于模型「歧视」，但提示应继续看 **组内 TPR/FPR、校准曲线**。  
Adult 历史数据含敏感属性，仅用于教学；上线前须做法务与伦理审查。
"""
            ),
            code(
                """
if group_col:
    tmp2 = tmp.copy()
    tmp2['hit'] = tmp2['y'] == tmp2['pred']
    print('group size\\n', tmp2.groupby('g').size())
    print('positive rate\\n', tmp2.groupby('g')['y'].mean())
"""
            ),
            md("## 10. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/10_adult_census')
out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'income': np.asarray(yte)})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred,
    str(out / 'sample_submission.csv'),
    str(out / 'submission.csv'),
    prediction_type='label',
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head(3)
"""
            ),
            md(
                """## 11. 检查清单
- [x] OOF / train-only TE 正确流程
- [x] 全量 TE 泄漏对照
- [x] `FE.target_encode` + fold-safe 警告
- [x] 分组表现入口
- [ ] 公平性：组间 TPR/FPR 差距
- [ ] 类别高基数：哈希 / 频率编码组合
"""
            ),
            md(
                """## 附录 A：目标编码决策树

1. 类别基数 < 20 且样本足 → One-Hot 或 有序编码  
2. 基数中等、树模型 → 频率编码 / CatBoost 内置  
3. 基数高、线性/神经网络 → **OOF Target Encoding** + smoothing  
4. 生产 → 把 TE 写入 sklearn `Pipeline` + `ColumnTransformer`，折内 `fit`  

`FE.target_encode` 适合快速试验，但文档字符串级别就要记住：**传入 DataFrame 的范围 = 拟合范围**。课堂演示在 `train_slice` 调用；竞赛 CV 应用 `oof_target_encode` 同款逻辑或第三方 `category_encoders`。

## 附录 B：Adult 字段与泄漏源

- `fnlwgt`：抽样权重，通常不直接入模或仅作 sample_weight  
- `education` 与 `education-num`：勿同时保留冗余  
- `relationship` / `marital-status`：语义相关，注意共线  
- **最危险操作**：`pd.concat([train,test])` 后 `target_encode` —— 本 Notebook 第 4 节专门打脸此操作  
"""
            ),
            md("## 附录 C：OOF 与 hold-out 列对齐检查"),
            code(
                """
oof_cols = [c + '_te_oof' for c in te_cols if c + '_te_oof' in Xtr_oof.columns]
print('OOF 特征数', len(oof_cols))
print('hold-out 特征数', len(feat))
print('示例 OOF 列均值', Xtr_oof[oof_cols[:2]].mean().to_dict())
"""
            ),
            md(
                """## 附录 D：读懂 wanglaoshi 文档中的 FE 警示

README / `book` 中多次强调：**目标编码统计量必须 fold-safe**。`AdvancedFeatureEngineer.target_encode` 不做折分，它是「单次 transform 工具」，不是 Kaggle Pipeline 替代品。正确做法：

1. 训练折估计映射  
2. 验证折 apply  
3. 测试折仅使用训练全集或各折模型集成后的映射（视策略而定）

与 `CompetitionTools.SubmissionGenerator` 联用时，预测列是离散收入标签（`<=50K` / `>50K` 的 0/1 表示），勿提交概率 CSV。
"""
            ),
            md("## 附录 E：类别基数一览"),
            code(
                """
cardinality = {c: Xtr[c].astype(str).nunique() for c in cat_cols}
top = sorted(cardinality.items(), key=lambda x: -x[1])[:10]
for c, n in top:
    print(c, n)
"""
            ),
            md(
                """## 附录 F：从朴素贝叶斯到 TE 的演进路径

1. **LabelEncoder + 线性模型**：快但把类别当有序，误伤高基数列  
2. **One-Hot**：稀疏、维度爆炸  
3. **Frequency / Count**：无监督压缩，不泄漏标签但丢失目标信息  
4. **Target Encoding**：有监督、高收益、**必须 OOF**  
5. **CatBoost ordered boosting**：工业级折内编码，Adult 冲榜常用  

本 Notebook 路径 3→4→`FE.target_encode` API 演示，与 `QuickBaseline` 选型衔接。QUICK=True 时抽样 4000 行是为证据脚本 5 分钟内跑通；全量训练请设 QUICK=False 并改用 KFold TE Pipeline。
"""
            ),
            md(TEACHER_E10),
        ],
    )


def e11() -> None:
    write(
        "11_house_prices",
        [
            md(
                """# E11 House Prices：高维混合、缺失含义与集成

> Issue #9 / Epic #6 / 路线图 E11

## 案例背景
Kaggle House Prices 是典型的**表格回归**：数值 + 高基数类别混合，缺失常带业务语义（如「无车库」时 Garage 列同时缺失）。  
竞赛常用 **log1p(SalePrice)** 训练以缓解右偏，提交时再 `expm1` 还原。

## 你将学到
1. 缺失指示与**高基数类别**的频率 / 目标编码思路（train-only）
2. **log 目标**与 RMSE 在价格空间解读
3. **wanglaoshi `ModelEnsemble.blending`**：树模型 + 线性模型互补
4. `SubmissionGenerator`：`prediction_type='label'`（价格点预测，非概率）

## 理论：为何 log 目标
房价右偏：绝对误差对大户型权重过高。对 \(\log(1+y)\) 优化近似相对误差，更稳定。  
集成动机：线性模型抓全局趋势，树模型抓非线性与交互。

## 数据
- 本地：`data/house_prices_train.csv`
- 回退：OpenML / 合成 DEMO
"""
            ),
            *env_cells(),
            md("## 1. 加载"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/house_prices_train.csv')
DEMO = False
if path.exists():
    train = pd.read_csv(path)
    print('local', train.shape)
else:
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml('house_prices', version=1, as_frame=True, parser='auto')
        train = raw.frame.copy()
        print('OpenML', train.shape)
    except Exception:
        DEMO = True
        print('合成 DEMO 房价数据')
        rng = np.random.RandomState(RANDOM_STATE)
        n = 800
        neighborhoods = [f'Nb_{i}' for i in range(25)]
        train = pd.DataFrame({
            'Id': np.arange(1, n + 1),
            'LotArea': rng.randint(1000, 20000, n),
            'OverallQual': rng.randint(1, 11, n),
            'YearBuilt': rng.randint(1900, 2011, n),
            'GarageArea': rng.choice([0, 200, 400, 600], n, p=[0.15, 0.3, 0.3, 0.25]),
            'Neighborhood': rng.choice(neighborhoods, n),
            'SalePrice': None,
        })
        train['SalePrice'] = (
            5e4 + 30 * train['LotArea'] + 2e4 * train['OverallQual']
            + 50 * train['GarageArea'] + rng.normal(0, 2e4, n)
        ).clip(3e4, None)

if 'SalePrice' not in train.columns:
    for c in train.columns:
        if 'sale' in c.lower() or c.lower() == 'price':
            train = train.rename(columns={c: 'SalePrice'})
            break
if QUICK and len(train) > 1200:
    train = train.sample(1200, random_state=RANDOM_STATE).reset_index(drop=True)
print('DEMO', DEMO)
train.head(3)
"""
            ),
            md("## 2. EDA：缺失可能有含义"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = train.sample(min(len(train), 3000), random_state=RANDOM_STATE) if QUICK else train
analyzer = DataAnalyzer(eda_df)
miss = analyzer.missing_value_analysis()
display(miss.head(20))
garage_cols = [c for c in train.columns if c.startswith('Garage')]
print('garage-related cols (sample)', garage_cols[:8])
"""
            ),
            md("## 3. log1p 目标与划分"),
            code(
                """
from sklearn.model_selection import train_test_split

work = train.copy()
y = np.log1p(work['SalePrice'].astype(float))
X = work.drop(columns=['SalePrice'], errors='ignore')
if 'Id' in X.columns:
    X = X.drop(columns=['Id'])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print('log-target mean', ytr.mean())
"""
            ),
            md(
                """## 4. 预处理：数值中位数 + 高基数类别频率编码

高基数 `Neighborhood` 等不宜直接 One-Hot。频率编码用**训练集**类别占比映射（未见类别 → 0）。  
也可用 `FE.frequency_encode`；统计量同样必须 train-only。
"""
            ),
            code(
                """
from wanglaoshi import FeatureEngineering as FE

num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in Xtr.columns if c not in num_cols]
# 取缺失率适中且高基数的若干类别列
card = {c: Xtr[c].astype(str).nunique() for c in cat_cols}
cat_use = sorted(card, key=card.get, reverse=True)[:12 if not QUICK else 8]
print('high-cardinality cats', [(c, card[c]) for c in cat_use[:5]])

med = Xtr[num_cols].median(numeric_only=True)
Xtr_n = Xtr[num_cols].fillna(med).copy()
Xte_n = Xte[num_cols].fillna(med).copy()
for c in cat_use:
    freq = Xtr[c].astype(str).value_counts(normalize=True)
    Xtr_n[c + '_freq'] = Xtr[c].astype(str).map(freq).fillna(0.0)
    Xte_n[c + '_freq'] = Xte[c].astype(str).map(freq).fillna(0.0)

engineer = FE.AdvancedFeatureEngineer()
if cat_use:
    demo = Xtr[[cat_use[0]]].copy()
    demo = engineer.frequency_encode(demo, cat_use[0])
    print('FE.frequency_encode demo col', demo.columns[-1])
print('design matrix', Xtr_n.shape)
"""
            ),
            md("## 5. 单模型 QuickBaseline（log 空间）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_squared_error, mean_absolute_error

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_n, ytr, Xte_n, yte, task_type='regression')
comparison = baseline.compare_models(results, top_n=6)
display(comparison)
best_name = comparison.iloc[0]['Model']
pred_best = np.asarray(results[best_name]['predictions'])
print('best', best_name, 'RMSE(log)', mean_squared_error(yte, pred_best, squared=False))
"""
            ),
            md(
                """## 6. `ModelEnsemble.blending`：树 + 线性

Blending 在训练集拟合基模型，在验证折上融合。  
回归任务下 `meta_model=None` 时返回**加权平均**预测函数（见 `wanglaoshi.CompetitionTools.ModelEnsemble`）。
"""
            ),
            code(
                """
from wanglaoshi.CompetitionTools import ModelEnsemble
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

X_fit, X_val, y_fit, y_val = train_test_split(
    Xtr_n, ytr, test_size=0.2, random_state=RANDOM_STATE + 1
)
models = [
    GradientBoostingRegressor(n_estimators=60 if QUICK else 120, random_state=RANDOM_STATE),
    RandomForestRegressor(n_estimators=80 if QUICK else 150, random_state=RANDOM_STATE, n_jobs=-1),
    Ridge(alpha=3.0),
]
ensemble = ModelEnsemble()
blend_pack = ensemble.blending(
    models=models,
    X_train=X_fit,
    y_train=y_fit,
    X_val=X_val,
    y_val=y_val,
    meta_model=None,
    weights=[0.45, 0.45, 0.10],
)
blend_pred = blend_pack['predict'](Xte_n)
single_rmse = mean_squared_error(yte, pred_best, squared=False)
blend_rmse = mean_squared_error(yte, blend_pred, squared=False)
print('holdout RMSE(log) single', round(single_rmse, 4), 'blend', round(blend_rmse, 4))
print('holdout MAE(price)', mean_absolute_error(np.expm1(yte), np.expm1(blend_pred)))
"""
            ),
            md(
                """## 6.1 缺失指示列（Garage 语义）

若多列 `Garage*` 同时缺失，可新增 `HasGarage=0`；有值时 `HasGarage=1`。  
树模型可自动分裂，但显式指示列往往利于线性模型与可解释性。
"""
            ),
            code(
                """
if garage_cols:
    miss_all = train[garage_cols].isna().all(axis=1).astype(int)
    print('rows all-Garage-missing', miss_all.sum(), '/', len(train))
    Xtr_n['HasGarage_hint'] = (~Xtr[garage_cols[:3]].isna().all(axis=1)).astype(int)
    Xte_n['HasGarage_hint'] = (~Xte[garage_cols[:3]].isna().all(axis=1)).astype(int)
"""
            ),
            md(
                """## 6.2 集成理论：何时 Blending 有效

- **多样性**：GBDT 抓非线性，Ridge 抓全局线性趋势
- **过拟合风险**：Blending 权重在 hold-out 上调；正式比赛应使用 stacking + OOF
- `ModelEnsemble.stacking` 需要 `cv` 折 OOF 预测，比本章 Blending 更严但更稳
"""
            ),
            code(
                """
print('单模型 top3:', comparison['Model'].head(3).tolist())
print('集成输出维度', np.asarray(blend_pred).shape)
"""
            ),
            md("## 7. 模拟提交（还原到价格空间）"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

price_pred = np.expm1(blend_pred)
price_true = np.expm1(yte.to_numpy())
out = Path('outputs/11_house_prices')
out.mkdir(parents=True, exist_ok=True)
ids = np.arange(1, len(price_pred) + 1)
sample = pd.DataFrame({'Id': ids, 'SalePrice': price_true})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    price_pred,
    str(out / 'sample_submission.csv'),
    str(out / 'submission.csv'),
    prediction_type='label',
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head(3)
"""
            ),
            md(
                """## 8. 检查清单
- [x] 缺失含义意识（Garage 等）
- [x] log1p 目标与 expm1 提交
- [x] 高基数类别频率编码 + `FE.frequency_encode`
- [x] `ModelEnsemble.blending`
- [ ] 缺失指示列
- [ ] stacking 出折 OOF 预测再 meta
"""
            ),
            md(
                """## 附录：房价回归方法论笔记

**log1p 与 RMSLE**：Kaggle House Prices 官方 metric 与 log 误差同源。训练在 log 空间，汇报时可同时给 `expm1` 后的 MAE 便于业务理解。

**高基数 Neighborhood**：频率编码是极简 baseline；更强方案是 TE（仍需 OOF）或 CatBoost。我们看到 `FE.frequency_encode` 与手写 `value_counts(normalize=True)` 等价，强调 **map 统计量来自 Xtr**。

**缺失结构**：`None` 在部分列表示「无该项设施」，不是 MCAR。缺失指示 + 填充「Unknown」水平，常比单纯 `median` 更好。

**ModelEnsemble**：`blending` 用固定验证子集；`stacking` 用 CV OOF，方差更小。权重 `[0.45,0.45,0.10]` 仅为示例，应在验证集上网格搜索。

**提交规范**：`SubmissionGenerator` 的 `prediction_type='label'` 输出美元价格点估计；勿把置信区间或 log 值直接写入 SalePrice 列。

**QUICK 模式**：子样本 + 浅树，RMSE 仅作流程验证；真要冲榜需全量特征工程 + 五折 stacking + 模型融合。
"""
            ),
            md("## 附录 B：数值特征与房价相关（抽样）"),
            code(
                """
corr_cols = [c for c in num_cols if c in Xtr.columns][:8]
if corr_cols:
    corrs = pd.concat([Xtr[corr_cols], ytr.rename('log_price')], axis=1).corr()['log_price'].drop('log_price')
    display(corrs.sort_values(ascending=False))
"""
            ),
            md("## 附录 C：`ModelEnsemble.stacking` 与本案关系"),
            code(
                """
# stacking 需要多折 OOF，QUICK 下仅打印 API 入口，避免长耗时
print('如需 stacking: ModelEnsemble().stacking(models, meta_model, X_train, y_train, cv=5)')
print('本案 blending 更适合课堂 5 分钟内跑通 evidence')
"""
            ),
            md("## 附录 D：SalePrice 分布与 log 变换动机"),
            code(
                """
import matplotlib.pyplot as plt
raw_y = np.expm1(ytr)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].hist(raw_y, bins=30, color='steelblue', alpha=0.85)
ax[0].set_title('SalePrice 原尺度')
ax[1].hist(ytr, bins=30, color='coral', alpha=0.85)
ax[1].set_title('log1p(SalePrice)')
plt.tight_layout()
plt.show()
print('raw skew', pd.Series(raw_y).skew(), 'log skew', pd.Series(ytr).skew())
"""
            ),
            md(
                """## 附录 E：高基数类别处理对照

| 方法 | 泄漏风险 | 适用模型 | 本案例 |
|------|----------|----------|--------|
| One-Hot Top-N | 低 | 线性/NN | 未用 |
| 频率编码 | 低（train map） | 树/线性 | ✅ |
| Target Encoding | 高（若全表） | 通用 | 见 Adult 案例 |
| CatBoost | 低（ordered） | 树 | 拓展 |

`ModelEnsemble.blending` 把多样性模型合成；当你仅有频率编码特征时，树模型往往单模最强，Ridge 提供线性基准防止过拟合。提交前 `expm1` 还原，并用 `SubmissionGenerator` 校验列名与 `Id` 对齐。
"""
            ),
            md(TEACHER_E11),
        ],
    )


def main() -> None:
    before = {s: _chars(s) for s in SLUGS}
    e05()
    e09()
    e10()
    e11()
    after = {s: _chars(s) for s in SLUGS}
    print('\n=== batch3 字符数 before -> after ===')
    for s in SLUGS:
        b, a = before[s], after[s]
        ratio = a / b if b else float('inf')
        print(f'{s}: {b} -> {a}  ({ratio:.2f}x)')
    total_b = sum(before.values())
    total_a = sum(after.values())
    print(f'TOTAL: {total_b} -> {total_a}  ({total_a / total_b:.2f}x)')


if __name__ == '__main__':
    main()
