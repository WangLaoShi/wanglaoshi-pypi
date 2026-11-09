"""生成 12 个足够细的进阶 Notebook（案例专属内容，非通用骨架）。

用法：
  python examples/advanced/_deepen_all.py

注意：会覆盖同名 .ipynb。E01/E02 若已手工打磨，脚本会一并重写为同深度标准版。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def env_cells() -> list:
    return [
        code(
            """
QUICK = True
RANDOM_STATE = 42

import warnings
warnings.filterwarnings('ignore')

from wanglaoshi import VERSIONS as V
from wanglaoshi import JupyterEnv as JE

JE.no_warning()
V.show_env_info()
"""
        )
    ]


def write(slug: str, cells: list) -> None:
    path = ROOT / f"{slug}.ipynb"
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", path.name, "cells=", len(cells))


# ---------------------------------------------------------------------------
# E01 Iris
# ---------------------------------------------------------------------------
def e01() -> None:
    write(
        "01_iris",
        [
            md(
                """# E01 Iris：小样本分层验证与标签提交

> Issue #9 / Epic #6 / 路线图 E01

## 你将学到
1. 小样本多分类为什么必须分层划分
2. 整数标签 vs 字符串标签如何正确生成 submission
3. 用 `wanglaoshi` 完成：环境检查 → EDA → 基线 → 提交校验

## 技术难点
- 每类只有 50 条，验证集波动大
- `target` 数字标签与 `species` 字符串标签提交格式不同
- 禁止对概率矩阵静默 argmax（见 B05 修复）
"""
            ),
            *env_cells(),
            md("## 1. 加载 Iris，同时保留数字标签与字符串标签"),
            code(
                """
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = iris.target_names[iris.target]
print('shape =', df.shape)
print('target counts:\\n', df['target'].value_counts().sort_index())
print('species counts:\\n', df['species'].value_counts())
df.head()
"""
            ),
            md("## 2. EDA：小样本下先看分布与相关"),
            code(
                """
from wanglaoshi import DataAnalyzer

analyzer = DataAnalyzer(df)
print(analyzer.get_basic_info())
display(analyzer.basic_statistics())
display(analyzer.missing_value_analysis())
"""
            ),
            md("## 3. 分层划分：对比非分层的风险"),
            code(
                """
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_cols = [c for c in df.columns if c not in ('target', 'species')]
X = df[feature_cols]
y_int = df['target']
y_str = df['species']

# 分层划分（正确）
Xtr, Xte, ytr, yte = train_test_split(
    X, y_int, test_size=0.3, random_state=RANDOM_STATE, stratify=y_int
)
print('stratified train:', ytr.value_counts().sort_index().to_dict())
print('stratified test :', yte.value_counts().sort_index().to_dict())

# 非分层（错误示范：某折可能缺类）
Xtr_bad, Xte_bad, ytr_bad, yte_bad = train_test_split(
    X, y_int, test_size=0.3, random_state=7, stratify=None
)
print('non-strat train:', ytr_bad.value_counts().sort_index().to_dict())

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(n_splits=5 if not QUICK else 3, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(pipe, X, y_int, cv=cv, scoring='accuracy')
print(f'CV accuracy mean±std = {scores.mean():.3f} ± {scores.std():.3f}')
"""
            ),
            md("## 4. QuickBaseline 对比"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr, ytr, Xte, yte, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("## 5. 提交：整数标签 vs 字符串标签"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

best = comparison.iloc[0]['Model']
pred_int = results[best]['predictions']
out = Path('outputs/01_iris'); out.mkdir(parents=True, exist_ok=True)

# A) 整数标签提交
sample_int = pd.DataFrame({'Id': np.arange(len(pred_int)), 'target': yte.to_numpy()})
sample_int.to_csv(out / 'sample_int.csv', index=False)
gen = CT.SubmissionGenerator()
sub_int = gen.create_submission(
    pred_int, str(out / 'sample_int.csv'), str(out / 'sub_int.csv'), prediction_type='label'
)
print('INT', gen.validate_submission(str(out / 'sub_int.csv'), str(out / 'sample_int.csv')))

# B) 字符串标签提交（把数字预测映射回物种名）
pred_str = iris.target_names[np.asarray(pred_int).astype(int)]
sample_str = pd.DataFrame({'Id': np.arange(len(pred_str)), 'species': y_str.loc[yte.index].to_numpy()})
sample_str.to_csv(out / 'sample_str.csv', index=False)
sub_str = gen.create_submission(
    pred_str, str(out / 'sample_str.csv'), str(out / 'sub_str.csv'),
    prediction_type='label', target_column='species'
)
print('STR', gen.validate_submission(str(out / 'sub_str.csv'), str(out / 'sample_str.csv')))
display(sub_int.head(3)); display(sub_str.head(3))
"""
            ),
            md("## 6. 误差分析"),
            code(
                """
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(yte, pred_int, target_names=list(iris.target_names)))
print(confusion_matrix(yte, pred_int))
"""
            ),
            md(
                """## 7. 检查清单
- [x] 分层划分与 CV 方差
- [x] 整数/字符串两种提交
- [x] SubmissionGenerator 校验
- [ ] 用概率提交时显式指定 prediction_type='probability'（禁止静默 argmax）
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E02 Titanic
# ---------------------------------------------------------------------------
def e02() -> None:
    write(
        "02_titanic",
        [
            md(
                """# E02 Titanic：缺失、文本特征与训练验证隔离

> Issue #9 / Epic #6 / 路线图 E02

## 目标
用 seaborn Titanic 演示真实竞赛三类坑：
1. 缺失填充统计只能在训练集上计算
2. 姓名/舱位等文本可衍生特征，但容易泄漏
3. 提交列名常是 `Survived` 而不是 `target`

## 难点清单
- Age/Embarked/Deck 缺失机制不同
- `alive` 几乎等于 survived，必须删除
- 用全量 median 填充会虚高线下分
"""
            ),
            *env_cells(),
            md("## 1. 加载并识别泄漏列"),
            code(
                """
import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset('titanic').copy()
print(df.shape)
print('columns:', list(df.columns))
leak_like = [c for c in ['alive', 'adult_male'] if c in df.columns]
print('potential leak-like columns:', leak_like)
df.head()
"""
            ),
            md("## 2. EDA：缺失与目标关系"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda = DataAnalyzer(df)
display(eda.missing_value_analysis())
display(eda.basic_statistics().head(15))
print('survived rate by class:\\n', df.groupby('class')['survived'].mean())
"""
            ),
            md("## 3. 正确缺失处理：仅在训练集拟合"),
            code(
                """
from sklearn.model_selection import train_test_split

work = df.drop(columns=[c for c in ['alive'] if c in df.columns]).copy()
y = work['survived']
X = work.drop(columns=['survived'])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_num = X[num_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_num, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ✅ 正确：只用训练集统计
med = X_train.median(numeric_only=True)
X_train_f = X_train.fillna(med)
X_valid_f = X_valid.fillna(med)

# ❌ 错误示范（勿用于正式评估）：med_all = X_num.median()
print('remaining NA train/valid:', X_train_f.isna().sum().sum(), X_valid_f.isna().sum().sum())
"""
            ),
            md("## 4. 类别频率编码（训练集拟合映射）"),
            code(
                """
cat_cols = [c for c in ['sex', 'embarked', 'class', 'who', 'embark_town', 'deck'] if c in work.columns]
X2 = work.drop(columns=['survived'])
y2 = work['survived']
Xtr, Xva, ytr, yva = train_test_split(X2, y2, test_size=0.25, random_state=RANDOM_STATE, stratify=y2)

for c in cat_cols:
    freq = Xtr[c].astype(str).value_counts(normalize=True)
    Xtr[c + '_freq'] = Xtr[c].astype(str).map(freq).fillna(0)
    Xva[c + '_freq'] = Xva[c].astype(str).map(freq).fillna(0)

num2 = Xtr.select_dtypes(include=[np.number]).columns
med2 = Xtr[num2].median()
Xtr_f = Xtr[num2].fillna(med2)
Xva_f = Xva[num2].fillna(med2)
print('numeric features after encoding:', len(num2))
"""
            ),
            md("## 5. QuickBaseline"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_f, ytr, Xva_f, yva, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("## 6. 提交：目标列名为 Survived"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
out = Path('outputs/02_titanic'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'PassengerId': np.arange(1, len(pred) + 1), 'Survived': yva.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head()
"""
            ),
            md(
                """## 7. 局限与下一步
- Cabin/Deck 稀疏性与票价异常的稳健策略未完成
- 目标编码未做折内 CV
- 下一步：StratifiedKFold 报告均值±方差，并对泄漏列做 ablation
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E03 Wine Quality
# ---------------------------------------------------------------------------
def e03() -> None:
    write(
        "03_wine_quality",
        [
            md(
                """# E03 Wine Quality：回归 / 有序分类与不均衡

> Issue #9 / Epic #6 / 路线图 E03

## 你将学到
1. `quality` 是有序整数：可当回归，也可当多分类
2. 极端分数（3、9）样本极少 → Accuracy 会骗人
3. 红/白酒化学特征不同，是否分开建模要对比

## 数据
- 优先：`data/winequality-red.csv`（UCI，`;` 分隔）
- 回退：OpenML `wine-quality-red`（需网络）
- **注意**：不是 `sklearn.datasets.load_wine`
"""
            ),
            *env_cells(),
            md("## 1. 加载 Wine Quality（带本地/OpenML 回退）"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/winequality-red.csv')
df = None
if path.exists():
    df = pd.read_csv(path, sep=';')
    print('loaded local', path, df.shape)
else:
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
        df = raw.frame.copy()
        # OpenML 列名可能是 class / quality
        if 'quality' not in df.columns and 'class' in df.columns:
            df = df.rename(columns={'class': 'quality'})
        print('loaded OpenML wine-quality-red', df.shape)
    except Exception as e:
        print('无法加载数据:', e)
        print('请下载 https://archive.ics.uci.edu/dataset/186/wine+quality 到 data/winequality-red.csv')

if df is not None:
    df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
    print(df['quality'].value_counts().sort_index())
    display(df.head())
"""
            ),
            md("## 2. EDA：目标不均衡与特征相关"),
            code(
                """
from wanglaoshi import DataAnalyzer

assert df is not None, '请先成功加载 df'
eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
print(analyzer.get_basic_info())
display(analyzer.missing_value_analysis())
display(analyzer.basic_statistics().head(20))
print('quality distribution %:\\n', (df['quality'].value_counts(normalize=True).sort_index() * 100).round(1))
"""
            ),
            md("## 3. 双视角：回归 vs 有序分类划分"),
            code(
                """
from sklearn.model_selection import train_test_split

feature_cols = [c for c in df.columns if c != 'quality']
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = df['quality'].astype(int)

# 分层按 quality（分类视角）
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
med = Xtr.median()
Xtr_f, Xte_f = Xtr.fillna(med), Xte.fillna(med)
print(Xtr_f.shape, 'train quality counts', ytr.value_counts().sort_index().to_dict())
"""
            ),
            md("## 4. 回归基线（把 quality 当连续）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_absolute_error, mean_squared_error

baseline = CT.QuickBaseline()
reg_results = baseline.run_all_models(Xtr_f, ytr, Xte_f, yte, task_type='regression')
reg_cmp = baseline.compare_models(reg_results, top_n=5)
display(reg_cmp)

best_reg = reg_cmp.iloc[0]['Model']
pred_reg = reg_results[best_reg]['predictions']
print('MAE', mean_absolute_error(yte, pred_reg))
print('RMSE', mean_squared_error(yte, pred_reg, squared=False))
# 有序性检查：把回归预测四舍五入后看 Accuracy
pred_round = np.clip(np.rint(pred_reg), y.min(), y.max()).astype(int)
print('rounded accuracy', (pred_round == yte.to_numpy()).mean())
"""
            ),
            md("## 5. 分类基线（注意极端类稀少）"),
            code(
                """
from sklearn.metrics import classification_report, f1_score

clf_results = baseline.run_all_models(Xtr_f, ytr, Xte_f, yte, task_type='classification')
clf_cmp = baseline.compare_models(clf_results, top_n=5)
display(clf_cmp)
best_clf = clf_cmp.iloc[0]['Model']
pred_clf = clf_results[best_clf]['predictions']
print(classification_report(yte, pred_clf, zero_division=0))
print('macro-F1', f1_score(yte, pred_clf, average='macro', zero_division=0))
"""
            ),
            md("## 6. 模拟提交与局限"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/03_wine_quality'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_reg)), 'quality': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_reg, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] 区分回归 / 分类两种建模视角
- [x] 报告 macro-F1（不只 Accuracy）
- [x] 本地/OpenML 双路径加载
- [ ] 红白酒合并 vs 分开建模对比
- [ ] 用 ordinal 模型或相邻类惩罚
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E04 Breast Cancer
# ---------------------------------------------------------------------------
def e04() -> None:
    write(
        "04_breast_cancer",
        [
            md(
                """# E04 Breast Cancer：缩放、PCA 与召回权衡

> Issue #9 / Epic #6 / 路线图 E04

## 你将学到
1. 特征量纲差异大 → 必须缩放再进线性/距离模型
2. PCA 降维可加速，但可解释性下降
3. 医疗场景：**漏诊成本高**，不能只看 Accuracy

## 指标重点
- Recall（召回）/ Precision-Recall 曲线
- 阈值扫描：用业务成本选阈值，而不是默认 0.5
"""
            ),
            *env_cells(),
            md("## 1. 加载 breast_cancer"),
            code(
                """
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
print(df.shape)
print(df['target'].value_counts())  # 1=malignant? 以 data.target_names 为准
print('target_names:', list(data.target_names))
df.head()
"""
            ),
            md("## 2. EDA：量纲差异"),
            code(
                """
from wanglaoshi import DataAnalyzer

analyzer = DataAnalyzer(df)
display(analyzer.basic_statistics().head(12))
print('feature scale span (max/min of mean):',
      df.drop(columns=['target']).mean().max() / max(df.drop(columns=['target']).mean().min(), 1e-9))
"""
            ),
            md("## 3. Pipeline：StandardScaler + PCA + LogisticRegression"),
            code(
                """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

X = df.drop(columns=['target'])
y = df['target']
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10 if QUICK else 15, random_state=RANDOM_STATE)),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
print(classification_report(yte, pred, target_names=list(data.target_names)))
print('ROC-AUC', roc_auc_score(yte, proba))
print('AP', average_precision_score(yte, proba))
"""
            ),
            md("## 4. 阈值扫描：提高召回"),
            code(
                """
from sklearn.metrics import recall_score, precision_score, f1_score

rows = []
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (proba >= thr).astype(int)
    rows.append({
        'thr': thr,
        'precision': precision_score(yte, p, zero_division=0),
        'recall': recall_score(yte, p, zero_division=0),
        'f1': f1_score(yte, p, zero_division=0),
    })
thr_df = pd.DataFrame(rows)
display(thr_df)

prec, rec, ths = precision_recall_curve(yte, proba)
plt.figure(figsize=(5, 4))
plt.plot(rec, prec)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve')
plt.grid(True, alpha=0.3)
plt.show()
"""
            ),
            md("## 5. QuickBaseline 对比（无 PCA）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_s, ytr, Xte_s, yte, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

best = comparison.iloc[0]['Model']
pred_best = results[best]['predictions']
out = Path('outputs/04_breast_cancer'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_best)), 'target': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_best, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] 分层划分
- [x] 缩放 + PCA 管线
- [x] 阈值-召回权衡与 PR 曲线
- [ ] 按假阴性/假阳性业务成本选最终阈值
- [ ] 校准概率（calibration）后再定阈值
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E05 Credit Card Fraud
# ---------------------------------------------------------------------------
def e05() -> None:
    write(
        "05_credit_card_fraud",
        [
            md(
                """# E05 Credit Card Fraud：时间划分、极端不均衡与 PR-AUC

> Issue #9 / Epic #6 / 路线图 E05

## 你将学到
1. 正类极稀 → Accuracy 几乎无意义
2. 应按 `Time` 切分，避免用未来信息训练
3. 关注 PR-AUC / Recall@Precision，而不是 ROC 一家独大

## 数据
- 需要本地下载 Kaggle `creditcard.csv` → `data/creditcard.csv`
- 若缺失，本 Notebook 会用合成不均衡数据演示流程（标注为 DEMO）
"""
            ),
            *env_cells(),
            md("## 1. 加载真实数据或合成 DEMO"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

path = Path('../../data/creditcard.csv')
DEMO = False
if path.exists():
    df = pd.read_csv(path)
    if QUICK and len(df) > 80000:
        # 保持正类：分层抽样
        pos = df[df['Class'] == 1]
        neg = df[df['Class'] == 0].sample(min(20000, (df['Class'] == 0).sum()), random_state=RANDOM_STATE)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_STATE)
    print('loaded', path, df.shape)
    print(df['Class'].value_counts(normalize=True))
else:
    DEMO = True
    print('未找到 creditcard.csv，使用合成不均衡 DEMO 数据')
    X, y = make_classification(
        n_samples=8000 if QUICK else 30000,
        n_features=20, n_informative=8, weights=[0.985, 0.015],
        flip_y=0.01, random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f'V{i}' for i in range(X.shape[1])])
    df['Time'] = np.arange(len(df))
    df['Amount'] = np.abs(np.random.RandomState(RANDOM_STATE).normal(50, 30, len(df)))
    df['Class'] = y
    print(df['Class'].value_counts(normalize=True))
df.head()
"""
            ),
            md("## 2. EDA：类别比例与 Amount"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.basic_statistics().head(10))
print('fraud rate', df['Class'].mean())
print(df.groupby('Class')['Amount'].describe() if 'Amount' in df.columns else 'no Amount')
"""
            ),
            md("## 3. 时间切分（禁止随机打乱）"),
            code(
                """
# 按 Time 排序后切末尾 20% 作验证
work = df.sort_values('Time').reset_index(drop=True)
split = int(len(work) * 0.8)
train, valid = work.iloc[:split], work.iloc[split:]
feature_cols = [c for c in work.columns if c not in ('Class', 'Time')]
Xtr, Xva = train[feature_cols], valid[feature_cols]
ytr, yva = train['Class'], valid['Class']
print('train fraud rate', ytr.mean(), 'valid fraud rate', yva.mean())
print('time range train', train['Time'].min(), '->', train['Time'].max())
print('time range valid', valid['Time'].min(), '->', valid['Time'].max())
"""
            ),
            md("## 4. 基线 + PR-AUC（忽略 Accuracy）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns)
Xva_s = pd.DataFrame(scaler.transform(Xva), columns=Xva.columns)

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_s, ytr, Xva_s, yva, task_type='classification')
comparison = baseline.compare_models(results, top_n=5)
display(comparison)

best = comparison.iloc[0]['Model']
model = results[best]['model']
# 尽量取概率
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(Xva_s)[:, 1]
else:
    proba = results[best]['predictions']
pred = (np.asarray(proba) >= 0.5).astype(int) if np.ndim(proba) else results[best]['predictions']
print(classification_report(yva, pred, zero_division=0))
try:
    print('PR-AUC', average_precision_score(yva, proba))
    print('ROC-AUC', roc_auc_score(yva, proba))
except Exception as e:
    print('metric skip', e)
"""
            ),
            md("## 5. 阈值：在固定 Precision 下最大化 Recall"),
            code(
                """
from sklearn.metrics import precision_recall_curve

if hasattr(model, 'predict_proba'):
    prec, rec, ths = precision_recall_curve(yva, proba)
    # 找 precision >= 0.5 时最高 recall
    mask = prec[:-1] >= 0.5
    if mask.any():
        idx = np.argmax(rec[:-1][mask])
        thr = ths[mask][idx]
        print('chosen thr', thr, 'precision', prec[:-1][mask][idx], 'recall', rec[:-1][mask][idx])
    else:
        print('无 precision>=0.5 的阈值，请放宽约束')
else:
    print('模型无 predict_proba，跳过阈值搜索')
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/05_credit_card_fraud'); out.mkdir(parents=True, exist_ok=True)
pred_label = results[best]['predictions']
sample = pd.DataFrame({'Id': np.arange(len(pred_label)), 'Class': yva.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_label, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
print('DEMO mode =', DEMO)
"""
            ),
            md(
                """## 7. 检查清单
- [x] 时间切分而非随机划分
- [x] 报告 PR-AUC
- [x] 无数据时 DEMO 可跑通流程
- [ ] class_weight / 重采样对比
- [ ] 成本敏感阈值上线验证
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E06 California Housing
# ---------------------------------------------------------------------------
def e06() -> None:
    write(
        "06_california_housing",
        [
            md(
                """# E06 California Housing：空间分组 vs 随机划分

> Issue #9 / Epic #6 / 路线图 E06

## 你将学到
1. 地理邻近样本相关 → 随机划分会泄漏空间信息
2. 用经纬度分桶 / GroupKFold 对比随机划分的乐观偏差
3. 不同房价区间的误差结构不同

## 数据
`sklearn.datasets.fetch_california_housing`
"""
            ),
            *env_cells(),
            md("## 1. 加载数据"),
            code(
                """
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()
print(df.shape)
df.head()
"""
            ),
            md("## 2. EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(5000, random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.basic_statistics())
print(df[['Latitude', 'Longitude', 'MedHouseVal']].describe())
"""
            ),
            md("## 3. 对比：随机划分 vs 空间分桶划分"),
            code(
                """
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

feature_cols = [c for c in df.columns if c != 'MedHouseVal']
X = df[feature_cols]
y = df['MedHouseVal']

# A) 随机划分
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
model = RandomForestRegressor(
    n_estimators=50 if QUICK else 150, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12
)
model.fit(Xtr_r, ytr_r)
pred_r = model.predict(Xte_r)
print('RANDOM MAE', mean_absolute_error(yte_r, pred_r),
      'RMSE', mean_squared_error(yte_r, pred_r, squared=False))

# B) 空间分桶：把 lat/lon 量化成网格 group
lat_bin = pd.cut(df['Latitude'], bins=10, labels=False)
lon_bin = pd.cut(df['Longitude'], bins=10, labels=False)
groups = (lat_bin.astype(str) + '_' + lon_bin.astype(str))
# 用一组 holdout groups
uniq = groups.unique()
rng = np.random.RandomState(RANDOM_STATE)
hold = set(rng.choice(uniq, size=max(2, len(uniq) // 5), replace=False))
mask = groups.isin(hold)
Xtr_g, Xte_g = X.loc[~mask], X.loc[mask]
ytr_g, yte_g = y.loc[~mask], y.loc[mask]
model.fit(Xtr_g, ytr_g)
pred_g = model.predict(Xte_g)
print('SPATIAL MAE', mean_absolute_error(yte_g, pred_g),
      'RMSE', mean_squared_error(yte_g, pred_g, squared=False))
print('holdout groups', len(hold), 'test size', mask.sum())
"""
            ),
            md("## 4. QuickBaseline（随机划分，便于对比模型）"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

# 为速度可抽样
if QUICK:
    sample_idx = Xtr_r.sample(min(4000, len(Xtr_r)), random_state=RANDOM_STATE).index
    Xtr_b, ytr_b = Xtr_r.loc[sample_idx], ytr_r.loc[sample_idx]
else:
    Xtr_b, ytr_b = Xtr_r, ytr_r

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_b, ytr_b, Xte_r, yte_r, task_type='regression')
comparison = baseline.compare_models(results, top_n=5)
comparison
"""
            ),
            md("## 5. 分房价区间看误差"),
            code(
                """
best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
err = pd.DataFrame({'y': yte_r.to_numpy(), 'pred': np.asarray(pred)})
err['abs_err'] = (err['y'] - err['pred']).abs()
err['bin'] = pd.qcut(err['y'], 4, duplicates='drop')
display(err.groupby('bin', observed=True)['abs_err'].agg(['mean', 'median', 'count']))
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/06_california_housing'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'MedHouseVal': yte_r.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] 随机 vs 空间划分对比
- [x] 分位数区间误差
- [ ] GroupKFold 完整报告均值±方差
- [ ] 去掉 Latitude/Longitude 做 ablation（测泄漏程度）
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E07 Abalone
# ---------------------------------------------------------------------------
def e07() -> None:
    write(
        "07_abalone",
        [
            md(
                """# E07 Abalone：混合编码、目标变换与残差

> Issue #9 / Epic #6 / 路线图 E07

## 你将学到
1. `Sex` 类别编码方式影响很大
2. 目标 `Rings` 可做 log1p 再还原
3. 残差分析看是否系统偏差

## 数据
- 本地 `data/abalone.csv` 或 OpenML `abalone`
"""
            ),
            *env_cells(),
            md("## 1. 加载 Abalone"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/abalone.csv')
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    from sklearn.datasets import fetch_openml
    raw = fetch_openml('abalone', version=1, as_frame=True, parser='auto')
    df = raw.frame.copy()
    print('OpenML', df.shape)

# 统一列名
rename = {c: c.strip() for c in df.columns}
df = df.rename(columns=rename)
if 'Rings' not in df.columns:
    # OpenML 有时叫 class
    for c in df.columns:
        if c.lower() in ('rings', 'class'):
            df = df.rename(columns={c: 'Rings'})
            break
df['Rings'] = pd.to_numeric(df['Rings'], errors='coerce')
print(df.head())
print(df['Sex'].value_counts() if 'Sex' in df.columns else df.columns)
"""
            ),
            md("## 2. EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

analyzer = DataAnalyzer(df.dropna())
display(analyzer.basic_statistics())
display(analyzer.missing_value_analysis())
"""
            ),
            md("## 3. 编码 Sex + 划分"),
            code(
                """
from sklearn.model_selection import train_test_split

work = df.dropna().copy()
# one-hot Sex
work = pd.get_dummies(work, columns=['Sex'], drop_first=True)
y = work['Rings']
X = work.drop(columns=['Rings']).apply(pd.to_numeric, errors='coerce')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
print(Xtr.shape)
"""
            ),
            md("## 4. 原尺度 vs log1p 目标"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_absolute_error, mean_squared_error

baseline = CT.QuickBaseline()

# A) 原尺度
res_a = baseline.run_all_models(Xtr, ytr, Xte, yte, task_type='regression')
cmp_a = baseline.compare_models(res_a, top_n=4)
display(cmp_a)
best_a = cmp_a.iloc[0]['Model']
pred_a = res_a[best_a]['predictions']
print('raw MAE', mean_absolute_error(yte, pred_a))

# B) log1p
ytr_l, yte_l = np.log1p(ytr), np.log1p(yte)
res_b = baseline.run_all_models(Xtr, ytr_l, Xte, yte_l, task_type='regression')
cmp_b = baseline.compare_models(res_b, top_n=4)
display(cmp_b)
best_b = cmp_b.iloc[0]['Model']
pred_b = np.expm1(res_b[best_b]['predictions'])
print('log1p->expm1 MAE', mean_absolute_error(yte, pred_b))
"""
            ),
            md("## 5. 残差分析"),
            code(
                """
import matplotlib.pyplot as plt

resid = yte.to_numpy() - np.asarray(pred_b)
plt.figure(figsize=(6, 4))
plt.scatter(yte, resid, s=10, alpha=0.5)
plt.axhline(0, color='k', lw=1)
plt.xlabel('true Rings'); plt.ylabel('residual (true-pred)')
plt.title('Residual vs Rings')
plt.show()
print('residual mean', resid.mean(), 'std', resid.std())
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/07_abalone'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_b)), 'Rings': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_b, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] Sex one-hot
- [x] log1p 目标对比
- [x] 残差图
- [ ] 目标编码 Sex 的折内对比
- [ ] 分年龄段校准
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E08 Bike Sharing
# ---------------------------------------------------------------------------
def e08() -> None:
    write(
        "08_bike_sharing",
        [
            md(
                """# E08 Bike Sharing：时间验证与可用滞后特征

> Issue #9 / Epic #6 / 路线图 E08

## 你将学到
1. 必须按时间切分，禁止随机打乱
2. 滞后特征只能用预测时刻已知信息
3. 常用 RMSLE 作为指标

## 数据
- 本地 `data/bike_sharing_hour.csv`
- 回退：OpenML / 合成时间序列 DEMO
"""
            ),
            *env_cells(),
            md("## 1. 加载数据"),
            code(
                """
from pathlib import Path
import pandas as pd
import numpy as np

path = Path('../../data/bike_sharing_hour.csv')
DEMO = False
if path.exists():
    df = pd.read_csv(path)
    print('local', df.shape)
else:
    try:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml('Bike_Sharing_Demand', version=1, as_frame=True, parser='auto')
        df = raw.frame.copy()
        print('OpenML', df.shape)
    except Exception:
        DEMO = True
        print('使用合成时间序列 DEMO')
        n = 2000 if QUICK else 5000
        rng = np.random.RandomState(RANDOM_STATE)
        idx = pd.date_range('2012-01-01', periods=n, freq='h')
        df = pd.DataFrame({
            'datetime': idx,
            'temp': 20 + 10 * np.sin(np.arange(n) / 24) + rng.normal(0, 2, n),
            'hum': 60 + rng.normal(0, 10, n),
            'windspeed': np.abs(rng.normal(10, 3, n)),
            'season': ((idx.month % 12) // 3 + 1),
            'holiday': 0,
            'workingday': (idx.dayofweek < 5).astype(int),
            'cnt': None,
        })
        df['cnt'] = (
            100 + 40 * np.sin(np.arange(n) / 24) + 20 * df['workingday'] + rng.normal(0, 15, n)
        ).clip(0, None)

# 统一目标列
if 'cnt' not in df.columns:
    for c in ['count', 'Count', 'total']:
        if c in df.columns:
            df = df.rename(columns={c: 'cnt'})
            break
print(df.head())
print('DEMO', DEMO)
"""
            ),
            md("## 2. 时间特征与排序"),
            code(
                """
work = df.copy()
# 解析时间
time_col = None
for c in ['dteday', 'datetime', 'date', 'timestamp']:
    if c in work.columns:
        time_col = c
        break
if time_col:
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.sort_values(time_col).reset_index(drop=True)
    work['hour'] = work[time_col].dt.hour if hasattr(work[time_col].dt, 'hour') else 0
    work['dow'] = work[time_col].dt.dayofweek
else:
    work = work.reset_index(drop=True)
    work['hour'] = work['hr'] if 'hr' in work.columns else 0
    work['dow'] = work['weekday'] if 'weekday' in work.columns else 0

print(work.shape)
"""
            ),
            md("## 3. 安全滞后特征（只用过去）"),
            code(
                """
# 滞后 1/24 小时的 cnt；验证集构造时注意 shift 不穿越未来
work['cnt_lag1'] = work['cnt'].shift(1)
work['cnt_lag24'] = work['cnt'].shift(24)
work['cnt_roll24'] = work['cnt'].shift(1).rolling(24).mean()
work = work.dropna().reset_index(drop=True)

split = int(len(work) * 0.8)
train, valid = work.iloc[:split], work.iloc[split:]
drop_cols = [c for c in ['cnt', time_col, 'casual', 'registered', 'instant', 'dteday', 'datetime'] if c and c in work.columns]
feature_cols = [c for c in work.columns if c not in drop_cols]
# 只用数值
Xtr = train[feature_cols].select_dtypes(include=[np.number])
Xva = valid[feature_cols].select_dtypes(include=[np.number])
ytr, yva = train['cnt'], valid['cnt']
print('features', list(Xtr.columns))
print('train', Xtr.shape, 'valid', Xva.shape)
"""
            ),
            md("## 4. 基线 + RMSLE"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr.fillna(0), ytr, Xva.fillna(0), yva, task_type='regression')
comparison = baseline.compare_models(results, top_n=5)
display(comparison)
best = comparison.iloc[0]['Model']
pred = np.clip(np.asarray(results[best]['predictions']), 0, None)
print('MAE', mean_absolute_error(yva, pred))
print('RMSLE', mean_squared_log_error(yva, pred, squared=False))
"""
            ),
            md("## 5. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/08_bike_sharing'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'cnt': yva.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] 时间切分
- [x] 滞后特征只用过去
- [x] RMSLE
- [ ] 天气/节假日交互
- [ ] 禁止把 casual+registered 当特征（泄漏）
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E09 Mall Customers
# ---------------------------------------------------------------------------
def e09() -> None:
    write(
        "09_mall_customers",
        [
            md(
                """# E09 Mall Customers：缩放、聚类数与稳定性

> Issue #9 / Epic #6 / 路线图 E09

## 你将学到
1. 聚类前必须缩放
2. k 的选择：肘部法 + silhouette
3. 多次随机种子检查簇稳定性

## 数据
- 本地 `data/Mall_Customers.csv`
- 回退：合成客户画像 DEMO
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
    n = 300
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
display(analyzer.basic_statistics())
"""
            ),
            md("## 3. 选特征、缩放、搜 k"),
            code(
                """
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 常见列名兼容
cols = []
for a, b in [
    ('Annual Income (k$)', 'Annual Income (k$)'),
    ('Spending Score (1-100)', 'Spending Score (1-100)'),
    ('Age', 'Age'),
]:
    if a in df.columns:
        cols.append(a)
if len(cols) < 2:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
    cols = [c for c in cols if 'id' not in c.lower()]

X = df[cols].astype(float)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lab = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xs, lab))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(list(ks), inertias, marker='o'); ax[0].set_title('Elbow'); ax[0].set_xlabel('k')
ax[1].plot(list(ks), sils, marker='o'); ax[1].set_title('Silhouette'); ax[1].set_xlabel('k')
plt.tight_layout(); plt.show()
best_k = list(ks)[int(np.argmax(sils))]
print('best_k by silhouette', best_k, 'score', max(sils))
"""
            ),
            md("## 4. 多随机种子稳定性"),
            code(
                """
from sklearn.metrics import adjusted_rand_score

base = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
aris = []
for seed in range(5 if QUICK else 20):
    lab = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit_predict(Xs)
    aris.append(adjusted_rand_score(base, lab))
print('ARI vs base seed: mean', np.mean(aris), 'min', np.min(aris), aris)
"""
            ),
            md("## 5. 画像解读"),
            code(
                """
labels = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
profile = df.copy()
profile['cluster'] = labels
display(profile.groupby('cluster')[cols].mean())
"""
            ),
            md(
                """## 6. 检查清单
- [x] 缩放后聚类
- [x] elbow + silhouette
- [x] 多 seed ARI 稳定性
- [ ] 业务命名簇（高消费/低消费等）
- [ ] 与监督标签（若有）对照
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E10 Adult Census
# ---------------------------------------------------------------------------
def e10() -> None:
    write(
        "10_adult_census",
        [
            md(
                """# E10 Adult Census：折内目标编码与分组表现

> Issue #9 / Epic #6 / 路线图 E10

## 你将学到
1. 目标编码必须在折内拟合，否则泄漏
2. 不同人群（性别等）表现差异需单独评估
3. 高基数类别处理

## 数据
- 本地 `data/adult.csv` 或 OpenML `adult`
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

# 目标列统一
target_col = None
for c in ['income', 'class', 'target', '>50K']:
    if c in df.columns:
        target_col = c
        break
assert target_col, list(df.columns)
print('target', target_col, df[target_col].value_counts())
df.head()
"""
            ),
            md("## 2. EDA"),
            code(
                """
from wanglaoshi import DataAnalyzer

eda_df = df.sample(min(len(df), 5000), random_state=RANDOM_STATE) if QUICK else df
analyzer = DataAnalyzer(eda_df)
display(analyzer.missing_value_analysis().head(15))
display(analyzer.basic_statistics().head(15))
"""
            ),
            md("## 3. 折内目标编码 vs 错误的全量编码"),
            code(
                """
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

work = df.copy()
# 二值化目标
y_raw = work[target_col].astype(str)
y = (y_raw.str.contains('>50K')).astype(int) if y_raw.str.contains('>50K').any() else LabelEncoder().fit_transform(y_raw)
work = work.drop(columns=[target_col])

cat_cols = work.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
print('cat', cat_cols[:8], 'num', num_cols[:8])

Xtr, Xte, ytr, yte = train_test_split(
    work, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

def target_encode_inplace(train, valid, y_train, cols):
    train = train.copy(); valid = valid.copy()
    for c in cols:
        means = y_train.groupby(train[c].astype(str)).mean()
        global_mean = y_train.mean()
        train[c + '_te'] = train[c].astype(str).map(means).fillna(global_mean)
        valid[c + '_te'] = valid[c].astype(str).map(means).fillna(global_mean)
    return train, valid

# ✅ 只用训练集统计做 TE（仍建议进一步用 KFold；这里先演示 train-only map）
Xtr_te, Xte_te = target_encode_inplace(Xtr, Xte, pd.Series(ytr, index=Xtr.index), cat_cols[:6])

# ❌ 错误：用全量（含验证）统计 —— 仅作对比，勿上线
# full_means = pd.Series(y, index=work.index).groupby(work[c]).mean()

feat = num_cols + [c + '_te' for c in cat_cols[:6]]
Xtr_f = Xtr_te[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
Xte_f = Xte_te[feat].apply(pd.to_numeric, errors='coerce').fillna(0)
print(Xtr_f.shape)
"""
            ),
            md("## 4. QuickBaseline"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import classification_report, roc_auc_score

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_f, pd.Series(ytr), Xte_f, pd.Series(yte), task_type='classification')
comparison = baseline.compare_models(results, top_n=5)
display(comparison)
best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
print(classification_report(yte, pred, zero_division=0))
"""
            ),
            md("## 5. 分组表现（如性别）"),
            code(
                """
group_col = 'sex' if 'sex' in Xte.columns else ('Sex' if 'Sex' in Xte.columns else None)
if group_col:
    tmp = pd.DataFrame({'y': yte, 'pred': pred, 'g': Xte[group_col].astype(str).to_numpy()})
    print(tmp.groupby('g').apply(lambda d: (d['y'] == d['pred']).mean()))
else:
    print('无 sex 列，跳过分组评估；可用 race/gender 等列自行替换')
"""
            ),
            md("## 6. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/10_adult_census'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred)), 'income': np.asarray(yte)})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
"""
            ),
            md(
                """## 7. 检查清单
- [x] train-only 目标编码
- [x] 分组表现评估入口
- [ ] 真正的 KFold 折内 TE
- [ ] 公平性指标（TPR 差距等）
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E11 House Prices
# ---------------------------------------------------------------------------
def e11() -> None:
    write(
        "11_house_prices",
        [
            md(
                """# E11 House Prices：高维混合、缺失含义与集成

> Issue #9 / Epic #6 / 路线图 E11

## 你将学到
1. 缺失本身可能有业务含义（没有车库）
2. 目标常用 log1p
3. 树模型 + 线性模型简单加权集成

## 数据
- Kaggle House Prices `train.csv` → `data/house_prices_train.csv`
- 回退：OpenML `house_prices` 或合成 DEMO
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
        train = pd.DataFrame({
            'LotArea': rng.randint(1000, 20000, n),
            'OverallQual': rng.randint(1, 11, n),
            'YearBuilt': rng.randint(1900, 2011, n),
            'GarageArea': rng.choice([0, 200, 400, 600], n, p=[0.15, 0.3, 0.3, 0.25]),
            'Neighborhood': rng.choice(list('ABCDEFGH'), n),
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
print(train.head())
print('DEMO', DEMO)
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
# 示例：Garage* 同时缺失常表示无车库
garage_cols = [c for c in train.columns if c.startswith('Garage')]
print('garage cols', garage_cols[:8])
"""
            ),
            md("## 3. 简单预处理 + log1p 目标"),
            code(
                """
from sklearn.model_selection import train_test_split

work = train.copy()
y = np.log1p(work['SalePrice'].astype(float))
X = work.drop(columns=['SalePrice'])
# 仅演示：数值填中位数 + 类别频率编码（train-only）
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in Xtr.columns if c not in num_cols][:15]

med = Xtr[num_cols].median(numeric_only=True)
Xtr_n = Xtr[num_cols].fillna(med)
Xte_n = Xte[num_cols].fillna(med)
for c in cat_cols:
    freq = Xtr[c].astype(str).value_counts(normalize=True)
    Xtr_n[c + '_freq'] = Xtr[c].astype(str).map(freq).fillna(0)
    Xte_n[c + '_freq'] = Xte[c].astype(str).map(freq).fillna(0)
print(Xtr_n.shape)
"""
            ),
            md("## 4. QuickBaseline + 简单集成"),
            code(
                """
from wanglaoshi import CompetitionTools as CT
from sklearn.metrics import mean_squared_error, mean_absolute_error

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_n, ytr, Xte_n, yte, task_type='regression')
comparison = baseline.compare_models(results, top_n=6)
display(comparison)

# 取前两名做等权平均（在 log 空间）
names = comparison['Model'].head(2).tolist()
preds = [np.asarray(results[n]['predictions']) for n in names]
blend = np.mean(preds, axis=0)
print('single best RMSE(log)', mean_squared_error(yte, preds[0], squared=False))
print('blend RMSE(log)', mean_squared_error(yte, blend, squared=False))
print('blend MAE(price)', mean_absolute_error(np.expm1(yte), np.expm1(blend)))
"""
            ),
            md("## 5. 模拟提交（还原到价格）"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

price_pred = np.expm1(blend)
price_true = np.expm1(yte.to_numpy())
out = Path('outputs/11_house_prices'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(1, len(price_pred) + 1), 'SalePrice': price_true})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    price_pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head()
"""
            ),
            md(
                """## 6. 检查清单
- [x] 缺失含义意识（Garage 等）
- [x] log1p 目标与还原提交
- [x] 简单加权集成
- [ ] 缺失指示列 + 类别缺失单独水平
- [ ] stacking 出折预测
"""
            ),
        ],
    )


# ---------------------------------------------------------------------------
# E12 Digits
# ---------------------------------------------------------------------------
def e12() -> None:
    write(
        "12_digit_recognizer",
        [
            md(
                """# E12 Digit Recognizer：高维 EDA、PCA 与错分分析

> Issue #9 / Epic #6 / 路线图 E12

## 你将学到
1. 高维像素的 EDA：看图，而不是看 64/784 列直方图
2. PCA 二维投影与方差解释
3. 混淆矩阵定位易混数字对，并抽查错误样本

## 数据
本地用 `sklearn.datasets.load_digits`（8×8）；Kaggle 28×28 同思路。
"""
            ),
            *env_cells(),
            md("## 1. 加载 digits"),
            code(
                """
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
print(X.shape, np.bincount(y))
"""
            ),
            md("## 2. 样本可视化"),
            code(
                """
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, img, lab in zip(axes.ravel(), X[:16].reshape(-1, 8, 8), y[:16]):
    ax.imshow(img, cmap='gray')
    ax.set_title(str(lab))
    ax.axis('off')
plt.suptitle('Sample digits')
plt.tight_layout()
plt.show()
"""
            ),
            md("## 3. PCA 投影"),
            code(
                """
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
Z = pca2.fit_transform(Xtr)
plt.figure(figsize=(6, 5))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=ytr, cmap='tab10', s=10)
plt.colorbar(sc)
plt.title(f'PCA 2D (var={pca2.explained_variance_ratio_.sum():.2%})')
plt.show()

pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(Xtr)
print('components for 95% variance:', pca.n_components_)
"""
            ),
            md("## 4. QuickBaseline"),
            code(
                """
from wanglaoshi import CompetitionTools as CT

Xtr_df = pd.DataFrame(Xtr)
Xte_df = pd.DataFrame(Xte)
baseline = CT.QuickBaseline()
results = baseline.run_all_models(
    Xtr_df, pd.Series(ytr), Xte_df, pd.Series(yte), task_type='classification'
)
comparison = baseline.compare_models(results, top_n=5)
comparison
"""
            ),
            md("## 5. 混淆矩阵与易混对"),
            code(
                """
from sklearn.metrics import classification_report, confusion_matrix

best = comparison.iloc[0]['Model']
pred = np.asarray(results[best]['predictions'])
print(classification_report(yte, pred))
cm = confusion_matrix(yte, pred)
print(cm)

# 找非对角最大混淆
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
print(f'most confused pair: true {i} -> pred {j} (count={cm_off[i, j]})')
"""
            ),
            md("## 6. 错误样本抽查"),
            code(
                """
wrong = np.where(pred != yte)[0][:8]
fig, axes = plt.subplots(1, max(1, len(wrong)), figsize=(12, 2))
if len(wrong) == 0:
    print('无错误样本')
else:
    if len(wrong) == 1:
        axes = [axes]
    for ax, idx in zip(axes, wrong):
        ax.imshow(Xte[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f't{yte[idx]} p{pred[idx]}')
        ax.axis('off')
    plt.show()
"""
            ),
            md("## 7. 模拟提交"),
            code(
                """
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

out = Path('outputs/12_digit_recognizer'); out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'ImageId': np.arange(1, len(pred) + 1), 'Label': yte})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head()
"""
            ),
            md(
                """## 8. 检查清单
- [x] 样本可视化
- [x] PCA 投影与方差解释
- [x] 混淆矩阵 + 错误抽查
- [x] Label 列提交校验
- [ ] 对易混对做专门增强/模型
"""
            ),
        ],
    )


def main() -> None:
    e01()
    e02()
    e03()
    e04()
    e05()
    e06()
    e07()
    e08()
    e09()
    e10()
    e11()
    e12()
    print("done: 12 detailed notebooks")


if __name__ == "__main__":
    main()
