"""Batch1：将 E01/E02/E04/E12 扩充为 ~3× 课程深度（Issue #9）。

用法：
  python examples/advanced/_deepen_curriculum_batch1.py

会覆盖 01_iris / 02_titanic / 04_breast_cancer / 12_digit_recognizer 四个 .ipynb。
"""

from __future__ import annotations

from _deepen_all import code, env_cells, md, write


# ---------------------------------------------------------------------------
# E01 Iris (Fisher 1936)
# ---------------------------------------------------------------------------
def e01() -> None:
    write(
        "01_iris",
        [
            md(
                """# E01 Iris（Fisher 1936）：多分类、决策边界与分层验证

> Issue #9 | [UCI Iris](https://archive.ics.uci.edu/ml/datasets/iris) | Fisher (1936) 线性判别分析经典数据

## 你将学到
1. **分类 vs 回归**：分类预测离散标签，回归预测连续量；决策区域由分类器在特征空间划分的区域决定（见 [sklearn 分类概览](https://scikit-learn.org/stable/supervised_learning.html#classification)）
2. **可线性分隔性**：*Iris setosa* 与其余两类几乎线性可分；*versicolor* 与 *virginica* 在花瓣特征上有重叠
3. **小样本多分类**：每类仅 50 条 → 必须 **分层** 划分与 CV，否则某折缺类、指标方差膨胀
4. **预处理泄漏**：`StandardScaler` 只能在训练折上 `fit`；用 `Pipeline` 把缩放与模型绑定
5. **提交格式**：整数 `target` vs 字符串 `species`；`prediction_type` 用 `label` / `proba` / `auto`（**禁止** `probability`）

## 技术难点
- 花瓣长度/宽度对判别贡献最大（萼片特征次要）
- 混淆矩阵读法：哪两类互混？macro-F1 对类不均衡更稳健
"""
            ),
            *env_cells(),
            md("## 0. 分类与回归：任务类型一览"),
            code(
                """
# 本 Notebook 是 **多分类**；若预测连续花瓣长度则是回归。
# 分类器学习 f: R^4 -> {0,1,2}，等价于在 R^4 中划出三个决策区域（decision regions）。
TASK = 'multiclass_classification'
print('task =', TASK)
"""
            ),
            md("## 1. 加载 Iris：数字标签 + 物种类名字符串"),
            code(
                """
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = iris.target_names[iris.target]
feature_cols = [c for c in df.columns if c not in ('target', 'species')]
print('shape =', df.shape)
print('features:', feature_cols)
print('target counts:\\n', df['target'].value_counts().sort_index())
print('species counts:\\n', df['species'].value_counts())
df.head()
"""
            ),
            md("## 2. EDA：描述统计与缺失"),
            code(
                """
from wanglaoshi.analyze import DataAnalyzer

analyzer = DataAnalyzer(df)
print(analyzer.get_basic_info())
display(analyzer.basic_statistics())
display(analyzer.missing_value_analysis())
"""
            ),
            md(
                """### 2.1 相关结构：花瓣特征主导判别

Pearson 相关系数衡量线性关联；与 `target` 相关性高的特征往往在 PCA/逻辑回归中权重大。
"""
            ),
            code(
                """
import matplotlib.pyplot as plt
import seaborn as sns

corr = df[feature_cols + ['target']].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Feature correlation (incl. target)')
plt.tight_layout()
plt.show()
petal_corr = corr['target'].drop('target').abs().sort_values(ascending=False)
print('|corr(target)| ranking:\\n', petal_corr)
"""
            ),
            md("### 2.2 配对图：Setosa 分离 vs Versicolor/Virginica 重叠"),
            code(
                """
# QUICK 下用全量 150 条仍可接受
g = sns.pairplot(
    df, hue='species', vars=feature_cols,
    corner=True, plot_kws={'s': 25, 'alpha': 0.8}, height=2.0
)
g.fig.suptitle('Pairplot by species', y=1.02)
plt.show()
"""
            ),
            md("### 2.3 二维切片：Petal length × Petal width"),
            code(
                """
plt.figure(figsize=(6, 4.5))
for sp, mk in zip(iris.target_names, ('o', 's', '^')):
    sub = df[df['species'] == sp]
    plt.scatter(sub['petal length (cm)'], sub['petal width (cm)'], label=sp, marker=mk, s=40, alpha=0.85)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.title('Decision-boundary intuition: setosa isolated on the left')
plt.grid(True, alpha=0.3)
plt.show()
"""
            ),
            md(
                """## 3. 分层划分：为什么小样本多分类必须 stratify？

若随机划分，验证集可能某一类过少甚至为 0，Accuracy / F1 会剧烈波动。
下面用 **类计数** 与 **CV 标准差** 对比分层 vs 非分层（后者仅作反面教材）。
"""
            ),
            code(
                """
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold

X = df[feature_cols]
y_int = df['target']
y_str = df['species']

Xtr, Xte, ytr, yte = train_test_split(
    X, y_int, test_size=0.3, random_state=RANDOM_STATE, stratify=y_int
)
print('✅ stratified train:', ytr.value_counts().sort_index().to_dict())
print('✅ stratified test :', yte.value_counts().sort_index().to_dict())

# 反面教材：固定 seed=7 且无 stratify（不用于正式评估）
_, _, ytr_bad, yte_bad = train_test_split(
    X, y_int, test_size=0.3, random_state=7, stratify=None
)
print('❌ non-strat train:', ytr_bad.value_counts().sort_index().to_dict())
print('❌ non-strat test :', yte_bad.value_counts().sort_index().to_dict())
"""
            ),
            md("### 3.1 交叉验证：分层 K 折 vs 普通 K 折的稳定性"),
            code(
                """
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
n_splits = 3 if QUICK else 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
acc_strat = cross_val_score(pipe, X, y_int, cv=skf, scoring='accuracy')
acc_plain = cross_val_score(pipe, X, y_int, cv=kf, scoring='accuracy')
print(f'StratifiedKFold accuracy: {acc_strat.mean():.3f} ± {acc_strat.std():.3f}')
print(f'KFold (no strat) accuracy: {acc_plain.mean():.3f} ± {acc_plain.std():.3f}')
print('通常分层 CV 的 std 更小 —— 更可信的泛化估计')
"""
            ),
            md(
                """## 4. 数据泄漏示范：在全量数据上 fit 缩放器

**错误**：`scaler.fit(X)` 用到了验证/测试集的均值方差 → 乐观偏差。  
**正确**：只在 `Xtr` 上 `fit`，或把 `StandardScaler` 放进 `Pipeline` 并在 CV 各折内 fit。
"""
            ),
            code(
                """
from sklearn.metrics import accuracy_score

# ❌ 泄漏：在全量 X 上 fit
from sklearn.preprocessing import StandardScaler as SS
bad_scaler = SS()
X_all_scaled = bad_scaler.fit_transform(X)
Xtr_bad, Xte_bad, _, _ = train_test_split(
    X_all_scaled, y_int, test_size=0.3, random_state=RANDOM_STATE, stratify=y_int
)
clf_bad = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
clf_bad.fit(Xtr_bad, ytr)
acc_leak = accuracy_score(yte, clf_bad.predict(Xte_bad))

# ✅ Pipeline：缩放仅在训练折 fit
clf_good = Pipeline([
    ('scaler', SS()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
clf_good.fit(Xtr, ytr)
acc_pipe = accuracy_score(yte, clf_good.predict(Xte))
print(f'accuracy with leaky scaling: {acc_leak:.3f}')
print(f'accuracy with Pipeline scaling: {acc_pipe:.3f}')
"""
            ),
            md("## 5. QuickBaseline：多模型对比（训练集上 fit，验证集评估）"),
            code(
                """
from wanglaoshi import compete as CT

baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr, ytr, Xte, yte, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("## 6. 混淆矩阵与 macro-F1 解读"),
            code(
                """
from sklearn.metrics import classification_report, confusion_matrix, f1_score

best = comparison.iloc[0]['Model']
pred_int = np.asarray(results[best]['predictions']).astype(int)
print(classification_report(yte, pred_int, target_names=list(iris.target_names)))
cm = confusion_matrix(yte, pred_int)
print('confusion matrix (rows=true, cols=pred):\\n', cm)
macro_f1 = f1_score(yte, pred_int, average='macro')
print('macro-F1 =', round(macro_f1, 4))
print('macro-F1 对每一类 F1 等权平均 —— 小类不被大类 Accuracy 淹没')
"""
            ),
            md("### 6.1 PCA 二维可视化：类别在主轴上的分离"),
            code(
                """
from sklearn.decomposition import PCA

pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
Z = pca_vis.fit_transform(Xtr)
plt.figure(figsize=(6, 5))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=ytr, cmap='tab10', s=35, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ticks=[0, 1, 2], label='target')
plt.xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]:.1%} var)')
plt.ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]:.1%} var)')
plt.title('PCA 2D on train (fit on train only)')
plt.grid(True, alpha=0.3)
plt.show()
print('累计解释方差:', pca_vis.explained_variance_ratio_.sum())
"""
            ),
            md("## 7. 提交：整数标签 vs 字符串物种名"),
            code(
                """
from pathlib import Path
from wanglaoshi import compete as CT

out = Path('outputs/01_iris')
out.mkdir(parents=True, exist_ok=True)
gen = CT.SubmissionGenerator()

# A) 整数 target
sample_int = pd.DataFrame({'Id': np.arange(len(pred_int)), 'target': yte.to_numpy()})
sample_int.to_csv(out / 'sample_int.csv', index=False)
sub_int = gen.create_submission(
    pred_int, str(out / 'sample_int.csv'), str(out / 'sub_int.csv'), prediction_type='label'
)
print('INT validate:', gen.validate_submission(str(out / 'sub_int.csv'), str(out / 'sample_int.csv')))

# B) 字符串 species
pred_str = iris.target_names[pred_int]
sample_str = pd.DataFrame({
    'Id': np.arange(len(pred_str)),
    'species': y_str.loc[yte.index].to_numpy(),
})
sample_str.to_csv(out / 'sample_str.csv', index=False)
sub_str = gen.create_submission(
    pred_str, str(out / 'sample_str.csv'), str(out / 'sub_str.csv'),
    prediction_type='label', target_column='species',
)
print('STR validate:', gen.validate_submission(str(out / 'sub_str.csv'), str(out / 'sample_str.csv')))
display(sub_int.head(3))
display(sub_str.head(3))
"""
            ),
            md("### 7.1 若需概率提交：使用 `prediction_type='proba'`"),
            code(
                """
# 禁止 prediction_type='probability' —— 库枚举为 auto | label | proba | multioutput
model = results[best]['model']
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(Xte)
    print('proba shape', proba.shape, '— 竞赛多列概率时用 prediction_type=\\'proba\\'')
else:
    print('当前最优模型无 predict_proba，本单元跳过')
"""
            ),
            md("### 7.2 二类子问题：Versicolor vs Virginica（最难边界）"),
            code(
                """
mask_two = df['species'].isin(['versicolor', 'virginica'])
X2 = df.loc[mask_two, feature_cols]
y2 = df.loc[mask_two, 'target']
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.3, random_state=RANDOM_STATE, stratify=y2
)
pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
pipe2.fit(X2_tr, y2_tr)
pred2 = pipe2.predict(X2_te)
print('binary versicolor/virginica only:')
print(classification_report(y2_te, pred2, target_names=['versicolor', 'virginica']))
print('macro-F1', f1_score(y2_te, pred2, average='macro'))
"""
            ),
            md("### 7.3 二维决策区域（仅花瓣两维，教学可视化）"),
            code(
                """
from matplotlib.colors import ListedColormap

pair = ['petal length (cm)', 'petal width (cm)']
X_vis = df[pair].to_numpy()
y_vis = df['target'].to_numpy()
Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
    X_vis, y_vis, test_size=0.3, random_state=RANDOM_STATE, stratify=y_vis
)
clf_vis = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf_vis.fit(Xv_tr, yv_tr)
x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
Zmesh = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Zmesh, alpha=0.25, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(Xv_te[:, 0], Xv_te[:, 1], c=yv_te, cmap=ListedColormap(['r', 'g', 'b']), edgecolors='k', s=45)
plt.xlabel(pair[0])
plt.ylabel(pair[1])
plt.title('Decision regions (logistic, petal features only)')
plt.show()
"""
            ),
            md("### 7.4 混淆矩阵热力图与 per-class F1"),
            code(
                """
import seaborn as sns

plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('predicted')
plt.ylabel('true')
plt.title('Confusion matrix heatmap')
plt.tight_layout()
plt.show()
per_f1 = f1_score(yte, pred_int, average=None)
for name, f1v in zip(iris.target_names, per_f1):
    print(f'F1[{name}] = {f1v:.3f}')
"""
            ),
            md("### 7.5 重复分层 CV：accuracy / macro-F1 双指标"),
            code(
                """
from sklearn.model_selection import cross_validate

scoring = {'acc': 'accuracy', 'f1_macro': 'f1_macro'}
cv_out = cross_validate(pipe, X, y_int, cv=skf, scoring=scoring)
print('CV accuracy', cv_out['test_acc'].mean(), '±', cv_out['test_acc'].std())
print('CV macro-F1', cv_out['test_f1_macro'].mean(), '±', cv_out['test_f1_macro'].std())
"""
            ),
            md(
                """### 7.6 Fisher LDA 历史注记

Fisher (1936) 在线性判别框架下用 Iris 说明类间可分性与投影方向；现代 `LogisticRegression` / SVM 是同一决策几何的不同估计方式。参见 [Fisher Iris 原论文摘要](https://aspace.agilis-network.com/handle/20.500.12142/2007)。
"""
            ),
            code(
                """
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis()),
])
lda.fit(Xtr, ytr)
lda_acc = lda.score(Xte, yte)
print('LDA test accuracy:', lda_acc)
coef = lda.named_steps['lda'].coef_
print('LDA coef shape', coef.shape, '— each row is class vs rest direction')
"""
            ),
            md(
                """## 8. 检查清单
- [x] 分类任务与决策区域直觉
- [x] 花瓣特征主导 + Setosa 可分
- [x] 分层划分 vs 非分层 + CV std 对比
- [x] Pipeline 避免缩放泄漏
- [x] 混淆矩阵 + macro-F1
- [x] PCA 2D 类别可视化
- [x] 整数/字符串 SubmissionGenerator
- [ ] 对 versicolor/virginica 专门做二元分类实验
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
                """# E02 Titanic：缺失机制、特征工程与训练/验证隔离

> Issue #9 | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) | 本案例用 `seaborn.load_dataset('titanic')` **离线可跑**

## 你将学到
1. **生存率 EDA**：性别、舱位、年龄、票价、登船港口与 `survived` 的关系
2. **缺失机制**：MCAR/MAR/MNAR 直觉 —— Age/Embarked/Deck 不能盲目用全表统计填充
3. **泄漏列**：`alive` 几乎等于标签，必须删除后再建模
4. **特征工程**：Title（有 `name` 则解析，否则用 `who`）、FamilySize、IsAlone、Deck —— **映射只在训练集拟合**
5. **`LeakageDetector`**：编码后扫描异常高相关特征
6. **错误 vs 正确填充**：全表 median 填充会虚高离线分

## 提交
竞赛列名常为 `Survived`；`prediction_type='label'`。
"""
            ),
            *env_cells(),
            md("## 1. 加载数据与列概览"),
            code(
                """
import seaborn as sns
import pandas as pd
import numpy as np

raw = sns.load_dataset('titanic').copy()
print('shape', raw.shape)
print('columns:', list(raw.columns))
raw.head(3)
"""
            ),
            md("## 2. 删除泄漏列 `alive`"),
            code(
                """
leak_cols = [c for c in ['alive'] if c in raw.columns]
df = raw.drop(columns=leak_cols)
if leak_cols:
    print('dropped leak columns:', leak_cols)
if 'alive' in raw.columns:
    print('corr(alive, survived) =', raw['alive'].map({'yes': 1, 'no': 0}).corr(raw['survived']))
print('remaining cols:', list(df.columns))
"""
            ),
            md("## 3. EDA：目标分布与缺失"),
            code(
                """
from wanglaoshi.analyze import DataAnalyzer

analyzer = DataAnalyzer(df)
display(analyzer.missing_value_analysis())
display(analyzer.basic_statistics().head(12))
print('overall survival rate:', df['survived'].mean().round(3))
"""
            ),
            md("### 3.1 生存率：Sex / Pclass"),
            code(
                """
rate_sex = df.groupby('sex')['survived'].agg(['mean', 'count'])
rate_class = df.groupby('class')['survived'].agg(['mean', 'count'])
print('by sex:\\n', rate_sex)
print('by class:\\n', rate_class)
"""
            ),
            md("### 3.2 生存率：Age 分箱 / Fare / Embarked"),
            code(
                """
import matplotlib.pyplot as plt

work = df.copy()
work['age_bin'] = pd.cut(work['age'], bins=[0, 12, 18, 35, 60, 100], labels=['child','teen','adult','mid','senior'])
print('by age_bin:\\n', work.groupby('age_bin', observed=True)['survived'].mean())
print('by embarked:\\n', work.groupby('embarked')['survived'].mean())
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
work.boxplot(column='fare', by='survived', ax=ax[0])
ax[0].set_title('Fare by survived')
work.groupby('embarked')['survived'].mean().plot(kind='bar', ax=ax[1], rot=0)
ax[1].set_title('Survival rate by embarked')
plt.suptitle('')
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """## 4. 划分训练/验证（stratify survived）

此后所有统计量（中位数、众数、频率编码）**只能来自训练集**。
"""
            ),
            code(
                """
from sklearn.model_selection import train_test_split

y = df['survived']
idx_train, idx_valid = train_test_split(
    df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
train_df = df.loc[idx_train].copy()
valid_df = df.loc[idx_valid].copy()
print('train survival', train_df['survived'].mean(), 'valid', valid_df['survived'].mean())
print('train size', len(train_df), 'valid size', len(valid_df))
"""
            ),
            md("## 5. 特征工程（fit on train only）"),
            code(
                """
import re

def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Title
    if 'name' in out.columns:
        out['title'] = out['name'].astype(str).apply(
            lambda s: re.search(r',\\s*([^\\.]+)\\.', s).group(1).strip() if re.search(r',\\s*([^\\.]+)\\.', s) else 'Unknown'
        )
    else:
        out['title'] = out['who'].astype(str)  # seaborn 版无 name，用 who 近似
    out['family_size'] = out['sibsp'].fillna(0) + out['parch'].fillna(0) + 1
    out['is_alone'] = (out['family_size'] == 1).astype(int)
    if 'deck' in out.columns:
        out['deck_clean'] = out['deck'].astype(str).fillna('Unknown')
    else:
        out['deck_clean'] = 'Unknown'
    return out

train_fe = add_engineered_features(train_df)
valid_fe = add_engineered_features(valid_df)
print('title counts (train top):\\n', train_fe['title'].value_counts().head(8))
"""
            ),
            md("### 5.1 缺失填充：仅用训练集 Age median、Embarked mode"),
            code(
                """
age_med = train_fe['age'].median()
emb_mode = train_fe['embarked'].mode(dropna=True).iloc[0] if train_fe['embarked'].notna().any() else 'S'

for part in (train_fe, valid_fe):
    part['age'] = part['age'].fillna(age_med)
    part['embarked'] = part['embarked'].fillna(emb_mode)
    part['deck_clean'] = part['deck_clean'].replace('nan', 'Unknown')

print('NA after fill — train', train_fe['age'].isna().sum(), 'valid', valid_fe['age'].isna().sum())
print('age median (train-only):', age_med, 'embarked mode:', emb_mode)
"""
            ),
            md("### 5.2 类别频率编码：映射来自训练集"),
            code(
                """
cat_cols = ['sex', 'embarked', 'class', 'who', 'embark_town', 'title', 'deck_clean']
cat_cols = [c for c in cat_cols if c in train_fe.columns]

def freq_encode(tr: pd.DataFrame, va: pd.DataFrame, cols: list) -> tuple:
    tr = tr.copy()
    va = va.copy()
    for c in cols:
        freq = tr[c].astype(str).value_counts(normalize=True)
        tr[c + '_freq'] = tr[c].astype(str).map(freq).fillna(0)
        va[c + '_freq'] = va[c].astype(str).map(freq).fillna(0)
    return tr, va

train_enc, valid_enc = freq_encode(train_fe, valid_fe, cat_cols)
num_cols = ['age', 'sibsp', 'parch', 'fare', 'family_size', 'is_alone']
num_cols += [c for c in train_enc.columns if c.endswith('_freq')]
Xtr = train_enc[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
Xva = valid_enc[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
ytr, yva = train_fe['survived'], valid_fe['survived']
print('feature matrix', Xtr.shape)
Xtr.head(3)
"""
            ),
            md("## 6. 错误 vs 正确填充（叙事对比）"),
            code(
                """
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ❌ 错误：在全表 df 上算 median 再填充后划分（信息泄漏 + 填充泄漏）
wrong_med = df['age'].median()
X_wrong = df.copy()
X_wrong['age'] = X_wrong['age'].fillna(wrong_med)
num_simple = ['age', 'sibsp', 'parch', 'fare']
Xw = X_wrong[num_simple]
yw = df['survived']
Xw_tr, Xw_va, yw_tr, yw_va = train_test_split(
    Xw, yw, test_size=0.25, random_state=RANDOM_STATE, stratify=yw
)
clf_wrong = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
clf_wrong.fit(Xw_tr, yw_tr)
acc_wrong = accuracy_score(yw_va, clf_wrong.predict(Xw_va))

# ✅ 正确：本 Notebook 第 5 节的 Xtr/Xva
clf_right = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
clf_right.fit(Xtr, ytr)
acc_right = accuracy_score(yva, clf_right.predict(Xva))
print(f'wrong-fill baseline accuracy: {acc_wrong:.3f}')
print(f'right-fill baseline accuracy: {acc_right:.3f}')
print('全表填充可能略乐观 —— 竞赛要用 train-only 统计')
"""
            ),
            md("## 7. LeakageDetector：可疑高相关特征"),
            code(
                """
from wanglaoshi import compete as CT

detector = CT.LeakageDetector()
leaks = detector.detect_target_leakage(Xtr, ytr, threshold=0.85, method='correlation')
print('suspicious features (corr>0.85):', leaks if leaks else 'none — good')
# 若仍有 alive 残留列会在此告警
"""
            ),
            md("## 8. QuickBaseline"),
            code(
                """
baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr, ytr, Xva, yva, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("### 7.1 FeatureEngineering 模块（可选辅助）"),
            code(
                """
from wanglaoshi import features as FE

eng = FE.AdvancedFeatureEngineer()
# 演示：对训练集数值列做安全预览（不在此替换主流程）
preview = train_enc[num_cols].copy()
print('AdvancedFeatureEngineer loaded:', type(eng).__name__)
print('numeric feature preview shape', preview.shape)
"""
            ),
            md("### 7.2 生存率可视化汇总"),
            code(
                """
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
df.groupby('sex')['survived'].mean().plot(kind='bar', ax=axes[0, 0], rot=0, color=['#4477AA', '#CC6677'])
axes[0, 0].set_title('Survival by sex')
df.groupby('class')['survived'].mean().plot(kind='bar', ax=axes[0, 1], rot=45)
axes[0, 1].set_title('Survival by class')
df.groupby('embarked')['survived'].mean().plot(kind='bar', ax=axes[1, 0], rot=0)
axes[1, 0].set_title('Survival by embarked')
df.assign(age_filled=df['age'].fillna(df['age'].median())).boxplot(
    column='fare', by='survived', ax=axes[1, 1]
)
axes[1, 1].set_title('Fare by survived')
plt.suptitle('Titanic survival EDA panel')
plt.tight_layout()
plt.show()
"""
            ),
            md("### 7.3 缺失 Age 与生存（训练集视角）"),
            code(
                """
train_plot = train_df.copy()
train_plot['age_missing'] = train_plot['age'].isna()
print(train_plot.groupby('age_missing')['survived'].agg(['mean', 'count']))
print('→ Age 缺失可能与生存相关 —— 应用训练集中位数填充而非 0')
"""
            ),
            md("## 9. 模拟提交（Survived 列）"),
            code(
                """
from pathlib import Path

best = comparison.iloc[0]['Model']
pred = results[best]['predictions']
out = Path('outputs/02_titanic')
out.mkdir(parents=True, exist_ok=True)
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
            md("### 9.1 StratifiedKFold 线下分（均值 ± 标准差）"),
            code(
                """
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

full_X = pd.concat([Xtr, Xva], axis=0)
full_y = pd.concat([ytr, yva], axis=0)
rf = RandomForestClassifier(n_estimators=80, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=3 if QUICK else 5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, full_X, full_y, cv=cv, scoring='accuracy')
print(f'Titanic RF stratified CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
"""
            ),
            md("### 9.2 特征分布对比（train vs valid）"),
            code(
                """
for col in ['fare', 'age', 'family_size']:
    print(col, 'train mean', round(train_fe[col].mean(), 2), 'valid mean', round(valid_fe[col].mean(), 2))
print('若 valid 与 train 分布差很大 —— 慎用过复杂的 train-only 映射')
"""
            ),
            md(
                """## 10. 检查清单
- [x] Sex/Pclass/Age/Fare/Embarked 生存 EDA
- [x] 删除 alive 泄漏列
- [x] Title/FamilySize/IsAlone/Deck 工程（train-only 统计）
- [x] 错误 vs 正确 Age 填充对比
- [x] LeakageDetector
- [x] QuickBaseline + Survived 提交
- [ ] Cabin 文本深挖与折内目标编码
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
                """# E04 Breast Cancer（Wisconsin）：量纲、PCA 与召回权衡

> Issue #9 | [UCI Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

## 你将学到
1. **医学二分类**：恶性漏诊（假阴性 FN）代价远高于误报（FP）→ 优先看 **Recall**
2. **量纲与缩放**：radius/area 等量纲差异大；距离/线性模型前必须 `StandardScaler`
3. **PCA 理论**：用正交主轴压缩方差；`mean_*` / `worst_*` / `error_*` 高度共线（多重共线性）
4. **`FeatureImportanceAnalyzer`**：`method='builtin'` 读取树模型分裂增益
5. **分层 CV + 混淆矩阵**：FN 即「将恶性判为良性」

## 指标
- Precision–Recall 权衡；阈值非默认 0.5
- `prediction_type='label'` 提交练习
"""
            ),
            *env_cells(),
            md("## 1. 加载数据"),
            code(
                """
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
print(df.shape)
print('target counts:\\n', df['target'].value_counts())
print('classes:', list(data.target_names), '— 通常 0=malignant, 1=benign（以 sklearn 为准）')
df.head(2)
"""
            ),
            md("## 2. EDA：量纲跨度与缺失"),
            code(
                """
from wanglaoshi.analyze import DataAnalyzer

analyzer = DataAnalyzer(df)
display(analyzer.basic_statistics().head(10))
display(analyzer.missing_value_analysis())
feat = df.drop(columns=['target'])
print('mean feature scale ratio max/min:', feat.mean().max() / max(feat.mean().min(), 1e-9))
"""
            ),
            md(
                """### 2.1 多重共线性：mean / worst / error 特征族

同一几何量（如 radius）的均值、标准误、极值往往高度相关，线性模型系数不稳定 → PCA 或正则化。
"""
            ),
            code(
                """
import matplotlib.pyplot as plt
import seaborn as sns

cols_sample = [c for c in feat.columns if 'radius' in c or 'area' in c][:6]
if len(cols_sample) >= 2:
    plt.figure(figsize=(5, 4))
    sns.heatmap(feat[cols_sample].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Multicollinearity among radius/area family')
    plt.tight_layout()
    plt.show()
"""
            ),
            md("## 3. 分层划分"),
            code(
                """
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X = feat
y = df['target']
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print('train counts', ytr.value_counts().to_dict())
print('test counts ', yte.value_counts().to_dict())
"""
            ),
            md("## 4. Pipeline：Scaler → PCA → LogisticRegression"),
            code(
                """
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

n_comp = 10 if QUICK else 15
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
print(classification_report(yte, pred, target_names=list(data.target_names)))
print('ROC-AUC', roc_auc_score(yte, proba))
print('AP', average_precision_score(yte, proba))
print('PCA n_components=', n_comp)
"""
            ),
            md("### 4.1 PCA 方差解释（在训练集 fit）"),
            code(
                """
pca_full = PCA(random_state=RANDOM_STATE).fit(StandardScaler().fit_transform(Xtr))
cum = np.cumsum(pca_full.explained_variance_ratio_)
k95 = int(np.searchsorted(cum, 0.95) + 1)
print(f'components for 95% variance: {k95} / {Xtr.shape[1]}')
"""
            ),
            md("## 5. 阈值扫描：提高 Recall（少漏诊）"),
            code(
                """
from sklearn.metrics import recall_score, precision_score, f1_score

rows = []
for thr in [0.15, 0.25, 0.35, 0.5, 0.65]:
    p = (proba >= thr).astype(int)
    rows.append({
        'thr': thr,
        'precision': precision_score(yte, p, zero_division=0),
        'recall': recall_score(yte, p, zero_division=0),
        'f1': f1_score(yte, p, zero_division=0),
    })
thr_df = pd.DataFrame(rows)
display(thr_df)

from sklearn.metrics import precision_recall_curve
prec, rec, _ = precision_recall_curve(yte, proba)
plt.figure(figsize=(5, 4))
plt.plot(rec, prec)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve — move along curve = change threshold')
plt.grid(True, alpha=0.3)
plt.show()
"""
            ),
            md("## 6. 混淆矩阵：假阴性 FN 的代价"),
            code(
                """
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(yte, pred)
print('confusion matrix [[TN, FP], [FN, TP]] assuming 0=malignant, 1=benign:')
print(cm)
print('FN = 恶性被判良性 —— 医学场景最危险')
"""
            ),
            md("## 7. QuickBaseline（缩放后全特征）"),
            code(
                """
from wanglaoshi import compete as CT

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)
baseline = CT.QuickBaseline()
results = baseline.run_all_models(Xtr_s, ytr, Xte_s, yte, task_type='classification')
comparison = baseline.compare_models(results, top_n=6)
comparison
"""
            ),
            md("## 8. FeatureImportanceAnalyzer（builtin，树模型）"),
            code(
                """
from sklearn.ensemble import RandomForestClassifier

best = comparison.iloc[0]['Model']
model = results[best]['model']
# 若最优非树模型，换 RF 演示 builtin
if not hasattr(model, 'feature_importances_'):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(Xtr_s, ytr)

fia = CT.FeatureImportanceAnalyzer()
imp = fia.calculate_importance(model, Xtr_s, ytr, method='builtin')
display(imp.head(12))
"""
            ),
            md("### 8.1 分层交叉验证稳定性"),
            code(
                """
from sklearn.ensemble import RandomForestClassifier as RFC

clf_cv = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RFC(n_estimators=50 if QUICK else 100, random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(n_splits=3 if QUICK else 5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(clf_cv, X, y, cv=cv, scoring='roc_auc')
print(f'stratified CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}')
"""
            ),
            md("### 8.2 全特征相关矩阵抽样（共线强度）"),
            code(
                """
sample_feats = list(feat.columns[:12])
plt.figure(figsize=(6, 5))
sns.heatmap(feat[sample_feats].corr(), cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Feature block correlation (first 12 cols)')
plt.tight_layout()
plt.show()
"""
            ),
            md("### 8.3 无 PCA vs 有 PCA：Quick 对比"),
            code(
                """
from sklearn.metrics import accuracy_score

pipe_full = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
pipe_full.fit(Xtr, ytr)
acc_full = accuracy_score(yte, pipe_full.predict(Xte))
acc_pca = accuracy_score(yte, pred)
print('test accuracy no-PCA logistic:', acc_full)
print('test accuracy PCA logistic:  ', acc_pca)
print('PCA 降维可能略损精度但加速推理、缓解共线')
"""
            ),
            md("### 8.4 Permutation importance（子集，QUICK）"),
            code(
                """
_sub = Xtr_s.sample(min(80, len(Xtr_s)), random_state=RANDOM_STATE)
imp_perm = fia.calculate_importance(
    model, _sub, ytr.loc[_sub.index],
    method='permutation', n_repeats=3 if QUICK else 5,
)
display(imp_perm.head(10))
"""
            ),
            md("## 9. 模拟提交"),
            code(
                """
from pathlib import Path

pred_best = np.asarray(results[best]['predictions'])
out = Path('outputs/04_breast_cancer')
out.mkdir(parents=True, exist_ok=True)
sample = pd.DataFrame({'Id': np.arange(len(pred_best)), 'target': yte.to_numpy()})
sample.to_csv(out / 'sample_submission.csv', index=False)
gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    pred_best, str(out / 'sample_submission.csv'), str(out / 'submission.csv'), prediction_type='label'
)
print(gen.validate_submission(str(out / 'submission.csv'), str(out / 'sample_submission.csv')))
sub.head()
"""
            ),
            md("### 9.1 敏感性：降低阈值对 FN 的影响"),
            code(
                """
thr_low = 0.25
pred_low = (proba >= thr_low).astype(int)
cm_default = confusion_matrix(yte, pred)
cm_low = confusion_matrix(yte, pred_low)
# sklearn: target 0=malignant, 1=benign —— FN = 恶性被判良性
fn_default = int(cm_default[0, 1])
fn_low = int(cm_low[0, 1])
print('FN count default thr 0.5:', fn_default)
print('FN count low thr 0.25:', fn_low)
print('扩大 Recall 通常减少 FN（恶性漏检）但增加 FP')
"""
            ),
            md("### 9.2 worst-radius 家族特征散点"),
            code(
                """
wcol = 'worst radius' if 'worst radius' in feat.columns else feat.columns[0]
plt.figure(figsize=(5, 3.5))
for t in sorted(df['target'].unique()):
    plt.hist(df.loc[df['target'] == t, wcol], bins=20, alpha=0.5, label=data.target_names[int(t)])
plt.xlabel(wcol)
plt.ylabel('count')
plt.title('Feature distribution by diagnosis')
plt.legend()
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """## 10. 检查清单
- [x] 医学 Recall 与阈值
- [x] 缩放 + PCA + 共线性讨论
- [x] FeatureImportanceAnalyzer builtin
- [x] 分层 CV + FN 代价
- [x] SubmissionGenerator
- [ ] 概率校准后再定阈值
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
                """# E12 Digit Recognizer：高维向量、降维直觉与错分分析

> Issue #9 | 本地 `sklearn.datasets.load_digits`（8×8）；Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) 为 28×28，流程相同

## 你将学到
1. **图像即向量**：每张图拉平为 64 维；高维空间稀疏，直接逐列直方图 EDA 低效 → **先看图**
2. **PCA / t-SNE 直觉**：PCA 保留全局方差；t-SNE 强调局部邻域（非线性，计算贵，QUICK 下对子集演示）
3. **易混数字对**：3/5、8/5 等在低分辨率下形态接近
4. **分层划分** + QuickBaseline
5. **错分可视化**：从混淆矩阵定位再抽查图像

## 提交
`Label` 列 + `prediction_type='label'`
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
print('X', X.shape, '— each row is 8x8=64 pixels flattened')
print('class counts:\\n', pd.Series(y).value_counts().sort_index())
"""
            ),
            md("## 2. 像素 EDA：均值图与每类一例"),
            code(
                """
mean_img = X.mean(axis=0).reshape(8, 8)
plt.figure(figsize=(3, 3))
plt.imshow(mean_img, cmap='gray')
plt.title('Mean digit (all classes)')
plt.axis('off')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit, ax in zip(range(10), axes.ravel()):
    idx = np.where(y == digit)[0][0]
    ax.imshow(X[idx].reshape(8, 8), cmap='gray')
    ax.set_title(f'digit {digit}')
    ax.axis('off')
plt.suptitle('One example per class')
plt.tight_layout()
plt.show()
"""
            ),
            md("## 3. 分层 train/test"),
            code(
                """
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print('train bincount', np.bincount(ytr))
print('test  bincount', np.bincount(yte))
"""
            ),
            md(
                """## 4. PCA：二维投影与方差解释

[PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca) 在正交方向上最大化方差；2D 投影便于观察类簇是否可分。
"""
            ),
            code(
                """
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
Z = pca2.fit_transform(Xtr)
plt.figure(figsize=(6, 5))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=ytr, cmap='tab10', s=12, alpha=0.85)
plt.colorbar(sc)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA 2D on train (var={pca2.explained_variance_ratio_.sum():.1%})')
plt.grid(True, alpha=0.3)
plt.show()

pca95 = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(Xtr)
print('n_components for 95% variance:', pca95.n_components_)
"""
            ),
            md("### 4.1 t-SNE 直觉（QUICK：子集 300 条）"),
            code(
                """
from sklearn.manifold import TSNE

n_sub = min(300, len(Xtr))
rng = np.random.RandomState(RANDOM_STATE)
sub_idx = rng.choice(len(Xtr), size=n_sub, replace=False)
X_sub, y_sub = Xtr[sub_idx], ytr[sub_idx]
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=300 if QUICK else 1000)
Zt = tsne.fit_transform(X_sub)
plt.figure(figsize=(6, 5))
plt.scatter(Zt[:, 0], Zt[:, 1], c=y_sub, cmap='tab10', s=15)
plt.title('t-SNE 2D (subset) — local neighborhood emphasis')
plt.colorbar()
plt.show()
"""
            ),
            md("## 5. QuickBaseline"),
            code(
                """
from wanglaoshi import compete as CT

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
            md("## 6. 分类报告与混淆矩阵"),
            code(
                """
from sklearn.metrics import classification_report, confusion_matrix

best = comparison.iloc[0]['Model']
pred = np.asarray(results[best]['predictions']).astype(int)
print(classification_report(yte, pred))
cm = confusion_matrix(yte, pred)
print(cm)
"""
            ),
            md("### 6.1 易混对：非对角最大项"),
            code(
                """
cm_off = cm.copy().astype(int)
np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
print(f'most confused: true {i} predicted as {j} (count={cm_off[i, j]})')
for pair in [(3, 5), (5, 8), (8, 3)]:
    a, b = pair
    if a < cm.shape[0] and b < cm.shape[1]:
        print(f'  {a}->{b}: {cm[a, b]} errors')
"""
            ),
            md("## 7. 错分样本可视化"),
            code(
                """
wrong_idx = np.where(pred != yte)[0]
print('n_errors', len(wrong_idx))
show = wrong_idx[: min(12, len(wrong_idx))]
ncols = max(1, len(show))
fig, axes = plt.subplots(1, ncols, figsize=(1.4 * ncols, 2.2))
if ncols == 1:
    axes = [axes]
for ax, wi in zip(axes, show):
    ax.imshow(Xte[wi].reshape(8, 8), cmap='gray')
    ax.set_title(f't={yte[wi]} p={pred[wi]}', fontsize=9)
    ax.axis('off')
plt.suptitle('Misclassified digits')
plt.tight_layout()
plt.show()
"""
            ),
            md("### 7.1 分层 CV 基线稳定性"),
            code(
                """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5)),
])
cv = StratifiedKFold(n_splits=3 if QUICK else 5, shuffle=True, random_state=RANDOM_STATE)
acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
print(f'KNN stratified CV accuracy: {acc.mean():.3f} ± {acc.std():.3f}')
"""
            ),
            md("### 7.2 每类平均字形（centroid 图像）"),
            code(
                """
fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
for d, ax in zip(range(10), axes.ravel()):
    centroid = X[y == d].mean(axis=0).reshape(8, 8)
    ax.imshow(centroid, cmap='gray')
    ax.set_title(f'avg {d}')
    ax.axis('off')
plt.suptitle('Class mean images (centroids)')
plt.tight_layout()
plt.show()
"""
            ),
            md("### 7.3 每类准确率与混淆热力图"),
            code(
                """
import seaborn as sns

per_class_acc = []
for d in range(10):
    m = yte == d
    per_class_acc.append((pred[m] == d).mean() if m.any() else np.nan)
print('per-class accuracy:', dict(enumerate(np.round(per_class_acc, 3))))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.xlabel('predicted')
plt.ylabel('true')
plt.title('Digits confusion heatmap')
plt.tight_layout()
plt.show()
"""
            ),
            md("### 7.4 像素方差最高的维度（信息量大的像素）"),
            code(
                """
var_px = X.var(axis=0).reshape(8, 8)
plt.figure(figsize=(4, 4))
plt.imshow(var_px, cmap='hot')
plt.title('Per-pixel variance across dataset')
plt.colorbar(fraction=0.046)
plt.axis('off')
plt.show()
print('top-5 variable pixel indices:', np.argsort(X.var(axis=0))[-5:][::-1])
"""
            ),
            md("## 8. 模拟提交"),
            code(
                """
from pathlib import Path

out = Path('outputs/12_digit_recognizer')
out.mkdir(parents=True, exist_ok=True)
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
            md("### 8.1 SVM 与 KNN 在 PCA 子空间的快速对比"),
            code(
                """
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

n_pca = 20 if QUICK else 30
pca_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_pca, random_state=RANDOM_STATE)),
    ('clf', LinearSVC(random_state=RANDOM_STATE, max_iter=5000)),
])
pca_clf.fit(Xtr, ytr)
print('LinearSVC + PCA acc', accuracy_score(yte, pca_clf.predict(Xte)))
knn_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_pca, random_state=RANDOM_STATE)),
    ('clf', KNeighborsClassifier(n_neighbors=3)),
])
knn_pca.fit(Xtr, ytr)
print('KNN + PCA acc', accuracy_score(yte, knn_pca.predict(Xte)))
"""
            ),
            md("### 8.2 展示一组训练样本栅格（加深图像 EDA）"),
            code(
                """
rng = np.random.RandomState(RANDOM_STATE)
ix = rng.choice(len(Xtr), size=min(32, len(Xtr)), replace=False)
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for ax, i in zip(axes.ravel(), ix):
    ax.imshow(Xtr[i].reshape(8, 8), cmap='gray')
    ax.set_title(str(ytr[i]), fontsize=8)
    ax.axis('off')
plt.suptitle('Random train batch')
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """## 9. 检查清单
- [x] 高维向量 + 图像 EDA
- [x] PCA 与 t-SNE 直觉
- [x] 混淆矩阵 + 3/5/8 易混讨论
- [x] 错分图 + 分层 CV
- [x] Label 提交
- [ ] 对易混对做数据增强或专用二分类器
"""
            ),
        ],
    )


def main() -> None:
    e01()
    e02()
    e04()
    e12()
    print("batch1 done: 01_iris, 02_titanic, 04_breast_cancer, 12_digit_recognizer")


if __name__ == "__main__":
    main()
