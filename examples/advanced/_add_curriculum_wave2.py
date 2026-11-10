#!/usr/bin/env python3
"""新增 E13–E20：补齐常见数据分析方法覆盖。

运行：
  PYTHONPATH=. python examples/advanced/_add_curriculum_wave2.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def _nb(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    return nb


def _write(slug: str, cells: list) -> None:
    path = OUT / f"{slug}.ipynb"
    nb = _nb(cells)
    nbformat.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")


def e13() -> None:
    slug = "13_ab_testing"
    cells = [
        new_markdown_cell(
            f"""# E13 A/B 试验：功效、效应量与两比例检验

> 覆盖矩阵：试验设计 · Issue #9 扩展 · 见 `docs/ANALYSIS_COVERAGE.md`

## 你将学到
1. **随机分流假设**：可交换性；偷看 / 多重比较如何毁掉结论。
2. **效应量 vs p 值**：提升百分点、相对提升、Cohen 类度量。
3. **样本量 / 功效**：给定基线转化率与 MDE，反推每臂需要多少人。
4. **两比例 z / χ²**：与 `scipy.stats` 对齐；**wanglaoshi 尚无 `ExperimentDesign`（缺口 P0）**。
5. 用 `DataAnalyzer` 做试验日志表的结构体检。

## 包侧优化启示
- 缺统一 `ExperimentDesign.sample_size / two_proportion_test / report`。
- 报告宜输出 `StageResult` 风格（效应、CI、功效、假设检查）。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell(
            """## 课程研讨笔录（E13）

### 课堂问答题
1. 基线转化率 10%，想检出绝对 +1pp（相对 +10%），α=0.05、功效 80%，每臂大约多少？
2. 为什么「边看边停」会抬高一类错误？
3. 若用户可重复进组，单位是会话还是用户？为何影响方差？

### wanglaoshi 缺口
当前课用 `scipy`/`statsmodels` 手写；目标 API 草图：`wanglaoshi.stats.ExperimentDesign`。
"""
        ),
        new_markdown_cell("## 1. 合成试验日志 + DataAnalyzer"),
        new_code_cell(
            """from pathlib import Path
import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
n_a, n_b = 8000, 8000
p_a, p_b = 0.10, 0.112  # 绝对 +1.2pp
df = pd.DataFrame({
    'user_id': np.arange(n_a + n_b),
    'variant': ['A'] * n_a + ['B'] * n_b,
    'converted': np.concatenate([
        rng.binomial(1, p_a, n_a),
        rng.binomial(1, p_b, n_b),
    ]),
    'revenue': np.concatenate([
        rng.lognormal(1.0, 0.8, n_a) * rng.binomial(1, p_a, n_a),
        rng.lognormal(1.05, 0.8, n_b) * rng.binomial(1, p_b, n_b),
    ]),
})
print(df.groupby('variant')['converted'].agg(['mean', 'count']))
info = DataAnalyzer(df).get_basic_info()
print({k: info[k] for k in list(info)[:8]})
"""
        ),
        new_markdown_cell("## 2. 两比例检验与置信区间"),
        new_code_cell(
            """from scipy import stats

def two_prop_z(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci = (p2 - p1) + np.array([-1, 1]) * 1.96 * se_unpooled
    return dict(p1=p1, p2=p2, lift_pp=(p2 - p1) * 100, z=z, pvalue=pval, ci95=ci)

g = df.groupby('variant')['converted']
x_a, n_a_ = int(g.get_group('A').sum()), int(g.get_group('A').count())
x_b, n_b_ = int(g.get_group('B').sum()), int(g.get_group('B').count())
res = two_prop_z(x_a, n_a_, x_b, n_b_)
print(res)
# χ² 独立性（2x2）
table = pd.crosstab(df['variant'], df['converted'])
chi2, p_chi, dof, exp = stats.chi2_contingency(table)
print('chi2', chi2, 'p', p_chi, 'dof', dof)
"""
        ),
        new_markdown_cell("## 3. 样本量与功效（正态近似）"),
        new_code_cell(
            """def n_per_arm(p1, mde_abs, alpha=0.05, power=0.8):
    '''两臂等权、双侧、正态近似下的每臂样本量。'''
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    p2 = p1 + mde_abs
    es = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=es, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')
    return int(np.ceil(n)), es

n_need, es = n_per_arm(0.10, 0.01)
print('每臂约需', n_need, 'effectsize(h)=', round(es, 4))
# 当前试验功效（事后）
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
es_obs = proportion_effectsize(res['p1'], res['p2'])
pow_obs = NormalIndPower().power(es_obs, nobs1=n_a_, alpha=0.05, ratio=n_b_ / n_a_)
print('观测效应 h=', round(es_obs, 4), '事后功效≈', round(pow_obs, 3))
"""
        ),
        new_markdown_cell("## 4. 多重比较陷阱（演示）"),
        new_code_cell(
            """# 对 20 个无关指标做检验：在真 H0 下仍会看到「显著」
false_hits = 0
for i in range(20):
    fake = rng.binomial(1, 0.1, size=len(df))
    t = pd.crosstab(df['variant'], fake)
    _, p, _, _ = stats.chi2_contingency(t)
    false_hits += int(p < 0.05)
print('20 次无关检验中 p<0.05 次数:', false_hits, '（期望≈1）')
print('启示：预注册主指标；次指标用 FDR / Bonferroni。')
"""
        ),
        new_markdown_cell("## 5. 交付摘要"),
        new_code_cell(
            """out = Path('outputs/13_ab_testing')
out.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame([{
    'p_A': res['p1'], 'p_B': res['p2'], 'lift_pp': res['lift_pp'],
    'z': res['z'], 'pvalue': res['pvalue'],
    'ci_low': res['ci95'][0], 'ci_high': res['ci95'][1],
    'n_per_arm_for_1pp': n_need,
}])
summary.to_csv(out / 'ab_summary.csv', index=False)
print('saved', out / 'ab_summary.csv')
print(summary.T)
"""
        ),
        new_markdown_cell(
            """## 附录：深度阅读与作业
- Kohavi et al. *Trustworthy Online Controlled Experiments*
- 作业：把主指标改成 `revenue`（连续），做均值差 t 检验 + bootstrap CI；讨论零膨胀。
- API 地图：`ready` / `DataAnalyzer`；待建 `ExperimentDesign`。
"""
        ),
    ]
    _write(slug, cells)


def e14() -> None:
    slug = "14_hypothesis_testing"
    cells = [
        new_markdown_cell(
            """# E14 假设检验：t / χ² / 单因素 ANOVA

> 覆盖矩阵：统计推断 · `docs/ANALYSIS_COVERAGE.md`

## 你将学到
1. 正态 / 方差齐性检查再选检验。
2. 独立样本 t、Welch、Mann–Whitney 对照。
3. 列联表 χ² 与期望频数警告。
4. 单因素 ANOVA + 事后两两比较意识（多重比较）。
5. **wanglaoshi 缺口**：无 `StatsToolkit` 统一封装与中文适用性报告。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成多组数据 + EDA"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer
from scipy import stats

rng = np.random.default_rng(RANDOM_STATE)
# 三渠道转化金额（略偏）
channels = []
for name, mu in [('organic', 40), ('paid', 48), ('email', 42)]:
    x = rng.normal(mu, 12, 120)
    channels.append(pd.DataFrame({'channel': name, 'revenue': x, 'device': rng.choice(['ios', 'android'], 120)}))
df = pd.concat(channels, ignore_index=True)
# 类别：是否高价值
df['high_value'] = (df['revenue'] > 50).astype(int)
print(df.groupby('channel')['revenue'].describe()[['mean', 'std', 'count']])
print(DataAnalyzer(df).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. 两独立样本 t（Welch）与效应量"),
        new_code_cell(
            """a = df.loc[df.channel == 'organic', 'revenue']
b = df.loc[df.channel == 'paid', 'revenue']
# Shapiro（大样本仅作提示）
print('shapiro organic p≈', stats.shapiro(a.sample(min(50, len(a)), random_state=RANDOM_STATE)).pvalue)
tt = stats.ttest_ind(a, b, equal_var=False)
# Cohen's d（pooled）
pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
d = (b.mean() - a.mean()) / pooled
print('Welch t=', tt.statistic, 'p=', tt.pvalue, "Cohen's d=", d)
mw = stats.mannwhitneyu(a, b, alternative='two-sided')
print('Mann-Whitney U p=', mw.pvalue)
"""
        ),
        new_markdown_cell("## 3. χ² 独立性（渠道 × 高价值）"),
        new_code_cell(
            """tab = pd.crosstab(df['channel'], df['high_value'])
print(tab)
chi2, p, dof, exp = stats.chi2_contingency(tab)
print('chi2', chi2, 'p', p, 'min expected', exp.min())
if exp.min() < 5:
    print('警告：期望频数过低，考虑合并类别或 Fisher')
"""
        ),
        new_markdown_cell("## 4. 单因素 ANOVA"),
        new_code_cell(
            """groups = [g['revenue'].to_numpy() for _, g in df.groupby('channel')]
f_stat, p_anova = stats.f_oneway(*groups)
print('ANOVA F=', f_stat, 'p=', p_anova)
# 简易事后：三对 Welch（未校正——课内强调需校正）
from itertools import combinations
for x, y in combinations(df['channel'].unique(), 2):
    gx = df.loc[df.channel == x, 'revenue']
    gy = df.loc[df.channel == y, 'revenue']
    p = stats.ttest_ind(gx, gy, equal_var=False).pvalue
    print(f'{x} vs {y}: raw p={p:.4f}（展示用，生产请 FDR/Tukey）')
"""
        ),
        new_markdown_cell("## 5. 交付"),
        new_code_cell(
            """from pathlib import Path
out = Path('outputs/14_hypothesis_testing')
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    'welch_p': tt.pvalue, 'cohens_d': d, 'anova_p': p_anova, 'chi2_p': p
}]).to_csv(out / 'stats_summary.csv', index=False)
print('saved', out)
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：对 `device` 做分层检验；讨论辛普森悖论。
- 包优化：`StatsToolkit.recommend_test(df, y, group)` → 适用性检查 + 效应量 + 中文解读。
"""
        ),
    ]
    _write(slug, cells)


def e15() -> None:
    slug = "15_timeseries_forecast"
    cells = [
        new_markdown_cell(
            """# E15 时间序列：分解、平稳性与基线预测

> 覆盖矩阵：时序（相对 E08 更偏「预测/分解」）

## 你将学到
1. 经典季节分解（趋势 / 季节 / 残差）。
2. ADF 平稳性提示；差分的直觉。
3. 朴素 / 季节朴素 / 线性趋势基线；**禁止随机打乱评估**。
4. 与 `extract_datetime_features`、`detect_time_leakage` 衔接。
5. **缺口**：无 `TimeSeriesTools.decompose / forecast_baseline`。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成日序列（趋势+周季节+噪声）"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(RANDOM_STATE)
n = 365 * 2
dates = pd.date_range('2022-01-01', periods=n, freq='D')
t = np.arange(n)
trend = 100 + 0.05 * t
season = 12 * np.sin(2 * np.pi * t / 7)
noise = rng.normal(0, 5, n)
y = trend + season + noise
df = pd.DataFrame({'ds': dates, 'y': y})
print(df.head(), df.shape)
"""
        ),
        new_markdown_cell("## 2. 季节分解（statsmodels）"),
        new_code_cell(
            """from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

s = df.set_index('ds')['y']
decomp = seasonal_decompose(s, model='additive', period=7)
fig = decomp.plot()
fig.set_size_inches(8, 6)
plt.tight_layout()
out = Path('outputs/15_timeseries_forecast'); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'decompose.png', dpi=100)
plt.close()
print('saved decompose.png')
"""
        ),
        new_markdown_cell("## 3. ADF 与差分"),
        new_code_cell(
            """from statsmodels.tsa.stattools import adfuller

def adf_report(series, name):
    r = adfuller(series.dropna(), autolag='AIC')
    print(name, 'adf=', round(r[0], 3), 'p=', round(r[1], 4))

adf_report(s, 'level')
adf_report(s.diff(), 'diff1')
"""
        ),
        new_markdown_cell("## 4. wanglaoshi 时间特征 + 泄漏意识"),
        new_code_cell(
            """from wanglaoshi import features as FE
from wanglaoshi import compete as CT

fe_df = df.rename(columns={'ds': 'datetime'}).copy()
eng = FE.AdvancedFeatureEngineer()
fe_df = eng.extract_datetime_features(fe_df, 'datetime')
print([c for c in fe_df.columns if c not in ('datetime', 'y')][:12])
# 故意制造未来泄漏列做对照
fe_df['y_lead1'] = fe_df['y'].shift(-1)
leak = CT.LeakageDetector()
# 接口：列名启发式
try:
    warnings = leak.detect_time_leakage(fe_df, date_col='datetime', target_col='y')
    print('leakage warnings:', warnings)
except Exception as e:
    print('detect_time_leakage note:', type(e).__name__, e)
"""
        ),
        new_markdown_cell("## 5. 时间切分基线预测"),
        new_code_cell(
            """from sklearn.metrics import mean_absolute_error

split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
# 季节朴素：上周同星期
pred_season = test['y'].copy()
hist = train['y'].to_numpy()
for i, idx in enumerate(test.index):
    pred_season.iloc[i] = hist[-(7 - (i % 7))] if len(hist) >= 7 else hist[-1]
    # 更稳：用训练末尾对齐星期
# 简化：用 shift(7) 在全序列上再切片
df['naive7'] = df['y'].shift(7)
pred = df.iloc[split:]['naive7']
mae = mean_absolute_error(test['y'], pred)
print('seasonal naive MAE', round(mae, 3))
# 线性趋势（仅用时间索引）
from sklearn.linear_model import LinearRegression
X_tr = np.arange(split).reshape(-1, 1)
X_te = np.arange(split, len(df)).reshape(-1, 1)
lr = LinearRegression().fit(X_tr, train['y'])
mae_lr = mean_absolute_error(test['y'], lr.predict(X_te))
print('linear trend MAE', round(mae_lr, 3))
pd.DataFrame({'mae_naive7': [mae], 'mae_linear': [mae_lr]}).to_csv(out / 'forecast_metrics.csv', index=False)
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：换 `period=30` 看月季节是否过度；对比 E08 Bike 小时级。
- 包优化：`TimeSeriesTools` 一键分解图 + ADF 表 + 安全滞后工厂。
"""
        ),
    ]
    _write(slug, cells)


def e16() -> None:
    slug = "16_survival_churn"
    cells = [
        new_markdown_cell(
            """# E16 生存 / 流失：Kaplan–Meier 与删失

> 覆盖矩阵：生存分析

## 你将学到
1. **删失**：观测结束时事件未发生，不能当「没流失」直接分类。
2. Kaplan–Meier 曲线与两组 log-rank 思想。
3. 与普通二分类流失模型的边界。
4. **缺口**：无 `SurvivalAnalyzer`；可选 extras `[survival]`。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成订阅流失数据"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
n = 600
plan = rng.choice(['basic', 'pro'], n, p=[0.6, 0.4])
# pro 流失更慢
scale = np.where(plan == 'pro', 18.0, 10.0)
tenure = rng.exponential(scale)
censor_time = 24.0
event = (tenure <= censor_time).astype(int)
time = np.minimum(tenure, censor_time)
df = pd.DataFrame({'plan': plan, 'time': time, 'event': event})
print(df.groupby('plan')[['time', 'event']].mean())
print(DataAnalyzer(df).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. 手写 KM（无额外依赖）"),
        new_code_cell(
            """def kaplan_meier(times, events):
    order = np.argsort(times)
    times, events = times[order], events[order]
    uniq = np.unique(times)
    surv, t_axis = [1.0], [0.0]
    at_risk = len(times)
    s = 1.0
    for t in uniq:
        died = events[times == t].sum()
        if at_risk > 0 and died > 0:
            s *= (1 - died / at_risk)
        surv.append(s)
        t_axis.append(t)
        at_risk -= np.sum(times == t)
    return np.array(t_axis), np.array(surv)

import matplotlib.pyplot as plt
from pathlib import Path
out = Path('outputs/16_survival_churn'); out.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(7, 4))
for name, g in df.groupby('plan'):
    t_axis, surv = kaplan_meier(g['time'].to_numpy(), g['event'].to_numpy())
    plt.step(t_axis, surv, where='post', label=name)
plt.xlabel('time'); plt.ylabel('S(t)'); plt.legend(); plt.title('KM by plan')
plt.tight_layout(); plt.savefig(out / 'km.png', dpi=100); plt.close()
print('saved km.png')
"""
        ),
        new_markdown_cell("## 3. 简易 log-rank（两组）"),
        new_code_cell(
            """# 使用 lifelines 若可用，否则跳过并打印提示
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    a = df[df.plan == 'basic']
    b = df[df.plan == 'pro']
    lr = logrank_test(a['time'], b['time'], event_observed_A=a['event'], event_observed_B=b['event'])
    print('log-rank p=', lr.p_value)
except Exception as e:
    print('lifelines 可选未装，跳过 log-rank；手写 KM 已足够跑通 evidence。', type(e).__name__)
"""
        ),
        new_markdown_cell("## 4. 错误做法对照：忽略删失直接分类"),
        new_code_cell(
            """# 错误：把 event 当普通标签、忽略 time —— 偏倚风险
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = LabelEncoder().fit_transform(df['plan']).reshape(-1, 1)
y = df['event']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
clf = LogisticRegression().fit(Xtr, ytr)
print('naive accuracy', clf.score(Xte, yte), '（教学：指标看起来「能跑」但问题形式错了）')
pd.DataFrame({'n': [len(df)], 'event_rate': [df.event.mean()]}).to_csv(out / 'survival_summary.csv', index=False)
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：加入协变量做 Cox（需 lifelines）；对比仅用 `event` 分类的偏差叙述。
- 包优化：`SurvivalAnalyzer.km_plot` / `logrank`，依赖放 extras。
"""
        ),
    ]
    _write(slug, cells)


def e17() -> None:
    slug = "17_text_eda"
    cells = [
        new_markdown_cell(
            """# E17 文本 EDA：浅特征、词袋与分类基线

> 覆盖矩阵：文本分析 · API：`AdvancedFeatureEngineer.extract_text_features`

## 你将学到
1. 长度 / 词数等 **basic** 浅特征的业务含义。
2. `method='tfidf'` 进表格模型前的稀疏性与泄漏（矢量器只 fit 训练折）。
3. 中文分词仍是缺口（空格分词不适用于中文）。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成评论数据"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
pos = [
    'great product fast shipping love it',
    'excellent quality highly recommend',
    'amazing value and wonderful support',
]
neg = [
    'terrible quality broke quickly',
    'awful experience never again',
    'poor packaging damaged item refund',
]
rows = []
for _ in range(80):
    rows.append({'text': rng.choice(pos) + ' ' + rng.choice(['', 'thanks', '!!!']), 'label': 1})
for _ in range(80):
    rows.append({'text': rng.choice(neg) + ' ' + rng.choice(['', 'disappointed', '...']), 'label': 0})
df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(df.label.value_counts())
print(DataAnalyzer(df).get_basic_info().get('dtypes'))
"""
        ),
        new_markdown_cell("## 2. extract_text_features(basic)"),
        new_code_cell(
            """from wanglaoshi import features as FE
eng = FE.AdvancedFeatureEngineer()
feat = eng.extract_text_features(df, 'text', method='basic')
print([c for c in feat.columns if c.startswith('text_')])
print(feat[['text_length', 'text_word_count', 'label']].groupby('label').mean())
"""
        ),
        new_markdown_cell("## 3. 折内 TF-IDF + 基线"),
        new_code_cell(
            """from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pathlib import Path

Xtr, Xte, ytr, yte = train_test_split(df['text'], df['label'], test_size=0.3, random_state=RANDOM_STATE, stratify=df['label'])
vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
Xtr_v = vec.fit_transform(Xtr)  # 只 fit 训练折
Xte_v = vec.transform(Xte)
clf = LogisticRegression(max_iter=500).fit(Xtr_v, ytr)
print(classification_report(yte, clf.predict(Xte_v), digits=3))
out = Path('outputs/17_text_eda'); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'n_features': [Xtr_v.shape[1]], 'acc': [clf.score(Xte_v, yte)]}).to_csv(out / 'text_metrics.csv', index=False)
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：对中文语料说明为何 `str.split` 失败；调研 jieba 作为可选依赖。
- 包优化：文档强调矢量器折内 fit；可选 `method='tfidf'` 返回稀疏矩阵接口统一。
"""
        ),
    ]
    _write(slug, cells)


def e18() -> None:
    slug = "18_anomaly_detection"
    cells = [
        new_markdown_cell(
            """# E18 异常检测：IsolationForest（无标签场景）

> 覆盖矩阵：无监督异常 · 区别于 E05 有标签欺诈

## 你将学到
1. 无标签时用隔离森林 / LOF；评估需注入已知异常或业务抽检。
2. 分数阈值与告警预算。
3. **缺口**：无 `AnomalyDetector` 封装。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成正常 + 注入异常"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
n_norm, n_out = 500, 25
X_norm = rng.normal(0, 1, size=(n_norm, 4))
X_out = rng.normal(4, 1.2, size=(n_out, 4))
X = np.vstack([X_norm, X_out])
y_true = np.array([0] * n_norm + [1] * n_out)  # 仅用于事后评估
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
df['is_injected_outlier'] = y_true
print(DataAnalyzer(df.drop(columns=['is_injected_outlier'])).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. IsolationForest"),
        new_code_cell(
            """from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

clf = IsolationForest(contamination=n_out / len(df), random_state=RANDOM_STATE)
pred = clf.fit_predict(df[[f'f{i}' for i in range(4)]])  # -1 异常
y_hat = (pred == -1).astype(int)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_hat, average='binary', zero_division=0)
print(dict(precision=p, recall=r, f1=f1, flagged=int(y_hat.sum())))
out = Path('outputs/18_anomaly_detection'); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'precision': [p], 'recall': [r], 'f1': [f1]}).to_csv(out / 'anomaly_metrics.csv', index=False)
"""
        ),
        new_markdown_cell("## 3. 与有监督欺诈课的边界"),
        new_code_cell(
            """print('E05 Fraud：有标签 → PR-AUC / class_weight')
print('E18：无标签 → 污染率假设 + 人工复核；分数需业务校准')
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：改 contamination，画 precision-recall 随阈值变化。
- 包优化：`AnomalyDetector.fit_score(df)` → 分数、阈值建议、规则分融合。
"""
        ),
    ]
    _write(slug, cells)


def e19() -> None:
    slug = "19_association_rules"
    cells = [
        new_markdown_cell(
            """# E19 关联规则：购物篮支持度 / 置信度 / 提升度

> 覆盖矩阵：关联规则（纯 Python，无 mlxtend 强依赖）

## 你将学到
1. 事务数据编码；支持度、置信度、提升度。
2. 高提升度 ≠ 可干预因果。
3. **缺口**：包内无关联规则 API（宜放可选 extras）。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成购物篮"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from itertools import combinations
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
items = ['bread', 'milk', 'eggs', 'butter', 'diaper', 'beer']
baskets = []
for i in range(200):
    b = set(rng.choice(items[:4], size=rng.integers(1, 4), replace=False))
    # 注入模式：diaper → beer
    if rng.random() < 0.25:
        b |= {'diaper', 'beer'}
    baskets.append(frozenset(b))
# 长表便于 Analyzer
rows = [{'txn': i, 'item': it} for i, b in enumerate(baskets) for it in b]
df = pd.DataFrame(rows)
print(df.head(), 'n_txn', df.txn.nunique())
print(DataAnalyzer(df).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. 频繁 1-/2-项集与规则"),
        new_code_cell(
            """from collections import Counter
from pathlib import Path

n_txn = len(baskets)
min_support = 0.08
c1 = Counter()
for b in baskets:
    for it in b:
        c1[frozenset([it])] += 1
freq1 = {k: v / n_txn for k, v in c1.items() if v / n_txn >= min_support}
c2 = Counter()
for b in baskets:
    for a, c in combinations(sorted(b), 2):
        c2[frozenset([a, c])] += 1
freq2 = {k: v / n_txn for k, v in c2.items() if v / n_txn >= min_support}
rules = []
for pair, sup in freq2.items():
    a, b = tuple(pair)
    for ant, con in ((a, b), (b, a)):
        ant_f = frozenset([ant])
        if ant_f not in freq1:
            continue
        conf = sup / freq1[ant_f]
        lift = conf / freq1[frozenset([con])]
        rules.append(dict(antecedent=ant, consequent=con, support=sup, confidence=conf, lift=lift))
rules_df = pd.DataFrame(rules).sort_values('lift', ascending=False)
print(rules_df.head(10))
out = Path('outputs/19_association_rules'); out.mkdir(parents=True, exist_ok=True)
rules_df.to_csv(out / 'rules.csv', index=False)
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：解释 diaper→beer 高 lift 仍可能是混杂；设计门店实验。
- 包优化：可选 `AssociationRules.mine(transactions, min_support=...)`。
"""
        ),
    ]
    _write(slug, cells)


def e20() -> None:
    slug = "20_causal_did"
    cells = [
        new_markdown_cell(
            """# E20 因果入门：合成双重差分（DiD）

> 覆盖矩阵：因果推断入门（观测数据陷阱）

## 你将学到
1. 相关 ≠ 因果；平行趋势假设。
2. 用合成面板做 2×2 DiD。
3. **缺口**：无因果模块；本课仅教学，不宣称包已具备因果能力。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成面板：处理组后期受干预"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
rows = []
true_effect = 5.0
for unit in range(80):
    treated = int(unit >= 40)
    for period in (0, 1):  # 0 pre, 1 post
        base = 20 + 3 * period + 2 * treated
        tau = true_effect if (treated == 1 and period == 1) else 0.0
        y = base + tau + rng.normal(0, 1.5)
        rows.append(dict(unit=unit, treated=treated, post=period, y=y))
df = pd.DataFrame(rows)
print(df.groupby(['treated', 'post'])['y'].mean().unstack())
print(DataAnalyzer(df).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. DiD 回归"),
        new_code_cell(
            """import statsmodels.formula.api as smf
from pathlib import Path

df['did'] = df['treated'] * df['post']
model = smf.ols('y ~ treated + post + did', data=df).fit()
print(model.summary().tables[1])
est = float(model.params['did'])
print('DiD estimate', round(est, 3), 'true', true_effect)
out = Path('outputs/20_causal_did'); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'did_est': [est], 'true_effect': [true_effect]}).to_csv(out / 'did_summary.csv', index=False)
"""
        ),
        new_markdown_cell("## 3. 错误对照：只看处理后截面相关"),
        new_code_cell(
            """post = df[df.post == 1]
print('post-only mean diff', post.loc[post.treated==1, 'y'].mean() - post.loc[post.treated==0, 'y'].mean())
print('（含组间固有差异，高估/混淆真效应）')
"""
        ),
        new_markdown_cell(
            """## 附录
- 作业：破坏平行趋势（给处理组不同 pre 斜率）再估 DiD，观察偏差。
- 包优化：长期可考虑文档化「因果不是基线工坊核心」；或独立 extras 教程，避免 API 承诺过度。
"""
        ),
    ]
    _write(slug, cells)


def main() -> None:
    e13()
    e14()
    e15()
    e16()
    e17()
    e18()
    e19()
    e20()
    print("wave2 done: E13–E20")


if __name__ == "__main__":
    main()
