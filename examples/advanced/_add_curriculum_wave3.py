#!/usr/bin/env python3
"""Wave3：补齐六大缺口。

- 新建 21 三表对账、22 MovieLens NDCG
- 并入 15 同比缺月、04 概率校准、10 公平性审计、13 贝叶斯 A/B

运行：
  PYTHONPATH=. python examples/advanced/_add_curriculum_wave3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


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
    nbformat.write(_nb(cells), path)
    print(f"wrote {path.relative_to(ROOT)} ({len(cells)} cells)")


def _append(slug: str, extra: list, after_title: str | None = None) -> None:
    """在附录之前插入；若无附录则追加到末尾。"""
    path = OUT / f"{slug}.ipynb"
    nb = nbformat.read(path, as_version=4)
    insert_at = len(nb.cells)
    for i, c in enumerate(nb.cells):
        src = c.source if isinstance(c.source, str) else "".join(c.source)
        if src.lstrip().startswith("## 附录") or src.lstrip().startswith("## 深度阅读"):
            insert_at = i
            break
    # 避免重复插入
    marker = extra[0].source if isinstance(extra[0].source, str) else "".join(extra[0].source)
    for c in nb.cells:
        src = c.source if isinstance(c.source, str) else "".join(c.source)
        if marker[:40] in src:
            print(f"skip append {slug}: already has section")
            return
    for j, cell in enumerate(extra):
        nb.cells.insert(insert_at + j, cell)
    # 更新标题提示覆盖点
    if after_title and nb.cells and nb.cells[0].cell_type == "markdown":
        src0 = nb.cells[0].source if isinstance(nb.cells[0].source, str) else "".join(nb.cells[0].source)
        if after_title not in src0:
            nb.cells[0].source = src0.rstrip() + "\n\n" + after_title + "\n"
    nbformat.write(nb, path)
    print(f"appended -> {slug}.ipynb (+{len(extra)} cells)")


# ---------------------------------------------------------------------------
# 新建 21 / 22
# ---------------------------------------------------------------------------

def e21() -> None:
    slug = "21_order_refund_reconciliation"
    cells = [
        new_markdown_cell(
            """# E21 订单、明细与退款三表对账：防止连接后金额翻倍

> 公开范例对照：[Looker Symmetric Aggregates](https://docs.cloud.google.com/looker/docs/best-practices/understanding-symmetric-aggregates)  
> 覆盖矩阵：多表关联膨胀 · `docs/ANALYSIS_COVERAGE.md`

## 你将学到
1. 订单↔明细 **一对多** 连接后，对订单金额 `sum` 会翻倍。
2. 正确做法：在订单粒度聚合，或对「会重复的度量」用 distinct/子查询后再汇总。
3. 加入退款表后，**销售额与退款额分别与源表对平**。
4. **wanglaoshi 缺口**：业务 D02 多表膨胀检查尚未产品化（深水区 S2）。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 合成三表：订单 / 明细 / 退款"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from pathlib import Path
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4],
    'order_amount': [100.0, 200.0, 150.0, 80.0],  # 订单头金额（业务源）
    'customer_id': ['c1', 'c2', 'c1', 'c3'],
})
# 一对多明细（行金额之和可与订单头一致或略有差异——课内以订单头为准对账）
items = pd.DataFrame({
    'order_id': [1, 1, 2, 2, 2, 3, 4, 4],
    'sku': ['A', 'B', 'A', 'C', 'D', 'B', 'A', 'E'],
    'line_amount': [40, 60, 50, 70, 80, 150, 30, 50],
})
refunds = pd.DataFrame({
    'refund_id': [10, 11, 12],
    'order_id': [2, 2, 4],
    'refund_amount': [20.0, 10.0, 15.0],
})
print('orders sum', orders.order_amount.sum(), 'items sum', items.line_amount.sum(), 'refunds sum', refunds.refund_amount.sum())
print(DataAnalyzer(orders).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. ❌ 错误：订单金额经明细一对多连接后求和"),
        new_code_cell(
            """bad = orders.merge(items, on='order_id', how='left')
print('行数膨胀', len(orders), '->', len(bad))
bad_sales = bad['order_amount'].sum()
true_sales = orders['order_amount'].sum()
print('错误汇总销售额', bad_sales, '真值', true_sales, '膨胀倍数', round(bad_sales / true_sales, 2))
assert bad_sales > true_sales, '应演示翻倍/膨胀'
"""
        ),
        new_markdown_cell("## 3. ✅ 正确：先在订单粒度取不重复金额，再与退款对账"),
        new_code_cell(
            """# 销售额：只在订单表汇总（或 merge 后对 order_id 去重再 sum）
sales_ok = orders['order_amount'].sum()
# 若必须从宽表取：按 order_id 保留一行订单金额
dedup = bad.drop_duplicates('order_id')
sales_from_dedup = dedup['order_amount'].sum()
assert abs(sales_ok - sales_from_dedup) < 1e-9

# 退款：在退款表汇总；若 join 订单后求和退款额，同样要防一对多
ref_bad = orders.merge(refunds, on='order_id', how='left')
# 订单1、3 无退款 → NaN；订单2 两行退款
ref_wrong = ref_bad['refund_amount'].fillna(0).sum()  # 这里碰巧未膨胀订单金额，但若 sum(order_amount) 仍会错
ref_ok = refunds['refund_amount'].sum()
print('退款真值', ref_ok, '| 经订单join后对 refund 求和', ref_wrong)

# 净额（教学口径）：销售额 - 退款
net = sales_ok - ref_ok
print(dict(sales=sales_ok, refunds=ref_ok, net=net))
"""
        ),
        new_markdown_cell("## 4. 对账断言：与源表勾稽"),
        new_code_cell(
            """out = Path('outputs/21_order_refund_reconciliation')
out.mkdir(parents=True, exist_ok=True)
recon = pd.DataFrame([{
    'metric': 'sales',
    'source': true_sales,
    'wrong_join_sum': bad_sales,
    'corrected': sales_ok,
    'balanced': abs(sales_ok - true_sales) < 1e-9,
}, {
    'metric': 'refunds',
    'source': ref_ok,
    'wrong_join_sum': ref_wrong,
    'corrected': ref_ok,
    'balanced': abs(ref_ok - refunds.refund_amount.sum()) < 1e-9,
}])
print(recon)
assert recon.loc[recon.metric == 'sales', 'balanced'].iloc[0]
assert recon.loc[recon.metric == 'refunds', 'balanced'].iloc[0]
recon.to_csv(out / 'reconciliation.csv', index=False)
print('saved', out / 'reconciliation.csv')
"""
        ),
        new_markdown_cell(
            """## 附录
- 对照 Looker：对称聚合解决的是「探索层自动防重复计数」；在 pandas ETL 里必须显式选粒度。
- 作业：把 `order_amount` 改成只存在于明细汇总，写「头表缺失时如何对账」流程。
- 包优化：`BusinessPipeline` D02 输出「膨胀倍数 + 对账差额」日志。
"""
        ),
    ]
    _write(slug, cells)


def e22() -> None:
    slug = "22_movielens_ndcg"
    cells = [
        new_markdown_cell(
            """# E22 MovieLens 电影推荐：用 NDCG@10 评估 Top‑K 排序

> 公开范例对照：[Recommenders Quick Start](https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/README.md)  
> 覆盖矩阵：推荐 / 排序指标（NDCG）

## 你将学到
1. 时间切分（按用户历史）避免泄漏。
2. **热门榜基线** vs 简易用户-物品相似度 / 共现。
3. 同一候选集上算 **NDCG@10**（及 Recall@K 辅助）。
4. 数据：优先本地/网络 MovieLens 100K 子集；失败则用合成交互（保证 evidence）。
"""
        ),
        new_code_cell(
            """QUICK = True
RANDOM_STATE = 42
from wanglaoshi.runtime import ready
ready(seed=RANDOM_STATE)
"""
        ),
        new_markdown_cell("## 1. 加载 MovieLens 或 DEMO 交互"),
        new_code_cell(
            """import numpy as np
import pandas as pd
from pathlib import Path
from wanglaoshi.analyze import DataAnalyzer

rng = np.random.default_rng(RANDOM_STATE)
DEMO = True
ratings = None
# 尝试本地
for cand in [Path('../../data/ml-100k/u.data'), Path('data/ml-100k/u.data')]:
    if cand.exists():
        ratings = pd.read_csv(cand, sep='\\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        DEMO = False
        print('local', cand, ratings.shape)
        break
if ratings is None:
    try:
        from urllib.request import urlopen
        import io
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
        raw = urlopen(url, timeout=8).read()
        ratings = pd.read_csv(io.BytesIO(raw), sep='\\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
        DEMO = False
        print('downloaded ml-100k', ratings.shape)
    except Exception as e:
        print('fallback DEMO', type(e).__name__, e)
        n_users, n_items, n = 80, 120, 2500
        ratings = pd.DataFrame({
            'user_id': rng.integers(1, n_users + 1, n),
            'item_id': rng.integers(1, n_items + 1, n),
            'rating': rng.integers(1, 6, n),
            'timestamp': np.sort(rng.integers(1_000_000_000, 1_100_000_000, n)),
        }).drop_duplicates(['user_id', 'item_id'])
        DEMO = True
# QUICK 子样
if QUICK and len(ratings) > 8000:
    keep_u = ratings['user_id'].drop_duplicates().sample(min(200, ratings.user_id.nunique()), random_state=RANDOM_STATE)
    ratings = ratings[ratings.user_id.isin(keep_u)].copy()
print('DEMO', DEMO, ratings.shape, DataAnalyzer(ratings).get_basic_info().get('shape'))
"""
        ),
        new_markdown_cell("## 2. 按用户时间切分：最后一次交互进测试"),
        new_code_cell(
            """ratings = ratings.sort_values(['user_id', 'timestamp'])
# 每用户至少 2 条
vc = ratings.groupby('user_id').size()
ratings = ratings[ratings.user_id.isin(vc[vc >= 2].index)].copy()

def leave_one_last(g):
    g = g.sort_values('timestamp')
    return g.iloc[:-1], g.iloc[-1:]

train_parts, test_parts = [], []
for _, g in ratings.groupby('user_id'):
    tr, te = leave_one_last(g)
    train_parts.append(tr)
    test_parts.append(te)
train = pd.concat(train_parts)
test = pd.concat(test_parts)
print(train.shape, test.shape, 'users', test.user_id.nunique())
"""
        ),
        new_markdown_cell("## 3. 热门榜基线 + 共现推荐"),
        new_code_cell(
            """# 热门：训练集全局频次
pop = train.groupby('item_id').size().sort_values(ascending=False)
# 用户已看集合
seen = train.groupby('user_id')['item_id'].apply(set).to_dict()

def recommend_popular(uid, k=10):
    s = seen.get(uid, set())
    rec = [i for i in pop.index if i not in s][:k]
    return rec

# 简易物品共现：对测试用户，取其训练物品的共现邻居投票
from collections import defaultdict, Counter
# 构建稀疏共现（QUICK：限制用户数）
user_items = train.groupby('user_id')['item_id'].apply(list).to_dict()
co = defaultdict(Counter)
for items in list(user_items.values())[: min(500, len(user_items))]:
    uniq = list(set(items))
    for a in uniq:
        for b in uniq:
            if a != b:
                co[a][b] += 1

def recommend_cooc(uid, k=10):
    s = seen.get(uid, set())
    scores = Counter()
    for it in s:
        scores.update(co.get(it, {}))
    ranked = [i for i, _ in scores.most_common(k + len(s)) if i not in s]
    if len(ranked) < k:
        # 回退热门补齐
        ranked += [i for i in recommend_popular(uid, k=k + 20) if i not in ranked and i not in s]
    return ranked[:k]
"""
        ),
        new_markdown_cell("## 4. NDCG@10（同一候选长度）"),
        new_code_cell(
            """def dcg_at_k(rels, k=10):
    rels = np.asarray(rels[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float((rels * discounts).sum())

def ndcg_at_k(recommended, relevant_set, k=10):
    rels = [1.0 if i in relevant_set else 0.0 for i in recommended[:k]]
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k([1.0] * min(k, len(relevant_set)), k)
    return 0.0 if ideal == 0 else dcg / ideal

def eval_method(fn, k=10):
    scores = []
    for row in test.itertuples(index=False):
        uid, iid = row.user_id, row.item_id
        rec = fn(uid, k=k)
        scores.append(ndcg_at_k(rec, {iid}, k=k))
    return float(np.mean(scores))

ndcg_pop = eval_method(recommend_popular, 10)
ndcg_co = eval_method(recommend_cooc, 10)
print(dict(ndcg10_popular=ndcg_pop, ndcg10_cooc=ndcg_co, n_test_users=int(test.user_id.nunique()), DEMO=DEMO))
out = Path('outputs/22_movielens_ndcg')
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    'ndcg10_popular': ndcg_pop,
    'ndcg10_cooc': ndcg_co,
    'demo': DEMO,
}]).to_csv(out / 'ndcg_summary.csv', index=False)
"""
        ),
        new_markdown_cell(
            """## 附录
- Recommenders SAR 等可作进阶；本课强调 **指标与切分契约**，不绑重依赖。
- 作业：改成 NDCG@5；对负采样候选集固定后再比模型。
- 包优化：可选 `RankingMetrics.ndcg_at_k`。
"""
        ),
    ]
    _write(slug, cells)


# ---------------------------------------------------------------------------
# 并入既有课
# ---------------------------------------------------------------------------

def patch_15_yoy() -> None:
    extra = [
        new_markdown_cell(
            """## 6. 月度销售同比：缺月、零基数与未完月

> 对照：[DAX SAMEPERIODLASTYEAR](https://learn.microsoft.com/en-us/dax/sameperiodlastyear-function-dax)

教学点：对齐「去年同月」前必须先 **补齐日历骨架**；区分缺月（无记录）与真实 0；零基数导致 YoY 无穷大；未完月不可与去年整月比。
"""
        ),
        new_code_cell(
            """# 小销售表：故意缺 2023-02，2023-03 为真实 0，2024-11 为未完月
sales_m = pd.DataFrame({
    'month': pd.to_datetime([
        '2023-01-01', '2023-03-01', '2023-04-01',
        '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-11-01',
    ]),
    'sales': [100.0, 0.0, 120.0, 110.0, 130.0, 0.0, 140.0, 50.0],
})
# 完整月骨架
idx = pd.date_range('2023-01-01', '2024-11-01', freq='MS')
cal = pd.DataFrame({'month': idx}).merge(sales_m, on='month', how='left')
cal['is_missing'] = cal['sales'].isna()  # 缺月
cal['sales_filled'] = cal['sales']  # 缺月保持 NaN，真实 0 保留 0
cal['sales_for_sum'] = cal['sales'].fillna(0)  # 仅用于某些展示——同比时不要盲目 fillna(0) 当「真零」

def yoy_row(row, df):
    prev = row['month'] - pd.DateOffset(years=1)
    hit = df.loc[df['month'] == prev]
    if hit.empty:
        return dict(yoy=np.nan, reason='no_prior_period')
    prev_sales = hit['sales'].iloc[0]
    cur = row['sales']
    if pd.isna(cur):
        return dict(yoy=np.nan, reason='current_missing_month')
    if pd.isna(prev_sales):
        return dict(yoy=np.nan, reason='prior_missing_month')
    if prev_sales == 0:
        return dict(yoy=np.nan, reason='zero_base')
    return dict(yoy=(cur - prev_sales) / prev_sales, reason='ok')

# 未完月：假设「今天」是 2024-11-15，则 2024-11 不可比去年 11 整月
as_of = pd.Timestamp('2024-11-15')
cal['incomplete'] = (cal['month'].dt.to_period('M') == as_of.to_period('M')) & (as_of.day < cal['month'].dt.days_in_month)

rows = []
for _, r in cal.iterrows():
    y = yoy_row(r, cal)
    if r['incomplete']:
        y['reason'] = 'incomplete_month'
        y['yoy'] = np.nan
    rows.append({**r.to_dict(), **y})
yoy_df = pd.DataFrame(rows)
print(yoy_df[['month', 'sales', 'is_missing', 'incomplete', 'yoy', 'reason']].to_string(index=False))
out = Path('outputs/15_timeseries_forecast'); out.mkdir(parents=True, exist_ok=True)
yoy_df.to_csv(out / 'yoy_edge_cases.csv', index=False)
print('saved yoy_edge_cases.csv')
"""
        ),
    ]
    _append(
        "15_timeseries_forecast",
        extra,
        after_title="> **本课增补：** 月度同比边缘情况（缺月 / 零基数 / 未完月）。",
    )


def patch_04_calibration() -> None:
    extra = [
        new_markdown_cell(
            """## 11. 概率校准：可靠性图、Brier 与 ECE

> 对照：[sklearn Calibration of Classifiers](https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html)

在**同一测试集**上比较校准前后；ECE 使用固定分箱口径（`n_bins=10`）。
"""
        ),
        new_code_cell(
            """from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
pipe.fit(Xtr, ytr)
prob_raw = pipe.predict_proba(Xte)[:, 1]

# 预拟合校准（用训练折内 CV，避免碰测试集）
calib = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
calib.fit(Xtr, ytr)
prob_cal = calib.predict_proba(Xte)[:, 1]

rows = []
for name, p in [('raw', prob_raw), ('isotonic', prob_cal)]:
    rows.append({
        'model': name,
        'brier': brier_score_loss(yte, p),
        'ece_10': expected_calibration_error(yte, p, n_bins=10),
    })
cal_df = pd.DataFrame(rows)
print(cal_df)

# 可靠性图数据（固定 10 bin）
frac_raw, mean_raw = calibration_curve(yte, prob_raw, n_bins=10, strategy='uniform')
frac_cal, mean_cal = calibration_curve(yte, prob_cal, n_bins=10, strategy='uniform')
out = Path('outputs/04_breast_cancer'); out.mkdir(parents=True, exist_ok=True)
cal_df.to_csv(out / 'calibration_metrics.csv', index=False)
pd.DataFrame({'mean_pred_raw': mean_raw, 'frac_pos_raw': frac_raw}).to_csv(out / 'reliability_raw.csv', index=False)
print('saved calibration_metrics.csv')
"""
        ),
    ]
    _append(
        "04_breast_cancer",
        extra,
        after_title="> **本课增补：** 可靠性图 / Brier / 固定 10 箱 ECE（校准前后同测试集）。",
    )


def patch_10_fairness() -> None:
    extra = [
        new_markdown_cell(
            """## 9.3 公平性审计：性别 × 种族交叉群体

> 对照：[Fairlearn GridSearch Census](https://fairlearn.org/main/auto_examples/plot_grid_search_census.html)

不强制安装 `fairlearn`：输出交叉群体的样本量、阳性率、TPR/FPR（误判结构）。
"""
        ),
        new_code_cell(
            """from sklearn.metrics import confusion_matrix

sex_col = next((c for c in ['sex', 'Sex', 'gender', 'Gender'] if c in Xte.columns), None)
race_col = next((c for c in ['race', 'Race'] if c in Xte.columns), None)
yhat = np.asarray(pred)
yte_a = np.asarray(yte)
assert len(yhat) == len(yte_a)

audit = Xte.copy()
audit['_y'] = yte_a
audit['_yh'] = yhat
if sex_col is None:
    audit['_sex'] = 'unknown'
    sex_col = '_sex'
if race_col is None:
    audit['_race'] = 'unknown'
    race_col = '_race'

rows = []
for key, gdf in audit.groupby([sex_col, race_col]):
    yt = gdf['_y'].to_numpy()
    yh = gdf['_yh'].to_numpy()
    tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    label = key if not isinstance(key, tuple) else '|'.join(map(str, key))
    rows.append({
        'group': label,
        'n': int(len(gdf)),
        'pos_rate': float(yt.mean()),
        'pred_pos_rate': float(yh.mean()),
        'tpr': tpr,
        'fpr': fpr,
        'fn': int(fn),
        'fp': int(fp),
    })
fair_df = pd.DataFrame(rows).sort_values('n', ascending=False)
print(fair_df.head(20))
out = Path('outputs/10_adult_census'); out.mkdir(parents=True, exist_ok=True)
fair_df.to_csv(out / 'fairness_audit.csv', index=False)
print('saved fairness_audit.csv | groups', len(fair_df))
try:
    import fairlearn  # noqa: F401
    print('fairlearn available: 可选继续 GridSearch（本课不强制）')
except Exception:
    print('fairlearn 未安装：手工分组表已覆盖审计教学点')
"""
        ),
    ]
    _append(
        "10_adult_census",
        extra,
        after_title="> **本课增补：** 性别×种族交叉群体公平性审计（样本量 / TPR / FPR）。",
    )


def patch_13_bayes() -> None:
    extra = [
        new_markdown_cell(
            """## 6. 贝叶斯 A/B：Beta-Binomial 与先验敏感性

> 对照：[PyMC Bayesian A/B Testing](https://www.pymc.io/projects/examples/en/2022.12.0/case_studies/bayesian_ab_testing_introduction.html)

用共轭解析解（不强制 PyMC）：后验为 Beta；估计「B 版转化率超过 A 且提升超过业务门槛 δ」的概率。
"""
        ),
        new_code_cell(
            """from scipy.stats import beta as beta_dist

# 沿用本节转化计数
x_a, n_a_ = int(g.get_group('A').sum()), int(g.get_group('A').count())
x_b, n_b_ = int(g.get_group('B').sum()), int(g.get_group('B').count())
delta = 0.005  # 业务门槛：绝对提升至少 0.5pp

def posterior_prob_b_beats_a(xa, na, xb, nb, delta=0.0, alpha0=1.0, beta0=1.0, n_mc=50000, rng=None):
    rng = np.random.default_rng(RANDOM_STATE) if rng is None else rng
    pa = rng.beta(alpha0 + xa, beta0 + na - xa, size=n_mc)
    pb = rng.beta(alpha0 + xb, beta0 + nb - xb, size=n_mc)
    return float(np.mean(pb > pa + delta))

priors = {
    'uniform(1,1)': (1.0, 1.0),
    'weak_center_10pct(2,18)': (2.0, 18.0),
    'strong_center_10pct(20,180)': (20.0, 180.0),
}
bayes_rows = []
for name, (a0, b0) in priors.items():
    p = posterior_prob_b_beats_a(x_a, n_a_, x_b, n_b_, delta=delta, alpha0=a0, beta0=b0)
    bayes_rows.append({'prior': name, 'P(pB>pA+delta)': p, 'delta': delta})
bayes_df = pd.DataFrame(bayes_rows)
print(bayes_df)
# 后验均值
print('posterior mean A/B (uniform)',
      (1 + x_a) / (2 + n_a_), (1 + x_b) / (2 + n_b_))
out = Path('outputs/13_ab_testing'); out.mkdir(parents=True, exist_ok=True)
bayes_df.to_csv(out / 'bayesian_ab.csv', index=False)
print('saved bayesian_ab.csv')
try:
    import pymc  # noqa: F401
    print('pymc available: 可用官方 Notebook 做 MCMC；本课解析解已覆盖教学点')
except Exception:
    print('pymc 未安装：使用 Beta-Binomial 共轭解析 + Monte Carlo')
"""
        ),
    ]
    _append(
        "13_ab_testing",
        extra,
        after_title="> **本课增补：** 贝叶斯 A/B（Beta-Binomial）与先验敏感性 / 业务门槛概率。",
    )


def main() -> None:
    e21()
    e22()
    patch_15_yoy()
    patch_04_calibration()
    patch_10_fairness()
    patch_13_bayes()
    print("wave3 done")


if __name__ == "__main__":
    main()
