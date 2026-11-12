# 精讲：交叉验证与划分策略

验证分只有在「划分方式 ≈ 真实部署方式」时才有意义。选错划分，模型再花哨也是自欺。

## 1. 决策表（先查这张）

| 数据形态 | 推荐划分 | 忌用 | 课内例子 |
| --- | --- | --- | --- |
| 独立同分布、类别均衡或可分层 | `StratifiedKFold` / `stratify=y` | 打乱后类别全空 | Iris、Breast Cancer |
| 有时间顺序、要预测未来 | 时间切分 / `TimeSeriesSplit` | 随机打乱 | Bike Sharing |
| 同一实体多行（用户/门店/患者） | `GroupKFold`（按 ID） | 同 ID 进训练又进验证 | California 区块、业务客户 |
| 极端不均衡 | 分层 + 合适指标（PR-AUC） | 只看 Accuracy | Credit Card Fraud |
| 高维小样本 | 外层 CV + 内层选参，预算写死 | 无尽搜参刷分 | Digits / 小表作业 |

## 2. 分层划分：为什么 Iris 必须 stratify

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

X, y = load_iris(return_X_y=True, as_frame=True)

def class_counts(y_arr):
    return pd.Series(y_arr).value_counts().sort_index().to_dict()

# 对比：有无 stratify
_, _, y_a, _ = train_test_split(X, y, test_size=0.3, random_state=0)  # 无分层
_, _, y_b, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

print("no stratify valid:", class_counts(y_a))
print("stratify valid:   ", class_counts(y_b))
```

**解读：** 无分层时，某一类可能在验证集偏少，Accuracy 方差变大，调参结论不稳。

## 3. 时间划分：不能 shuffle

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

n = 100
X = np.arange(n).reshape(-1, 1)
y = np.sin(np.linspace(0, 6, n))

tscv = TimeSeriesSplit(n_splits=5)
for i, (tr, va) in enumerate(tscv.split(X)):
    print(i, "train_end", tr[-1], "valid", va[0], "→", va[-1])
    assert tr.max() < va.min()  # 训练时刻严格早于验证
```

滞后特征规则：**只用预测时刻之前已发生的信息**。Bike 案例（E08）专门练这条。

## 4. 分组划分：同一 ID 不能两侧都出现

```python
from sklearn.model_selection import GroupKFold
import numpy as np

X = np.random.randn(12, 2)
y = np.array([0, 1] * 6)
groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # 每个 ID 两行

gkf = GroupKFold(n_splits=3)
for tr, va in gkf.split(X, y, groups):
    assert set(groups[tr]).isdisjoint(set(groups[va]))
    print("train groups", sorted(set(groups[tr])), "valid", sorted(set(groups[va])))
```

房价/地理数据里，若按「街区」部署，随机行划分会把同街区泄漏到验证集 → 分数虚高。

## 5. 用仓库工具做 CV（示意）

```python
from wanglaoshi.CompetitionTools import CrossValidator
# 按任务选择 strategy：stratified / time / group（以当前 API 文档为准）
cv = CrossValidator(n_splits=5, random_state=42)
# scores = cv.evaluate(model, X, y, strategy="stratified")
```

把 **策略名称、种子、折数** 写进实验记录（`ExperimentTracker`），否则别人无法复现你的「0.87」。

## 6. 自测

1. 预测「下周门店销量」，历史表按行随机 8:2 划分，错在哪？  
2. 欺诈检测 Accuracy=99.8%，为何仍可能毫无用处？  
3. 外层 CV 已经在选模型，内层又用同一验证集调参，会发生什么？

## 7. 下一步

- 对照错误墙：{doc}`common_mistakes`  
- 案例：{doc}`../cases/06_california_housing`、{doc}`../cases/08_bike_sharing`、{doc}`../cases/05_credit_card_fraud`
