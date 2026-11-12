# 精讲：三类数据泄露

泄露 = **验证/测试阶段用到了「当时不该有」的信息**。分数会虚高，上线或提交后立刻塌掉。竞赛与业务交付里，这是最高优先级纪律。

## 1. 三类泄露（先背清）

| 类型 | 典型症状 | 例子 |
| --- | --- | --- |
| **目标泄露** | 特征与标签几乎完美相关；单一特征 AUC≈1 | `df["label_copy"]`、事后才有的「是否违约已结清」 |
| **时间泄露** | 用了「未来」才知道的字段训练过去 | 用次日价格预测今日；用整段均值填缺失再划分 |
| **分布泄露** | 训练/验证分布人为一致，真实部署不一致 | 先全表标准化再 `train_test_split`；测试集混进训练集 ID |

`LeakageDetector` 帮你**提示可疑点**，不能代替你设计隔离流程。

## 2. 最小可运行：目标泄露侦测

```python
import pandas as pd
import numpy as np
from wanglaoshi.runtime import ready
from wanglaoshi.CompetitionTools import LeakageDetector

ready()

rng = np.random.default_rng(0)
n = 400
df = pd.DataFrame({
    "feat_ok": rng.normal(size=n),
    "feat_noise": rng.normal(size=n),
    "y": rng.integers(0, 2, size=n),
})
# 故意制造目标泄露：几乎是标签本身
df["feat_leak"] = df["y"] + rng.normal(0, 0.01, size=n)

det = LeakageDetector()
# 接口：把候选特征与目标交给检测器（具体参数名以本版 API 为准）
report = det.detect_target_leakage(
    df, target_col="y", feature_cols=["feat_ok", "feat_noise", "feat_leak"]
)
print(report)
```

**你应观察到：** `feat_leak` 被标成高风险；`feat_ok` / `feat_noise` 相对正常。若作业里删掉 `feat_leak` 再跑基线，验证分会「变差」——那才是更真实的分数。

## 3. 时间泄露：对照实验

错误流程（**不要学**）：

```python
# BAD：用全表未来信息填缺失，再划分
df["amount_filled"] = df["amount"].fillna(df["amount"].mean())  # 均值含「未来」行
train = df.iloc[:800]
valid = df.iloc[800:]
```

正确流程：

```python
# GOOD：先按时间切，再用「训练段」统计量填验证段
train = df.iloc[:800].copy()
valid = df.iloc[800:].copy()
fill_value = train["amount"].mean()
train["amount_filled"] = train["amount"].fillna(fill_value)
valid["amount_filled"] = valid["amount"].fillna(fill_value)
```

时间相关检测可参考：

```python
# 若有明确时间列与「不应出现在训练」的列
det.detect_time_leakage(df, time_col="event_time", suspect_cols=["next_day_price"])
```

## 4. 分布泄露：标准化位置

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# GOOD：Pipeline 保证 scaler 只在训练折 fit
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=500)),
])
clf.fit(X_train, y_train)
print(clf.score(X_valid, y_valid))
```

```{warning}
先 `StandardScaler().fit(全表)` 再划分 = 分布泄露。验证集参与了均值/方差估计。
```

## 5. 课堂讨论题（建议写进笔记）

1. 特征「距上次购买天数」在预测「明天是否购买」时，会不会泄露？取决于观测截止时刻怎么定义。  
2. 目标编码（mean target encoding）为何必须在 **CV 折内** 计算？见 {doc}`feature_fit_discipline`。  
3. `detect_distribution_leakage` 报可疑后，你下一步应改代码还是改数据说明？

## 6. 连到案例

- Titanic：缺失填充与称谓特征 — {doc}`../cases/02_titanic`  
- Adult：目标编码 — {doc}`../cases/10_adult_census`  
- Bike：滞后特征 — {doc}`../cases/08_bike_sharing`  
- 练习墙：{doc}`common_mistakes`
