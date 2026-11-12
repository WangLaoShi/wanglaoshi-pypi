```{note}
本页从仓库 `docs/tutorials/07_leakage_and_cv.md` 同步生成；请以仓库文件为编辑源。
```

# 07 · 泄漏检测 + 分层 CV（Titanic）

## 目标

15 分钟内：用 `LeakageDetector` 抓出目标泄漏列，再用 `CrossValidator.stratified_cv` 做可信验证。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `LeakageDetector` | `compete` | 目标 / 时间 / 分布泄漏提示 |
| `CrossValidator` | `compete` | 分层 / 分组 / 时序 CV |
| `ready` | `runtime` | 环境就绪 |

## 数据集

seaborn Titanic（可离线）。故意保留泄漏列 `alive` 做对比。

## 完整示例

```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from wanglaoshi.runtime import ready
from wanglaoshi.compete import LeakageDetector, CrossValidator

ready(seed=42)

raw = sns.load_dataset("titanic").copy()
y = raw["survived"]

# 故意把 alive 留在特征里（≈ 标签）
leaky = pd.get_dummies(
    raw.drop(columns=["survived", "deck", "embark_town", "class", "who", "adult_male"], errors="ignore"),
    drop_first=True,
).fillna(0)
det = LeakageDetector()
hits = det.detect_target_leakage(leaky, y, threshold=0.85, method="correlation")
print("leakage hits:", hits)

# 清洗后再 CV
clean = pd.get_dummies(
    raw.drop(
        columns=["survived", "alive", "deck", "embark_town", "class", "who", "adult_male"],
        errors="ignore",
    ),
    drop_first=True,
).fillna(0)
cv = CrossValidator()
model = LogisticRegression(max_iter=500, random_state=42)
print(cv.stratified_cv(model, clean, y, n_splits=5))
```

配套脚本：[`examples/tutorials/07_leakage_and_cv.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/07_leakage_and_cv.py)

## 常见坑

- 高相关只是「提示」，还要结合业务语义（如 `alive`）
- CV 分数是参照，不是拿验证折调到过拟合的许可证
- 分组/时间场景请改用 `group_cv` / `time_series_cv`

## 下一步

- 时序安全特征 → [08_timeseries_tools.md](08_timeseries_tools.md)
- 深课 → `examples/advanced/02_titanic.ipynb`、`book/lessons/leakage.md`
