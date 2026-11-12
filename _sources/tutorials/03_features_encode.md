```{note}
本页从仓库 `docs/tutorials/03_features_encode.md` 同步生成；请以仓库文件为编辑源。
```

# 03 · features 编码（合成高基数表）

## 目标

10 分钟内：用 `AdvancedFeatureEngineer` 做频率编码与目标编码，并理解「拟合范围 = 传入 DataFrame」。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `AdvancedFeatureEngineer` | `wanglaoshi.features` | 目标编码 / 频率编码等 |
| `ready` | `runtime` | 环境就绪 |

## 数据集

合成高基数列（模拟 Adult 职业类字段）。离线、无网络依赖。  
真实 Adult 流程见 `examples/advanced/10_adult_census.ipynb`。

## 完整示例

```python
import numpy as np
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.features import AdvancedFeatureEngineer

ready(seed=42)
rng = np.random.default_rng(42)
n = 400
cats = np.array([f"job_{i}" for i in range(40)])
job = rng.choice(cats, size=n)
# 目标与类别弱相关，便于看编码效果
base = pd.Series(job).str.extract(r"(\d+)")[0].astype(int) / 40.0
y = (rng.random(n) < (0.2 + 0.6 * base)).astype(int)

train = pd.DataFrame({"job": job[:300], "y": y[:300]})
valid = pd.DataFrame({"job": job[300:]})

eng = AdvancedFeatureEngineer()
# 只在 train 上拟合目标编码（课堂演示）；竞赛请用 OOF / 折内 Pipeline
train_te = eng.target_encode(train, "job", "y", smoothing=5.0)
freq_map = train["job"].value_counts(normalize=True)
valid = valid.copy()
valid["job_freq"] = valid["job"].map(freq_map).fillna(0.0)
# 把 train 上的 TE 映射到 valid
te_map = train_te.groupby("job")["job_target_encoded"].first()
valid["job_te"] = valid["job"].map(te_map)
print(train_te[["job", "job_target_encoded"]].head())
print(valid.head())
```

配套脚本：[`examples/tutorials/03_features_encode.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/03_features_encode.py)

## 常见坑

- `target_encode` 的拟合范围 = **传入的行集合**；禁止 `concat(train, test)` 后编码
- 验证/测试只能 **map** 训练集（或当前折）得到的映射
- 高基数 + 小样本务必加大 `smoothing`，并优先 OOF

## 下一步

- 统计检验 → [04_analytics_stats.md](04_analytics_stats.md)
- Adult 深课 → `examples/advanced/10_adult_census.ipynb`
- 精讲：折内 fit → `book/lessons/feature_fit_discipline.md`
