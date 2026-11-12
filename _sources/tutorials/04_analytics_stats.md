```{note}
本页从仓库 `docs/tutorials/04_analytics_stats.md` 同步生成；请以仓库文件为编辑源。
```

# 04 · analytics 统计工具（合成 A/B）

## 目标

10 分钟内：用 `StatsToolkit` 做独立样本 t 检验，并用 `ExperimentDesign` 看样本量直觉。

## 本节公开接口

| 符号 | 接口模块 | 作用 |
| --- | --- | --- |
| `StatsToolkit` | `wanglaoshi.analytics` | t / χ² / ANOVA 等 |
| `ExperimentDesign` | `wanglaoshi.analytics` | 比例检验与样本量 |
| `ready` | `runtime` | 环境就绪 |

## 数据集

合成两组连续指标（模拟对照组 / 实验组转化相关度量）。无需下载。

## 完整示例

```python
import numpy as np

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import StatsToolkit, ExperimentDesign

ready(seed=42)
rng = np.random.default_rng(42)
control = rng.normal(loc=10.0, scale=2.0, size=80)
treat = rng.normal(loc=10.8, scale=2.0, size=80)

tt = StatsToolkit.ttest_ind(control, treat)
print("t-test:", tt)

# 两比例：对照 8%，最小可检出绝对提升 4pp（→12%）
ss = ExperimentDesign.sample_size_two_proportion(p1=0.08, mde_abs=0.04, alpha=0.05, power=0.8)
print("sample size per arm:", ss)

prop = ExperimentDesign.two_proportion_test(x1=20, n1=200, x2=32, n2=200)
print("two-proportion:", prop)
```

配套脚本：[`examples/tutorials/04_analytics_stats.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/tutorials/04_analytics_stats.py)

## 常见坑

- p 值小 ≠ 业务重要：同时看效应量与置信区间
- 多重比较会抬高假阳性，需校正或预注册主指标
- A/B 深课（功效、贝叶斯门槛）见 E13

## 下一步

- 业务流水线 → [05_business_pipeline.md](05_business_pipeline.md)
- 假设检验课 → `examples/advanced/14_hypothesis_testing.ipynb`
- A/B 课 → `examples/advanced/13_ab_testing.ipynb`
