# 01 · ready + EDA（Iris / Tips）

## 目标

10 分钟内：用 `runtime.ready` 准备环境，用 `analyze.DataAnalyzer` 对公开表做基础画像。

## 用到的门面 API

| 符号 | 门面 | 作用 |
| --- | --- | --- |
| `ready` | `wanglaoshi.runtime` | 关警告、打环境、设种子 |
| `DataAnalyzer` | `wanglaoshi.analyze` | 描述统计、缺失、报告 |

## 数据集

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| Iris | `sklearn.datasets.load_iris` | 经典分类玩具集，可离线 |
| Tips | `seaborn.load_dataset("tips")` | 小费表，含类别与数值 |

## 完整示例

```python
from sklearn.datasets import load_iris
import seaborn as sns

from wanglaoshi.runtime import ready
from wanglaoshi.analyze import DataAnalyzer

ready(seed=42)

iris = load_iris(as_frame=True)
df_iris = iris.frame
az = DataAnalyzer(df_iris)
print(az.get_basic_info())
print(az.basic_statistics().head())
print(az.missing_value_analysis().head())
# az.generate_report("iris_eda.html")  # 需要时打开

tips = sns.load_dataset("tips")
az2 = DataAnalyzer(tips)
print(az2.basic_statistics().head())
```

配套脚本：[`examples/tutorials/01_ready_and_eda.py`](../../examples/tutorials/01_ready_and_eda.py)

## 常见坑

- 不要把 ID / 标签误解成「因果自变量」；相关 ≠ 因果
- `ready()` 已关警告，勿再叠 `filterwarnings` / `JE.no_warning`
- 生成 HTML 报告前确认工作目录可写

## 下一步

- 竞赛基线与提交 → [02_compete_baseline.md](02_compete_baseline.md)
- 完整 Iris 课 → `examples/advanced/01_iris.ipynb`
- API 签名 → [`docs/API_REFERENCE.md`](../API_REFERENCE.md) 中 `ready` / `DataAnalyzer`
