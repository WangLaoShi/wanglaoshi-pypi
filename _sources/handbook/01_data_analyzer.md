```{note}
本页从仓库 `docs/handbook/01_data_analyzer.md` 同步生成；请以仓库文件为编辑源。
```

# 使用手册 01 · `DataAnalyzer`

## 这个类是干什么的

`wanglaoshi.analyze.DataAnalyzer` 把一张表做成「可交付的数据画像」：描述统计、缺失、异常、相关、PCA、类别/时间摘要，并可生成 HTML 报告。

```python
from wanglaoshi.analyze import DataAnalyzer
az = DataAnalyzer(df)
```

## 方法地图

| 阶段 | 方法 | 你得到什么 |
| --- | --- | --- |
| 概览 | `get_basic_info` / `get_memory_usage` | 形状、类型、内存 |
| 质量 | `missing_value_analysis` / `duplicate_analysis` / `outlier_analysis` | 清洗优先级 |
| 分布 | `basic_statistics` / `normality_test` / `get_categorical_stats` / `get_time_stats` | 单变量画像 |
| 关系 | `correlation_analysis` / `multicollinearity_analysis` / `pca_analysis` | 共线与降维线索 |
| 可视化 | `plot_distribution` / `plot_correlation_heatmap` / `plot_time_series` | base64 图（报告用） |
| 汇总 | `explore_dataframe` / `interpret_results` / `generate_report` | 一站式探索与 HTML |

## 案例：餐厅小费表（tips）+ Iris

**数据：** `seaborn.load_dataset("tips")`（公开内置）、`sklearn` Iris。

```python
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.datasets import load_iris

from wanglaoshi.runtime import ready
from wanglaoshi.analyze import DataAnalyzer

ready(seed=42, quiet=True, show_env=False)

# --- tips：业务感更强的混合表 ---
tips = sns.load_dataset("tips")
az = DataAnalyzer(tips)

print("info", az.get_basic_info())
print("mem MB", round(az.get_memory_usage(), 4))
print(az.basic_statistics().head())
print(az.missing_value_analysis().head())
print(az.duplicate_analysis())
print(az.outlier_analysis().head())
print(az.get_categorical_stats())

# 数值关系
print(az.correlation_analysis().iloc[:5, :5])
print(az.multicollinearity_analysis().head())
print(az.normality_test().head())

# 图（返回 base64，不强制弹窗）
_ = az.plot_distribution("total_bill")
_ = az.plot_correlation_heatmap()

# 一站式 + 解读
explored = az.explore_dataframe(name="tips", show_plots=False)
interp = az.interpret_results()
assert isinstance(explored, dict) and isinstance(interp, dict)

# --- iris：PCA 更稳 ---
iris = load_iris(as_frame=True).frame
az2 = DataAnalyzer(iris)
pca = az2.pca_analysis()
print("pca keys", sorted(pca.keys())[:8])
# az2.generate_report("iris_handbook.html")  # 需要时打开
```

配套脚本：[`examples/handbook/01_data_analyzer.py`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/handbook/01_data_analyzer.py)

## 使用建议

1. 先 `get_basic_info` + `missing_value_analysis`，再决定是否填缺失  
2. ID 列勿进相关 / PCA  
3. `normality_test` 在小样本上只作提示  
4. 相关 ≠ 因果；`interpret_results` 是辅助措辞，不是自动结论

## 相关文档

- 短教程：[tutorials/01](../tutorials/01_ready_and_eda) / [12](../tutorials/12_analyze_helpers)  
- 签名表：[API_REFERENCE §2](../reference/api)  
- 深课：`examples/advanced/01_iris.ipynb`、`02_titanic.ipynb`
