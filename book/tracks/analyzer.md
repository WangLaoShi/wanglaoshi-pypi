# 主线：Analyzer / EDA

目标：在 **45–90 分钟**内，把一张表做成「别人能读懂的数据画像 + HTML 报告」，并知道哪些统计结论不能乱写。

## 1. 模块地图

| API | 典型用途 |
| --- | --- |
| `DataAnalyzer` | 描述统计、缺失、相关、分组对比、报告 |
| `Visualizer`（若暴露） | 单图输出；业务线里由 D08 归档带元数据的图 |
| `ready()` | 统一环境横幅与警告策略 |

```python
from wanglaoshi import ready, DataAnalyzer
import pandas as pd

ready()
df = pd.read_csv("your.csv")  # 或 sklearn 玩具数据
analyzer = DataAnalyzer(df)
analyzer.basic_statistics()
analyzer.missing_value_analysis()
# 按版本可能还有：correlation_analysis / outlier 检测等
analyzer.generate_report("report.html")
```

## 2. 标准作业流程（建议照做）

1. **字段角色**：谁是 ID、谁是时间、谁是数值、谁是类别、谁是标签（若有）  
2. **质量**：缺失率、重复键、常数列、明显错类型（数字存成字符串）  
3. **单变量**：分布、分位数；类别 top-k  
4. **关系**：相关 / 交叉表；**不写因果**  
5. **交付**：HTML 或业务归档；注明数据周期与来源  

## 3. 课堂完整小例子（Iris）

```python
from sklearn.datasets import load_iris
from wanglaoshi import ready, DataAnalyzer

ready()
iris = load_iris(as_frame=True)
df = iris.frame  # 含 target 列
az = DataAnalyzer(df)
print(az.basic_statistics())
print(az.missing_value_analysis())
az.generate_report("iris_eda.html")
print("打开 iris_eda.html，检查：样本量、各类计数、是否有缺失。")
```

**验收标准（给学生打分用）：**

- 报告里能回答：多少行、多少列、目标几类、有无缺失  
- 口头能解释：为何相关矩阵不能证明「花瓣长度导致种类」  

## 4. 常见坑（Analyzer 专场）

| 坑 | 后果 | 处理 |
| --- | --- | --- |
| 把 ID 当数值做相关 | 无意义高相关 | 分析前剔除 ID |
| 小样本做正态性结论过猛 | 误导建模假设 | 看 D12 适用性；n 太小降级为探索 |
| 报告生成了但没人读 | 作业空转 | 强制写 5 行「数据说明」摘要 |
| 中文列名乱码 | 交付失败 | 统一 UTF-8；读 CSV 指定编码 |

## 5. 和另外两条主线的衔接

- 竞赛：EDA 之后进入划分与基线 → {doc}`competition`、{doc}`../lessons/cv_and_splits`  
- 业务：EDA 是 D12，前后仍有接收/清洗/指标 → {doc}`business`  

## 6. 推荐案例与精讲

| 资源 | 用途 |
| --- | --- |
| {doc}`../cases/01_iris` | 最小闭环 |
| {doc}`../cases/02_titanic` | 缺失与类别 |
| {doc}`../cases/04_breast_cancer` | 医疗表质量与缩放 |
| {doc}`../lessons/common_mistakes` | 交作业前过墙 |
| {doc}`../reference/api` | API 盘点 |

## 7. 课后作业（可评分）

1. 任选一 CSV（≥100 行），产出 `report.html` + 一页「数据说明」（来源、周期、字段角色、已知问题）。  
2. 列出 3 个**不能**从当前 EDA 推出的结论，并说明缺什么证据。  
