# 主线：Analyzer / EDA

## 你要用的入口

```python
from wanglaoshi import DataAnalyzer
import pandas as pd

df = pd.read_csv("your.csv")
analyzer = DataAnalyzer(df)
analyzer.basic_statistics()
analyzer.missing_value_analysis()
analyzer.generate_report("report.html")
```

## 学习要点

- 先分清数值列 / 类别列，再谈相关与 PCA  
- 报告是给人读的交付物，不是「跑过就算」  
- 小样本上的正态性检验可能失败或警告（样本量要够）

## 推荐案例

- {doc}`../cases/01_iris`  
- {doc}`../cases/02_titanic`  
- {doc}`../cases/04_breast_cancer`  

## 参考

- API 盘点：{doc}`../reference/api`  
