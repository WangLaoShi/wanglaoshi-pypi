# 练习：Mall Customers

对应课：{doc}`../cases/09_mall_customers`

## 学习目标

- 评估聚类 **稳定性**（不止看一次 silhouette）  
- 避免把聚类标签当成「真实类别」硬套分类指标  

## 自测

1. 只报告一次 KMeans silhouette 有何不足？  
2. 特征未缩放时，收入与年龄量纲如何扭曲距离？  
3. 若业务要「可解释人群」，你会如何描述每个簇？  

## 动手

- 多次不同 `random_state` 跑 KMeans，看簇中心是否漂移  
- 标准化前后对比 silhouette  

## 提示方向

聚类是探索工具；稳定性与业务可解释性比单次分数更重要。
