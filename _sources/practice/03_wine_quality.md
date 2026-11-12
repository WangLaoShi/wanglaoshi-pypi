# 练习：Wine Quality

对应课：{doc}`../cases/03_wine_quality`

## 学习目标

- 分清回归 / 有序分类 / 普通多分类的建模选择  
- 处理类别不均衡时的评估  

## 自测

1. 把 quality 当连续回归与当分类，评估指标应如何不同？  
2. 少数高分酒样很少时，accuracy 可能有什么误导？  
3. 注意：UCI Wine Quality ≠ `sklearn.load_wine`，差在哪？  

## 动手

- 同时报回归 MAE 与「四舍五入后」的分类报告  
- 统计各 quality 档样本量，决定是否合并档位  

## 提示方向

先对齐任务定义（分数预测 vs 档位预测），再选指标与模型。
