# 练习：Breast Cancer

对应课：{doc}`../cases/04_breast_cancer`

## 学习目标

- 理解缩放 / PCA 在流水线中的位置  
- 在医疗场景下权衡 **召回 vs 精确率**  

## 自测

1. PCA 拟合必须在训练集上——若先 PCA 全表再划分，错在哪？  
2. 若漏诊代价远高于误报，你更该盯哪个指标？阈值应往哪边调？  
3. 标准化放在 PCA 前还是后？为什么？  

## 动手

- 画 PR 曲线或比较不同阈值的 recall  
- 用 Pipeline 把 `StandardScaler → PCA → 分类器` 绑在一起做 CV  

## 提示方向

先缩放再 PCA；评估指标要贴业务代价，不只有 accuracy。
