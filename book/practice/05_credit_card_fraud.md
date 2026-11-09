# 练习：Credit Card Fraud

对应课：{doc}`../cases/05_credit_card_fraud`

## 学习目标

- 在极端不均衡下选择 **PR-AUC / recall** 等指标  
- 避免被 99% accuracy 欺骗  

## 自测

1. 全部预测「非欺诈」的 accuracy 大概多少？这说明什么？  
2. ROC-AUC 在极不均衡时可能过于乐观，为何还要看 PR？  
3. 重采样若在划分前做，会造成什么问题？  

## 动手

- 报告 precision/recall/PR-AUC，而不是只报 accuracy  
- 尝试 class_weight 或阈值调整，观察召回变化  

## 提示方向

先定业务代价（漏检 vs 误报），再选指标与阈值。
