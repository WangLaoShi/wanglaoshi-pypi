# 练习：Adult Census（绑定 wanglaoshi API）

对应课：{doc}`../cases/10_adult_census` · 精讲：{doc}`../lessons/feature_fit_discipline`

## 本课强制 API 清单

| API | 纪律 |
| --- | --- |
| `FE.AdvancedFeatureEngineer.target_encode` | **演示 API**；竞赛必须折外 / 折内编码 |
| `FE.frequency_encode` / `one_hot_encode` | 高基数 vs 低基数策略 |
| `CT.CrossValidator.stratified_cv` | 分层评估 |
| `CT.LeakageDetector` | 编码后复查异常高相关 |

## 学习目标

- 说清：为何「全表 `groupby(col)[y].mean()` 再划分」会泄露  
- 能手写 OOF（out-of-fold）目标编码骨架（精讲课有模板）  
- 区分：包提供的 `target_encode`（便利）≠ 已替你做好 CV 隔离  

## 自测

1. 测试集目标编码时，能否用「全部训练集」均值？CV 训练折呢？  
2. `target_encode` 的 `smoothing` 是做什么的？  
3. 若练习里只用全表 `target_encode` 就交作业，扣分点是什么？  

## 动手加难

1. 实现精讲课中的 `oof_target_encode`，与全表编码各训一个模型，比较验证分差异。  
2. 用 `stratified_cv` 报告均值±标准差，而不是单次 holdout。  
3. 在 Markdown 写一句：**本课对应仓库 API 说明书哪一节**。  

## 参考思路

Adult 的价值在「高基数类别 + 目标编码纪律」，不是把准确率背下来。
