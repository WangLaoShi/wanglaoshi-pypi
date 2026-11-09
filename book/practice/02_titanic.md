# 练习：Titanic（绑定 wanglaoshi API）

对应课：{doc}`../cases/02_titanic` · 精讲：{doc}`../lessons/leakage` · {doc}`../lessons/feature_fit_discipline`

## 本课强制 API 清单

| API | 用途 |
| --- | --- |
| `DataAnalyzer.missing_value_analysis` | 先看缺失再建模 |
| `FE.AdvancedFeatureEngineer` | 称谓/舱位等类别处理（折内） |
| `CT.LeakageDetector.detect_target_leakage` | 抓「几乎等于标签」的列 |
| `CT.QuickBaseline` | 同划分基线 |

## 学习目标

- 缺失填充的统计量只能来自**训练折**  
- 能解释：先全表 `fillna(median)` 再划分 = 分布泄露  
- 会用检测器提示可疑特征，但不把检测器当流程本身  

## 自测

1. 用全表中位数填 `Age` 再 `train_test_split`，泄露类型是哪一种？  
2. 若某特征与 `Survived` 相关 0.99，你下一步删特征还是改划分？  
3. `detect_target_leakage` 的返回值是纯字符串列表吗？（对照说明书）  

## 动手加难

1. 写两段 Pipeline：**错误**（全表 fit）vs **正确**（折内 fit），对比验证 AUC/Accuracy。  
2. 对训练特征跑一次 `detect_target_leakage(..., threshold=0.8)`，解读输出里的 `feature` 字段。  
3. 在练习文档里贴出你调用的**完整方法名**（禁止只写「用了包里的工具」）。  

## 参考思路

Titanic 教学重点不是刷分，而是 **隔离**：任何 `fit` 都不能看见验证折的统计。
