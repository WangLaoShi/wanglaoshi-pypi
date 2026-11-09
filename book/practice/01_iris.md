# 练习：Iris

对应课：{doc}`../cases/01_iris`

## 学习目标

- 解释小样本多分类为何要用 **分层划分**  
- 区分整数标签与字符串标签的 submission 形态  
- 用 `SubmissionGenerator` 完成本地生成与校验  

## 自测

1. 若 `train_test_split(..., stratify=None)`，在 Iris 上最可能出什么问题？  
2. sample 模板目标列叫 `Species`（字符串）时，预测数组应是什么 dtype？校验时统计应走 categorical 还是数值？  
3. 把验证准确率写进提交文件——这违反了哪条竞赛纪律？  

## 动手

- 改 `test_size`，观察各类在验证集中的计数是否仍近似均衡  
- 故意传错 `target_column`，确认会报错而不是静默猜错  

## 提示方向

分层是为了「每一折/每一份都看到所有类」；提交文件只应对测试 ID 负责，不含线下分数。
