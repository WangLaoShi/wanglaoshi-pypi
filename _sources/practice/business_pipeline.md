# 练习：业务流水线

对应课：{doc}`../cases/business_pipeline`

## 学习目标

- 读懂 `StageResult` 四种状态：`passed` / `warning` / `blocked` / `skipped`  
- 解释上游 blocked 为何不能正式交付  
- 会配置字段角色与脱敏字段  

## 自测

1. 缺必需列时，D01 更可能是 `warning` 还是 `blocked`？缺文件呢？  
2. `stop_on_block=False` 时下游会发生什么？交付物还能标 deliverable=True 吗？  
3. 二八分解结果能直接写成「因为头部渠道所以因果增长」吗？手册里怎么说？  
4. 建议里的 `expected_uplift='pending'` 表示什么？  

## 动手

- 故意传入不存在的 CSV，观察 skipped 链  
- 打开 `report.md` 与 `delivery_index.json`，核对 sha256 与 deliverable  
- 增加 `redact_fields`，确认敏感字段变成 `***REDACTED***`  

## 提示方向

业务链的「正确」首先是可追溯与可阻断，其次才是指标好看。
