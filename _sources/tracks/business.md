# 主线：业务流水线

目标：把销售/运营 CSV 跑成 **可追溯交付包**（备份、底稿、图、报告、索引），并在证据不足时诚实阻断或标记 `pending`。

## 1. 为什么不是「再写一个大 Notebook」

| Notebook 临时分析 | `BusinessPipeline` |
| --- | --- |
| 结果在内存里 | 每阶段落盘 + 状态 |
| 数字难以回放 | 指标底稿可对账 |
| 缺列仍可能出「漂亮结论」 | `blocked` → 不可交付 |

Skill 来源与 API 映射：{doc}`../reference/skill-mapping`。

## 2. 阶段心智模型（D01–D12）

| 阶段 | 做什么 | 学生要检查的证据 |
| --- | --- | --- |
| D01 接收 | 归档 + sha256 台账 | 原始文件只读备份 |
| D02–D03 | 列名规整、可配置清洗 | 新数据集 + 影响日志 |
| D12 EDA | 画像 + 统计适用性 | 方法是否适用的说明 |
| D04–D06 | 衍生、四方向计划、指标 | 指标字典与占比校验 |
| D07–D08 | 六视角分析、图归档 | 标题/单位/周期/来源 |
| D09–D10 | 结论/建议、报告与索引 | 证据编号；收益无依据→pending |
| D11 | 总控 | 上游阻断后不假装交付成功 |

深水区（多表、缺期、同比等）尚未等同「勾选完成」——见 {doc}`../reference/project-status`。

## 3. 最小可运行用法

```python
from wanglaoshi.runtime import ready
from wanglaoshi.business import BusinessPipeline

ready()
pipe = BusinessPipeline(
    "archive/",
    required_columns=["order_id", "amount"],
    date_col="order_date",
    value_col="amount",
    dim_cols=["channel"],
    period="2024-Q1",
)
record = pipe.run("orders.csv")
print(record.summary())
```
配置样例：`wanglaoshi/business/config_sales_example.yaml`（关键字段缺失不得静默瞎猜）。

## 4. 课堂任务设计（建议 60–90 分钟）

1. 跑通 {doc}`../cases/business_pipeline`  
2. 打开归档目录，画出「文件从哪来到哪去」  
3. 删掉金额列或制造重复订单键，观察是否阻断  
4. 精讲：{doc}`../lessons/business_traceability`  
5. 练习：{doc}`../practice/business_pipeline`  

## 5. 结论与建议的写作纪律

- **现状 / 亮点 / 问题** 必须带证据编号或表格出处  
- 贡献分解 ≠ 业务因果（Pareto 只能当结构描述）  
- 没有对照或基线时，量化 uplift 写 `pending`，不要编百分比  

## 6. 本地其它入口

- 短 demo：`examples/business_intake_demo.ipynb`  
- 测试：`pytest -m business`（若已标记）  

## 7. 课后作业（可评分）

1. 自备一张 ≥30 行的脱敏业务表，跑通流水线，提交归档 zip（可去隐私列）。  
2. 写一页「口径附录」：金额是否含退款、时间用下单还是支付。  
3. 列出 2 个当前 MVP **做不到**、需要人工补的深水区能力，对照项目进度文档。  
