# 精讲：业务数字可追溯

业务交付与竞赛刷分不同：老板要的是 **「这个数从哪来、哪一步改过数据、缺证据能不能交差」**。`BusinessPipeline` 把 Skill 落成阶段，每个阶段返回 `StageResult`。

## 1. 心智模型

```text
原始 CSV ──D01 备份+哈希──► 台账
        ──D02/D03 规整+清洗──► 新表 + 逐规则影响日志
        ──D12 EDA──► 画像 / 方法适用性
        ──D04–D06 衍生+计划+指标──► 指标字典 + 底稿
        ──D07–D08 分析+图──► 带元数据的图与数据
        ──D09–D10 结论+归档──► 报告 + 交付索引
```

任一关键阶段 `blocked` → 运行记录应标记 **不可正式交付**（`deliverable=False`）。

## 2. 跑通最小销售样例

```python
from pathlib import Path
import pandas as pd
from wanglaoshi import ready, BusinessPipeline

ready()

# 构造一份「像业务」的小表（课堂可替换为真实脱敏 CSV）
demo = Path("biz_demo")
demo.mkdir(exist_ok=True)
orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6],
    "order_date": pd.date_range("2024-01-01", periods=6, freq="D"),
    "channel": ["线上", "线下", "线上", "线上", "线下", "线上"],
    "amount": [100.0, 80.0, 120.0, 90.0, 70.0, 110.0],
})
csv_path = demo / "orders.csv"
orders.to_csv(csv_path, index=False)

archive = demo / "archive"
pipe = BusinessPipeline(
    str(archive),
    required_columns=["order_id", "amount"],
    date_col="order_date",
    value_col="amount",
    dim_cols=["channel"],
    period="2024-01",
)
record = pipe.run(str(csv_path))
print(record.summary())
# 逐阶段查看
for stage in getattr(record, "stages", []) or []:
    print(stage)
```

**课堂要求：** 打开归档目录，指出：

1. 原始备份文件与哈希是否存在  
2. 清洗是否写出**新文件**（而不是原地偷偷改）  
3. 报告里的数字能否在指标底稿里找到同一行  

## 3. 读懂状态枚举

| 状态 | 含义 | 学生常见误解 |
| --- | --- | --- |
| `passed` | 本阶段契约满足 | 「等于业务结论一定对」——否，只表示流程通过 |
| `warning` | 可继续，但有风险 | 忽略 warning 直接交差 |
| `blocked` | 下游不得当正式交付 | 手动改报告「弄好看」 |
| `skipped` | 因上游阻断或开关关闭而跳过 | 当成通过 |

收益类建议若无对照实验支撑，应标 **`pending`**，不能编造 uplift%。

## 4. 人为制造阻断（理解「不能交差」）

```python
# 缺必需列 → D01/结构阶段应阻断或警告（依配置）
bad = orders.drop(columns=["amount"])
bad_path = demo / "orders_bad.csv"
bad.to_csv(bad_path, index=False)
rec2 = pipe.run(str(bad_path))
print(rec2.summary())
# 期望：不可交付，而不是「空报告也算成功」
```

## 5. 与竞赛线的分工

| | 竞赛工具链 | 业务流水线 |
| --- | --- | --- |
| 成功标准 | 合法 submission + 可信验证分 | 可追溯数字 + 完整交付包 |
| 典型输出 | CSV 提交 | 报告、底稿、图、索引 |
| 失败形态 | 校验失败 / 泄露虚高 | 缺证据仍标 deliverable |

两者都在「基线工坊」里，但**评价尺子不同**。

## 6. 深水区诚实说明（避免过度承诺）

主链 D01–D12 是 **MVP**：多表关联膨胀、缺期对齐、同比、头腰尾分层等仍在深水区（见 {doc}`../reference/project-status`）。课堂演示用单表干净数据可以；真实脏业务表要预期更多 `warning`/`blocked`。

## 7. 下一步

- 专章 Notebook：{doc}`../cases/business_pipeline`  
- 练习：{doc}`../practice/business_pipeline`  
- Skill 映射：{doc}`../reference/skill-mapping`
