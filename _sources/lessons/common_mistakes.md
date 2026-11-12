# 精讲：常见错误墙

把作业/项目里最高频的翻车列在墙上。对照检查比「再看一遍 Notebook」更有效。

## A. 划分与泄露

| # | 错误 | 正确 |
| ---: | --- | --- |
| A1 | 先全表标准化 / PCA / 选特征，再划分 | Pipeline 或折内 fit |
| A2 | 时间序列随机 `shuffle=True` | 时间切分或 `TimeSeriesSplit` |
| A3 | 同一 `user_id` 出现在 train 与 valid | `GroupKFold` |
| A4 | 用测试标签调参 | 测试集只最终评估一次 |
| A5 | 滞后特征用到「当天尚未发生」的量 | 移位，使特征严格过去 |

详解：{doc}`leakage`、{doc}`cv_and_splits`、{doc}`feature_fit_discipline`

## B. 提交与标签

| # | 错误 | 正确 |
| ---: | --- | --- |
| B1 | 不管 sample 列名，自己叫 `label` | 跟模板 |
| B2 | 概率矩阵静默 `argmax` | 显式映射 / 多列契约 |
| B3 | 字符串标签当 float 写文件 | 按 categorical 处理 |
| B4 | 提交里带验证分或模型名 | 只保留 ID+预测 |
| B5 | ID 顺序与 sample 不一致仍上传 | 先 `validate_submission` |

详解：{doc}`submission_contracts`

## C. 指标与叙述

| # | 错误 | 正确 |
| ---: | --- | --- |
| C1 | 不均衡问题只报 Accuracy | 加 PR-AUC / F1 / 召回 |
| C2 | 相关/贡献写成「导致」 | 贡献 ≠ 因果 |
| C3 | 小样本强行谈显著性 | 先看方法适用性（D12） |
| C4 | 业务建议编造「预计提升 20%」 | 无证据 → `pending` |

详解：{doc}`business_traceability`

## D. 环境与复现

| # | 错误 | 正确 |
| ---: | --- | --- |
| D1 | 笔记本里叠三次 `filterwarnings` | 一次 `wanglaoshi.ready()` |
| D2 | 口头说 random_state=42 但没写进代码 | 代码 + 实验记录双写 |
| D3 | 「我本地能跑」无依赖说明 | 对照 {doc}`../reference/dependencies` |
| D4 | 把演示跑通当成测试通过 | 看 `tests/` 与证据文档 |

## 5 分钟自检清单（交作业前）

- [ ] 划分方式与业务/竞赛场景一致，并写进 Markdown 说明  
- [ ] 所有 `fit` 都在训练折；Pipeline 优先  
- [ ] `validate_submission` 已通过  
- [ ] 指标选择匹配类别分布  
- [ ] 业务交付：归档里找得到备份、底稿、报告；无编造收益  
- [ ] `ready()` 只调用一次；种子固定  

## 练习

从你最近一次作业里挑 **至少两条** 墙上的错误，写：

1. 原文怎么写的  
2. 会带来什么虚假好处  
3. 你改成了哪一段代码  

改完再跑对应案例课验证。
