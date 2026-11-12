# 精讲课总览

三条主线告诉你「模块在哪」；**精讲课**告诉你「为什么这样用、错一次会死在哪里」。每课都有：概念 → 可运行片段 → 对照错误 → 自测。

| 课 | 你带走的能力 | 预计时长 |
| --- | --- | --- |
| {doc}`leakage` | 识别目标/时间/分布三类泄露，并会用检测器 | 45–60 分钟 |
| {doc}`cv_and_splits` | 按任务选分层 / 时间 / 分组划分，解释分数可信度 | 45–60 分钟 |
| {doc}`submission_contracts` | 按模板契约生成与校验提交，处理标签 vs 概率 | 45 分钟 |
| {doc}`feature_fit_discipline` | 编码与选特征只在训练折拟合，避免「看起来很好」 | 45 分钟 |
| {doc}`business_traceability` | 读懂 `StageResult`、阻断与证据编号，能交付可追溯结果 | 60 分钟 |
| {doc}`common_mistakes` | 对照「错误墙」，快速自查作业与项目 | 30 分钟 |

```{admonition} 怎么和案例课配合
:class: tip
先读本精讲 → 再打开对应 Notebook（如 Titanic / Adult / Bike）→ 最后做练习页。不要反过来：只看 Notebook 很容易「跑通却讲不清」。
```

项目功能完成率（给助教 / 自学规划）：仓库根目录 [`PROJECT_STATUS.md`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/PROJECT_STATUS.md)，手册摘要见 {doc}`../reference/project-status`。
