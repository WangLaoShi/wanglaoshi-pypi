# 项目进度仪表盘（心中有数）

**更新日期：** 2026-11-09  
**产品名：** 基线工坊 · 从 EDA 到提交  
**包名：** `wanglaoshi`（导入路径不变）

> **本文件是「完成率」的权威摘要。**  
> 功能点定义与勾选规则仍以 [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) 为准；勾选条件：实现 + 验证 + 文档/证据。

---

## 1. 一眼看清

| 口径 | 总数 | 已完成 | 未完成 | 完成率 |
| --- | ---: | ---: | ---: | ---: |
| **路线图正式功能点（A/B/C/D/EM01–05/F 勾选框）** | **45** | **38** | **7** | **84.4%** |
| 其中：业务 Skill 主链 D01–D12 | 12 | 12 | 0 | 100%（**MVP**，深水区另计） |
| 其中：回归体系 C01–C12 | 12 | 5 | 7 | 41.7% |
| 进阶 Notebook **E01–E22**（表状态，非勾选框） | 22 | 22 | 0 | 100%（含对账/NDCG/六缺口收口；证据 22/22） |
| Skill 深水区待办（路线图正文「待补」，未单独编号） | 10 | 0 | 10 | 0%（见 §4） |
| 教材站强化（本轮新增 backlog G） | 8 | 1 | 7 | 12.5%（见 §5） |

**给你做决策用的读法：**

- 若问「这轮 Epic 勾选完了没」→ 看 **45 点 / 75.6%**  
- 若问「业务流水线能不能演示」→ D **12/12 MVP 已通**，但深水区几乎没做  
- 若问「测试体系硬不硬」→ C 只完成入口，**还差 11 项**  
- 若问「学生教材够不够」→ 案例 Notebook 有了，但 **Jupyter Book 讲解层仍偏薄**（§5）

---

## 2. 功能点记在哪里

| 文档 | 作用 |
| --- | --- |
| [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) | **功能点编号与勾选真源**（A/B/C/D/E/F） |
| **本文件 `PROJECT_STATUS.md`** | **完成数量、完成率、缺口优先级**（给你心中有数） |
| [docs/SKILL_API_MAPPING.md](docs/SKILL_API_MAPPING.md) | Skill → API → 测试映射 |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | **详尽 API 说明书**（可运行示例 + `scripts/verify_api_reference.py`） |
| [docs/API_INVENTORY.md](docs/API_INVENTORY.md) | 公开 API 短盘点 |
| [docs/evidence/ADVANCED_EVIDENCE.md](docs/evidence/ADVANCED_EVIDENCE.md) | 进阶 Notebook 跑通证据 |
| [docs/ANALYSIS_COVERAGE.md](docs/ANALYSIS_COVERAGE.md) | 常见分析方法覆盖与包优化缺口 |
| [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md) | 历史记录；**不作为本轮完成率依据** |
| GitHub Issues [#6](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/6) 等 | 讨论与拆分；勾选以路线图为准 |

---

## 3. 分工作流明细（45 个正式点）

### A 盘点与契约 — 4/4 = 100%

| ID | 状态 |
| --- | --- |
| A01–A04 | ✅ |

### B 提交生成修复 — 6/6 = 100%

| ID | 状态 |
| --- | --- |
| B01–B06 | ✅ |

### D Skill 固化（主链 MVP）— 12/12 = 100%

| ID | 状态 | 备注 |
| --- | --- | --- |
| D01–D12 | ✅ MVP | 深水区见 §4，不算进这 12 个勾选的「未完成」 |

### C 回归测试体系 — 5/12 = 41.7%

| ID | 状态 |
| --- | --- |
| C01 | ✅ |
| C04 | ✅ 数值容差 / 结构精确断言 |
| C07 | ✅ 边界数据集 |
| C08 | ✅ 折内拟合 / 目标编码防泄漏 |
| C09 | ✅ 脏样例（退款/单位/缺期/膨胀/隐私等） |
| C02–C03, C05–C06, C10–C12 | ❌ 未做 |

### E 进阶演示 — 课 E01–E22 + 配套 EM01–EM05

| ID | 状态 |
| --- | --- |
| E01–E12 | ✅ 竞赛/表格基线向，已加细 |
| E13–E20 | ✅ 统计/试验/时序/生存/文本/异常/关联/DiD |
| E21–E22 | ✅ 三表对账；MovieLens NDCG |
| 增补并入 | ✅ E13 贝叶斯 A/B；E15 同比；E04 校准；E10 公平性 |
| EM01–EM05 | ✅ 导航/数据说明/提交约束/滞后纪律/跑通证据（**不是课号**） |

### F 发布门槛 — 6/6 = 100%（部分为烟测级）

| ID | 状态 | 备注 |
| --- | --- | --- |
| F01–F06 | ✅ | F01 完整干净矩阵仍可加强 |

---

## 4. Skill 深水区（未单独编号，但是真实缺口）

这些在路线图 D 条目括号里写了「待补」，**主链勾选了也不等于做完**：

| # | 缺口 | 对应阶段 | 优先级建议 |
| ---: | --- | --- | --- |
| S1 | 单位换算 | D02 | P1 → ✅ `convert_units` |
| S2 | 多表关联 + 关联膨胀检查 | D02 | P0 → ✅ 最小版 `reconciliation.py` |
| S3 | 缺期对齐 / 同比 / 跨年 | D04 | P0 → ✅ 最小版 `calendar.py`（跨年精细规则仍可加深） |
| S4 | 价值分层（头腰尾）标签 | D04 | P1 → ✅ `add_value_tiers` |
| S5 | 指标口径版本 / 自定义公式 DSL | D06 | P1 → ✅ 版本登记最小版（完整 DSL 仍待） |
| S6 | 图表类型扩展（不止趋势/结构） | D08 | P2 |
| S7 | 可量化 uplift（非一律 pending） | D09 | P1 |
| S8 | 独立 recommendations 模块深化 | D09 | P2 |
| S9 | 统计检验自动设计（非仅清单） | D12 / statistical-analysis | P0 → ✅ `StatsToolkit` / `ExperimentDesign` 最小版 |
| S10 | 业务脏样例验收（退款/取消/隐私等，同 C09） | C09 ∩ D | P0 → ✅ 与 C09 同测 |

**深水区完成率：约 7/10（S1–S5/S9/S10 已有最小版；S5 DSL 仍浅）。**

### 4.1 由 E13–E22 反推的包 API 缺口（详见 `docs/ANALYSIS_COVERAGE.md`）

| 优先级 | 建议 API | 依据课号 |
| --- | --- | --- |
| P0 | `StatsToolkit` / `ExperimentDesign` | E14 / E13 → ✅ |
| P0 | 业务多表关联膨胀（S2） | E21 → ✅ |
| P1 | `TimeSeriesTools` / `AnomalyDetector` / `RankingMetrics` / 校准 / 公平性 / 生存 | → ✅ 最小版 |
| P2 | 文本中文分词、关联规则 extras | E17 / E19 |

---

## 5. 教材站（基线工坊）强化 backlog — G

学生反馈：手册讲解太简单、示例不够。以下为本轮新增跟踪（与路线图并列，进度写入本文件）：

| ID | 内容 | 状态 |
| --- | --- | --- |
| G01 | 建立本进度仪表盘并链到文档/手册 | ✅（本文件） |
| G02 | 增加「精讲课」：泄露 / 划分 / 提交契约 / 业务数字追溯 | 进行中 |
| G03 | 加厚三条主线文稿（Analyzer / 竞赛 / 业务）到可自学深度 | 进行中 |
| G04 | 每课练习增加「参考思路」与最小可运行片段 | 进行中（01–12 有；13–20 已挂练习页） |
| G05 | 新增 2+ 端到端精讲 Notebook（竞赛一条 + 业务一条） | 待办 |
| G06 | 手册内嵌「常见错误墙」与对照表 | 待办 |
| G07 | 学习路径按课时（45/90 分钟）重排 | 待办 |
| G08 | 教材完成度纳入发布检查 | ✅ `release_check.sh full` 含 build_book |
| G09 | 详尽 API 说明书 + 自动校验 | ✅ `docs/API_REFERENCE.md` |
| G10 | 发布步骤收成一条命令 | ✅ `scripts/release_check.sh` + `docs/RELEASE.md` |
| G11 | 中文术语表扩到统计/时序/生存/因果 | ✅ `book/reference/glossary.md` |
| G12 | 分析覆盖矩阵 + E01–E22（六缺口收口） | ✅ `docs/ANALYSIS_COVERAGE.md` |

---

## 6. 建议下一期（让完成率「对学生也有意义」）

1. **P0 工程：** `StatsToolkit` + `ExperimentDesign`；C09 + S2/S3  
2. **P0 教材：** G02/G03/G05；构建 Pages 使 20 课上线  
3. **P1：** `TimeSeriesTools` / 生存 / 异常封装；C02–C04  

---

## 7. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-11-09 | 初建仪表盘：45 点完成率 75.6%；登记深水区 10 项与教材 G01–G08 |
| 2026-11-09 | 补 G09：`docs/API_REFERENCE.md` + `scripts/verify_api_reference.py`（19/19 通过）；修复 `analyze_notebook` |
| 2026-11-09 | G10：`./scripts/release_check.sh` 统一日常/发版/大版本检查；见 `docs/RELEASE.md` |
| 2026-11-09 | G11/G12：术语表扩写；E13–E20；覆盖矩阵；证据 20/20 |
| 2026-11-09 | wave3：E21–E22 + 六缺口收口；证据 22/22 |
| 2026-11-09 | 课号统一为 E01–E22；配套勾选改称 EM01–EM05 |
| 2026-11-10 | P0：`reconciliation` / `calendar` / `stats` 最小版 + 测试 |
| 2026-11-10 | P1：`TimeSeriesTools` / `CalibrationReport` / `GroupMetricsReport` / `AnomalyDetector` / `RankingMetrics` |
| 2026-11-10 | C09 脏样例 + S1 `convert_units` + `SurvivalAnalyzer`；修复 report.md 脱敏 |
| 2026-11-10 | C08 折内拟合回归 + S4 `add_value_tiers`；修复 target_encode fillna 警告 |
| 2026-11-10 | C04/C07 回归 + S5 `metric_registry` + `asserting` 工具 |
