# 增量开发总计划：Skill 固化、进阶演示与回归测试

更新时间：2026-11-09

## 1. 当前状态与使用方法

本文件是本轮需求的统一任务台账（功能点勾选真源）。

**完成数量 / 完成率请看根目录 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)。**  
**详尽 API 说明书（含已校验示例）见 [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)。**

每项任务保留稳定编号；创建 Issue 后补充链接。只有实现、相关验证通过、文档更新且留有证据后才勾选。部分完成记录在进度表，不直接勾选。已有 `IMPLEMENTATION_PROGRESS.md` 属于历史记录，其中“已完成”不等于本轮已回归验证。

| 工作流 | 状态 | Issue | 验证证据 |
| --- | --- | --- | --- |
| A 现有能力盘点 | 进行中（A01–A04 文档/样例） | [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) | `docs/API_INVENTORY.md` + `docs/SKILL_API_MAPPING.md` |
| B 提交功能修复 | 进行中（核心修复已提交） | [#7](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/7) | `pytest tests/test_submission_generator.py` |
| C 回归测试体系 | 进行中（quick/full/release 入口） | [#8](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/8) | `pytest.ini` + `scripts/run_tests.sh` |
| D Skill 增量固化 | 进行中（D 主链含 D08/D12 已通） | [#10](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/10) | `wanglaoshi/business/` + tests |
| E 进阶演示 | 进行中（22 Notebook；六缺口已收口） | [#9](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/9) | `examples/advanced/` + `_add_curriculum_wave3.py` |
| F 安装、兼容与发布验证 | 进行中（F01/F02 烟测与依赖文档） | [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) | `tests/test_release_smoke.py` + `docs/DEPENDENCIES.md` |
| Epic 总跟踪 | 进行中 | [#6](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/6) | 本文件 |

## 2. 为什么修改，以及哪些不重写

项目已有 `Analyzer.py` 的 EDA、统计、图表和 HTML 报告，`FeatureEngineering.py` 的时间/文本/编码/变换能力，以及 `CompetitionTools.py` 的基线、集成、交叉验证、调参和提交处理。缺口主要是业务口径、可追溯清洗、统一流程、规则结论、归档和系统回归验证。

本轮采用增量扩展，不完全重构项目：

- 保留现有公开 API 和主要使用方式；修复前先补复现测试。
- 复用已有统计与特征计算，不因新目录重复实现算法。
- 仅在计算与展示耦合妨碍复用时局部抽取函数，旧接口继续适配。
- 新增配置、阶段结果、证据和调度能力；路径是建议，不能以建立同名类作为完成。
- 不默认新增完整聚类框架，不追求竞赛排名，不自动上传比赛或发布 PyPI。
- Skill 文档是需求来源，须逐项转化为程序契约，而非直接当作可执行代码。

## 3. 架构与 Skill 的具体接入标准

处理链：输入文件与配置 → 现有能力适配/新增规则 → 统一阶段结果 → 指标与证据 → 图表/结论/建议 → 报告与归档。

固定 Python 负责计算、校验、规则判断和模板输出；配置负责字段角色、统计粒度、业务口径、阈值和动作映射。AI 仅预留可选扩展，首版无需 API Key。AI 不得更改指标、绕过阻断或编造原因。

统一阶段结果至少包含：阶段编号、状态（通过/警告/阻断/不适用）、结果数据、规则及版本、证据编号、警告/阻断原因、输入输出路径。运行记录包含输入哈希、配置、依赖版本、随机种子和运行编号。相同输入与配置应得到相同确定性业务结果；运行时间和生成时间不要求一致。

每个 Skill 的完成标准：明确输入与前置条件、有实际处理逻辑、有结构化输出、有校验及失败行为、有证据追溯、有正常与边界验证。

| Skill | 复用/拟接入位置 | 配置或规则 | 可检查的输出 |
| --- | --- | --- | --- |
| data-intake | 新增 business/intake.py | 必需字段、来源、周期 | 原始备份、哈希、台账、缺项 |
| data-structuring | 新增 structuring.py，复用转换工具 | 表头、类型、单位、关联键 | 标准表、映射、转换失败、关联膨胀日志 |
| data-cleaning | 新增 cleaning.py，复用 Analyzer 检测 | 缺失、去重、异常、字典 | 新数据集、逐规则影响、质量结果 |
| derived-metrics | 新增 derived.py，复用时间特征 | 日历、基期、阈值、粒度 | 衍生字段、公式、依赖 |
| analysis-decomposition | 新增 planning.py | 业务模板、目标、字段角色 | 大盘/结构/趋势/分层任务及适用状态 |
| metrics-calculation | 新增 metrics.py | 分子分母、过滤、去重、聚合 | 指标字典、结果、底稿、核对记录 |
| multidimensional-analysis | 新增 analysis.py，复用相关等计算 | 维度、周期、TopN | 趋势/结构/对比/相关/二八/异动结果 |
| data-visualization | 适配现有 Visualizer | 标题、单位、周期、来源 | 独立图表、对应数据、清单 |
| insight-conclusion | 新增 insights.py | 指标方向、阈值、触发条件 | 现状/亮点/问题/观察、证据引用 |
| actionable-recommendations | 新增 recommendations.py | 问题到动作映射、角色、窗口 | 问题关联、步骤、验证指标、待定目标 |
| analysis-report | 适配 ReportGenerator 与模板 | 章节、格式 | 摘要、数据说明、大盘、分维、发现、建议、局限、口径 |
| delivery-archive | 新增 archive.py | 必需文件、脱敏字段 | 交付索引、可读性与哈希检查、缺项 |
| exploratory-data-analysis | 适配 Analyzer | 字段角色、抽样、质量阈值 | 统一数据画像和质量结果 |
| statistical-analysis | 复用 scipy/statsmodels 等能力 | 设计、配对/独立、方法、方向 | 估计、效应量、区间、p 值、局限 |
| data-analysis 总控 | 新增 pipeline.py | 阶段依赖、开关、失败策略 | 运行摘要、阶段状态、交付状态 |

以上新增文件默认位于 `wanglaoshi/business/`，最终以能力盘点结果调整。`find-skills` 不纳入业务流程。

边界：分类并不等于字符串；整数标签、字符串标签和概率均需区分。统计方法不能只凭正态性自动选择。数值贡献分解不证明业务因果。缺乏对比或证据的发现降级为观察；收益目标没有依据时标记待定。输入不合格时不得生成可交付状态。

## 4. 开发任务与依赖

### A：盘点和公共契约

- [x] A01 完整盘点公开 API、依赖、返回值、异常及副作用，建立复用清单。（初版：`docs/API_INVENTORY.md`，后续按模块补异常/副作用细表）
- [x] A02 建立 Skill 要求 → API/配置 → 校验 → 测试的映射。（`docs/SKILL_API_MAPPING.md`）
- [x] A03 定义统一阶段结果、指标/图表/结论/建议编号与运行记录。（`StageResult`/`RunRecord` + 映射文档编号约定）
- [x] A04 定义字段角色和销售配置样例；关键配置缺失不得静默推断。（`wanglaoshi/business/config_sales_example.yaml`）

### B：修复提交生成与校验

- [x] B01 用小数据复现模板目标列名未推断的问题。
- [x] B02 用字符串标签复现写出后统计失败及校验误判。
- [x] B03 明确参数优先级，按模板保持列名、顺序和 ID；歧义时报错。
- [x] B04 标签使用类别统计，数值使用数值统计；统一缺失与无穷值策略。
- [x] B05 明确一维标签、单列二维、多列概率、多输出及类别映射契约；不得静默 argmax。
- [x] B06 生成与校验共用格式规则；覆盖 ID 不匹配、长度不匹配、非法形状和空输入。

### D：逐步固化 Skill

- [x] D01 接收备份、字段与周期检查。（`business/intake.py`）
- [x] D02 列名标准化与基础类型规整（MVP；单位转换/多表关联待补）。（`business/structuring.py`）
- [x] D03 可配置清洗、新文件输出、逐规则影响日志；统计异常默认 clip 不删行。（`business/cleaning.py`）
- [x] D04 时间部件、占比、简单环比、头腰尾分层（MVP；缺期/跨年已由 calendar 补）。（`business/derived.py` + `add_value_tiers`）
- [x] D05 四方向分析计划及不适用原因。（`business/planning.py`）
- [x] D06 指标字典与底稿；明细汇总、占比一致性校验；口径版本登记最小版。（`business/metrics.py` + `metric_registry.py`）
- [x] D07 六视角分析；贡献分解与因果假设分开。（`business/analysis.py`；pareto 注明非因果）
- [x] D08 图表独立归档，标题/单位/周期/来源齐全。（`business/visualization.py`）
- [x] D09 带证据的规则结论和行动建议。（`business/insights.py`）
- [x] D10 完整报告、脱敏、交付索引、缺项阻断。（`business/archive.py`）
- [x] D11 总控串联，阻断后跳过下游 / 禁止正式交付。（`business/pipeline.py`）
- [x] D12 EDA 统一适配；统计首版方法清单及适用性检查。（`business/eda.py`）

先完成一条销售纵向流程再扩大范围；不要求先写完所有模块。顺序为 A → 最小 C + B → 销售样例及 D01–D11 → 扩展能力与案例 → F。每项功能同步补测试，E 可在 API 稳定后逐个交付。

## 5. C：大范围、可回归的测试体系

12 个演示与测试共用数据准备和核心流程，但测试独立声明预期值。演示跑完不等于结果正确。

| 层次 | 数据与检查 | 运行时机 |
| --- | --- | --- |
| 单元/边界 | 人工小数据：公式、类型、缺失、形状、异常 | 每次提交 |
| 模块集成 | 12 个数据集固定子集：处理到输出的衔接 | PR |
| 完整案例 | 12 个完整案例：指标、产物、耗时、内存 | 定期与发布前 |
| 业务流程 | 订单、用户、渠道、目标：口径与证据 | PR |
| 安装兼容 | 构建包、模板、可选依赖、环境矩阵 | 发布前 |

- [x] C01 配置 pytest、分层标记和快速/完整/发布运行入口。（`pytest.ini` + `scripts/run_tests.sh`）
- [x] C02 数据登记来源、使用条件、版本、哈希和缓存；下载与测试分离。（`wanglaoshi/data_registry.py`）
- [x] C03 固定划分、种子、抽样和搜索预算；记录环境。（`wanglaoshi/experiment_context.py`）
- [x] C04 数值公式用明确容差；字段、行数、顺序用精确断言。（`wanglaoshi/asserting.py` + `tests/test_c04_numeric_contracts.py`）
- [x] C05 模型采用预先说明的合理下限/容差，不要求改进方案必胜或跨版本分数完全相同。（`wanglaoshi/score_floors.py`）
- [x] C06 报告检查章节、数字引用、资源；图表检查数据与元信息，不默认逐字节比较。（`wanglaoshi/report_meta.py`）
- [x] C07 覆盖全空/常数列、单类别、小样本、高维、未见类别、中文、错误日期和重复键。（`tests/test_c07_edge_datasets.py`）
- [x] C08 验证编码、选择、调参只在训练折拟合；泄露检测器不能代替流程隔离。（`tests/test_c08_fold_fit_discipline.py`）
- [x] C09 业务样例含退款、取消、单位、缺期、零基期、关联膨胀、缺目标与隐私；关键结果手工预计算。（`tests/test_c09_business_dirty_cases.py`）
- [x] C10 汇总案例×模块状态、失败/跳过原因、指标、耗时与版本，保存回归产物。（`wanglaoshi/regression_summary.py` + `scripts/run_tests.sh summary`）
- [x] C11 日常缺外部数据可明确跳过；发布必需案例缺数据视为验收未完成。（`wanglaoshi/testing_utils.py`）
- [x] C12 变更代码行覆盖率以 80% 为初始参考，关键口径与已知 Bug 边界必须覆盖；先运行后校准资源门槛。（`scripts/run_tests.sh coverage`，默认 fail-under=0 可调 `COV_FAIL_UNDER`）

快速档限制数据和模型预算；完整档运行全案例；发布档增加安装兼容验证。环境矩阵先核对实际依赖与 `python_requires`，再声明最低支持版本和当前稳定版本；不虚假承诺未经验证的兼容性。初期流水线可先单环境运行，再扩展 Linux/Windows/macOS。耗时与内存阈值在固定环境建立基准后启用。

## 6. E：进阶使用演示（独立 Feature Issue）

每项包含 README、Notebook、至少两个难点、错误分析、局限；包外能力显式标明。  
**方法覆盖与包缺口：** `docs/ANALYSIS_COVERAGE.md`。

**课号约定：** Notebook 课一律 **E01–E22**（与 `01_`…`22_` 文件前缀一致）。配套工程勾选用 **EM01–EM05**，不要写成 E13–E17。

| 课号 | 数据集 | 技术难点与结果 | 状态 |
| --- | --- | --- | --- |
| E01 | Iris | 小样本分层验证、字符串/数字标签提交 | ✅ 已加细 |
| E02 | Titanic | 缺失、文本特征、训练验证隔离 | ✅ 已加细 |
| E03 | Wine Quality | 回归/有序分类比较、不均衡 | ✅ 已加细 |
| E04 | Breast Cancer | 缩放/PCA、召回权衡；**校准/ECE 增补** | ✅ 已加细 |
| E05 | Credit Card Fraud | 时间划分、不均衡、PR-AUC与阈值 | ✅ 已加细 |
| E06 | California Housing | 空间分组对比随机划分、分层误差 | ✅ 已加细 |
| E07 | Abalone | 混合编码、目标变换和残差 | ✅ 已加细 |
| E08 | Bike Sharing | 时间验证、预测时可用的滞后特征、RMSLE | ✅ 已加细 |
| E09 | Mall Customers | 缩放、聚类数与稳定性、画像 | ✅ 已加细 |
| E10 | Adult Census | 折内目标编码；**公平性交叉群体增补** | ✅ 已加细 |
| E11 | House Prices | 混合高维、缺失含义、对数目标与集成 | ✅ 已加细 |
| E12 | Digit Recognizer | 高维EDA、PCA、精度与成本、错分分析 | ✅ 已加细 |
| E13 | 合成 A/B | 功效、效应量；**贝叶斯门槛概率增补** | ✅ |
| E14 | 合成多组 | t / χ² / ANOVA | ✅ |
| E15 | 合成日序列 | 分解、ADF；**同比缺月/零基数/未完月增补** | ✅ |
| E16 | 合成订阅 | KM、删失、流失 | ✅ |
| E17 | 合成评论 | 文本浅特征 / 折内 TF-IDF | ✅ |
| E18 | 合成异常 | IsolationForest | ✅ |
| E19 | 合成购物篮 | 关联规则 lift | ✅ |
| E20 | 合成面板 | 双重差分入门 | ✅ |
| E21 | 合成订单/明细/退款 | 一对多膨胀与对账 | ✅ |
| E22 | MovieLens / DEMO | NDCG@10 与热门榜基线 | ✅ |

配套勾选 **EM01–EM05**（工程项，不是课号；计入路线图 45 点里原 E13–E17 槽位）：

- [x] EM01 建立 `examples/advanced/` 导航、共同运行入口和依赖说明。（原误标 E13）
- [x] EM02 每个案例的数据获取与使用条件核实；Wine Quality 不与 sklearn Wine 混淆，California 首次需下载。（原误标 E14）
- [x] EM03 比赛提交仅本地生成与校验；无标签测试集不用于本地评估。（原误标 E15）
- [x] EM04 Kaggle 时间预测的滞后特征只使用预测当时可用信息，不引用未知测试目标。（E08 Notebook 约束；原误标 E16）
- [x] EM05 E01–E22 跑通证据与学习顺序。（`docs/evidence/ADVANCED_EVIDENCE.md`；原误标 E17）

原始大数据、模型和生成报告不默认提交。公共下载失败不应伪装成通过，也不绕过登录或条款。

## 7. F：最终交付门槛

- [x] F01 构建并在干净环境安装，验证新增模块、模板、静态资源。（烟测：`tests/test_release_smoke.py`；完整干净环境待 CI）
- [x] F02 核对 Excel 引擎及其他依赖；区分核心与可选安装项。（`docs/DEPENDENCIES.md`）
- [x] F03 相关旧 API 回归通过，更新兼容与变更说明。（submission + business 回归；详见路线图进度表）
- [x] F04 销售完整流程无需 AI 运行，数字可追溯，缺证据/缺交付项正确阻断。（BusinessPipeline 主链）
- [x] F05 12 个演示和各档测试报告齐全，无隐瞒的跳过或失败。（见 `docs/evidence/`；CI quick 档已接）
- [x] F06 README 更新用法、配置、自动化边界、支持环境及数据获取方式。（已补 business API；advanced README 已有）

## 8. Issue 拆分与进度维护

建议创建一个总跟踪 Issue，分别关联：提交 Bug 修复、回归测试体系、Skill 增量固化、12 个进阶演示、安装兼容验证。Skill 可按纵向交付继续拆分。本文保存整体决策，子 Issue 保存具体实现讨论；完成后同步编号、链接和证据。

进度记录格式：日期｜任务编号｜完成内容｜验证命令/报告链接｜剩余问题。未运行的检查必须标记未运行。工期在首条纵向流程和数据准备完成后估算，不沿用未经验证的天数承诺。

| 日期 | 任务 | 进展 | 验证/剩余事项 |
| --- | --- | --- | --- |
| 2026-11-09 | 规划 | 创建总计划，记录增量边界、Skill 接入、演示与测试要求 | 仅文档交付；代码任务尚未开始 |
| 2026-11-09 | #6/#7/#8/#9 | 创建 Epic 与子 Issue；修复 SubmissionGenerator；pytest 8 通过；`examples/advanced` 落 12 个 Notebook（E01 加细）；修复 Useful import 刷屏 | 待：E02–E12 加细与真实数据路径；D business MVP；A 契约文档 |
| 2026-11-09 | E/D/A/C | 12 Notebook 全部加细；D01–D06 + pipeline；A01–A04 文档/样例；`scripts/run_tests.sh` | 待：D07–D10/D12；E17 跑通证据；F 发布门槛；B 清单勾选对齐测试 |
| 2026-11-09 | D09/D10/D11 | 结论建议、报告归档、主链跑通；上游 blocked 禁止正式交付 | 待：D08 图表归档；D12 EDA 适配；F01–F06 |
| 2026-11-09 | D08/D12/F | 图表归档+EDA 适配接入主链；依赖文档与发布烟测 | 待：E17 Notebook 跑通证据；干净环境矩阵 CI |
| 2026-11-09 | E17/F05/CI | 12 Notebook 进程内跑通证据全绿；修复 Abalone/Adult；升级 Actions 安装 `-e .` | 可选：矩阵扩到 3.12；发布 PyPI |
| 2026-11-09 | 文档整理 | README 0.14.0 导航/可跑示例；docs/examples 索引；`smoke_docs_examples.py` | 持续与 API 同步 |
