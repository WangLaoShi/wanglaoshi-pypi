# wanglaoshi 架构与迁移（零破坏）

**更新日期：** 2026-11-10  
**原则：** 先加门面 → 文档引导 → 再 deprecate → 最后搬实现。旧导入路径在过渡期内保持可用。

---

## 1. 目标分层

```text
wanglaoshi/
  compete/      # 竞赛：提交 / 基线 / 泄露 / CV / 集成 …
  analyze/      # EDA / 报告（主路径 DataAnalyzer）
  features/     # 特征工程
  business/     # 业务流水线（已有）
  analytics/    # 时序 / 生存 / 排序 / 公平 / 校准 / 异常 / stats
  runtime/      # ready / Jupyter* / VERSIONS / WebGetter
  _testing/     # 断言 / 分数下限 / 回归汇总 / 数据台账 / 实验上下文
```

| 门面 | 当前实现位置（Phase 1 仍住这里） | 推荐导入 |
| --- | --- | --- |
| `compete` | `CompetitionTools.py` | `from wanglaoshi.compete import SubmissionGenerator` |
| `analyze` | `Analyzer.py` | `from wanglaoshi.analyze import DataAnalyzer` |
| `features` | `FeatureEngineering.py` | `from wanglaoshi.features import AdvancedFeatureEngineer` |
| `business` | `business/*` | `from wanglaoshi.business import BusinessPipeline` |
| `analytics` | 根上各小模块 + `stats/` | `from wanglaoshi.analytics import TimeSeriesTools` |
| `runtime` | `ready` / `Jupyter*` / … | `from wanglaoshi.runtime import ready` |
| `_testing` | `asserting` / `score_floors` / … | `from wanglaoshi._testing import assert_close` |

顶层 `from wanglaoshi import DataAnalyzer, ready, …` **继续支持**（兼容课堂旧代码）。

---

## 2. 迁移顺序（已定）

### Phase 1 — 门面（本轮）✅

- 新增上述子包，仅 **re-export**，不搬文件。
- 写本文档；加架构烟测。
- **不发** DeprecationWarning（避免刷屏）。

### Phase 2 — 收缩公开面 ✅

- 文档/README 主示例改用门面导入；课堂旧导入仍可用。
- `__all__` 区分 `_STABLE_PUBLIC` 与 `_COMPAT_EXPORTS`。
- `Analyzer_Plain` / `wqe` 源码与 API 盘点标注 *legacy*。

### Phase 3 — soft deprecate

- 对 `from wanglaoshi.wqe import …`、`Analyzer_Plain` 等发 `DeprecationWarning`（可 env 关闭）。
- `CompetitionTools` 大类按职责拆文件，仍由 `compete` 与旧模块路径 re-export。

### Phase 4 — 搬实现

- 物理移动到子包目录；根上旧模块变为薄 shim：`from wanglaoshi.compete import *`。
- 可选 extras：`[compete]` / `[business]` / `[dev]`（测试门禁）。

---

## 3. 明确不做什么（Phase 1）

- 不改 Notebook 课号与 Book 目录。
- 不强制用户改 import。
- 不在本轮拆 `CompetitionTools.py` / `Analyzer.py` 实体文件（体积大，单独 PR）。

---

## 4. 兼容性验收

```bash
python -c "from wanglaoshi import DataAnalyzer, CompetitionTools, ready"
python -c "from wanglaoshi.compete import SubmissionGenerator"
python -c "from wanglaoshi.analyze import DataAnalyzer"
python -c "from wanglaoshi.analytics import StatsToolkit"
python -c "from wanglaoshi._testing import assert_close"
./scripts/run_tests.sh quick
```
