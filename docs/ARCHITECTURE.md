# wanglaoshi 架构与迁移

**更新日期：** 2026-11-11 · **当前版本：1.15.1**  
**原则：** 公开写法以**稳定公开接口**（Facade / public API surface）为准；根上旧路径仅作过渡 shim，**不再作为文档 / 示例 / 教材推荐**。

---

## 1. 目标分层

```text
wanglaoshi/
  compete/      # 竞赛：submission / baseline / leakage / …
  analyze/      # EDA（core.py）+ legacy（plain.py / wqe.py）
  features/     # 特征工程（engineering.py）
  business/     # 业务流水线
  analytics/    # 时序 / 生存 / 排序 / 公平 / 校准 / 异常 / stats/
  ml/           # ML/DL（core.py；torch 可选）
  runtime/      # ready / jupyter_* / versions / web_getter / useful
  _testing/     # 断言 / 分数下限 / 回归汇总 / 数据台账 / 实验上下文
```

| 接口模块 | 实现位置 | 根上兼容路径 |
| --- | --- | --- |
| `compete` | `compete/*.py` | `CompetitionTools.py`（shim） |
| `analyze` | `analyze/core.py` | `Analyzer.py`（shim） |
| `features` | `features/engineering.py` | `FeatureEngineering.py`（shim） |
| `business` | `business/*` | 同左 |
| `analytics` | `analytics/*.py` + `analytics/stats/` | `timeseries.py` / `stats/`（shim） |
| `ml` | `ml/core.py` | `MLDL.py`（shim） |
| `runtime` | `runtime/*.py` | `ready.py` / `JupyterEnv.py` 等（shim） |
| `_testing` | `_testing/*.py` | `asserting.py` 等（shim） |
| legacy EDA | `analyze/plain.py` / `analyze/wqe.py` | `Analyzer_Plain.py` / `wqe.py`（shim + warning） |

自 **1.15.1** 起，请统一：

```python
from wanglaoshi.runtime import ready
from wanglaoshi.analyze import DataAnalyzer
from wanglaoshi.compete import QuickBaseline, SubmissionGenerator
from wanglaoshi.features import AdvancedFeatureEngineer
from wanglaoshi.ml import DataPreprocessor, MLModel
```

旧写法（`from wanglaoshi import CompetitionTools as CT` / 顶层 `DataAnalyzer` 等）已放弃为推荐路径。

---

## 2. 迁移顺序（已定）

### Phase 1 — 稳定公开接口 ✅

- 新增子包，仅 re-export；写本文档；架构烟测。

### Phase 2 — 收缩公开面 ✅

- `__all__` 分 `_STABLE_PUBLIC` / `_COMPAT_EXPORTS`；README/API 改公开接口；Plain/wqe 标 legacy。

### Phase 3 — soft deprecate ✅

- 导入 `wqe` / `Analyzer_Plain` 发 `DeprecationWarning`（`WANGLAOSHI_SILENCE_LEGACY=1` 可关）。
- `CompetitionTools` 拆入 `compete/*`。

### Phase 4 — 搬实现 ✅

- `Analyzer` / `FeatureEngineering` / analytics 叶子 / runtime / `_testing` 物理迁入子包；根模块为薄 shim。
- extras：`[compete]`（=optuna）、`[business]`（=openpyxl）；`[dev]` 仍为测试门禁。

---

## 3. 1.15.1 起的公开约定

- `examples/`、`book/`、`docs/`、`README` **全部改为公开接口导入**；旧顶层捷径不再写入新示例。
- 根 shim 暂留，便于未升级课堂环境；**新代码勿再依赖**。
- Phase 4 之后补迁已完成：`stats/` → `analytics/stats/`；`Analyzer_Plain`/`wqe` → `analyze/{plain,wqe}.py`；`MLDL` → `ml/core.py`。

---

## 4. 兼容性验收

```bash
python scripts/check_architecture.py   # 或 ./scripts/run_tests.sh arch
./scripts/run_tests.sh quick
./scripts/run_tests.sh coverage        # 行覆盖门槛 + 函数级 100%
```

手工等价检查：

```bash
python -c "from wanglaoshi.runtime import ready; from wanglaoshi.analyze import DataAnalyzer"
python -c "from wanglaoshi.compete import SubmissionGenerator, QuickBaseline"
python -c "from wanglaoshi.features import AdvancedFeatureEngineer"
python -c "from wanglaoshi.analytics import StatsToolkit"
python -c "from wanglaoshi.ml import DataPreprocessor, MLModel"
python -c "from wanglaoshi._testing import assert_close"
```
