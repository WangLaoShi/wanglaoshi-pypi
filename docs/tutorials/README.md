# 门面教程（1.15.1+）

这是相对 E01–E22 **进阶案例**的第二学习路径：按 **门面 API** 拆开，用公开/内置数据在 10–20 分钟内跑通「怎么调用」。

- 详尽签名：[`docs/API_REFERENCE.md`](../API_REFERENCE.md)
- 架构分层：[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
- 深案例：[`examples/advanced/`](../../examples/advanced/)
- 可执行烟测脚本：[`examples/tutorials/`](../../examples/tutorials/)

## 学习顺序

```text
安装 wanglaoshi==1.15.1+
        │
        ▼
  01 ready + EDA  ──►  02 compete 基线/提交  ──►  03 features 编码
        │                      │
        │                      ├──►  07 泄漏 + CV
        │                      └──►  06 ml 预处理/建模
        │
        ├─► 04 analytics 统计 ──► 08 时序 ──► 09 校准/公平
        ├─► 05 business 流水线
        └─► 10 _testing 复现治理
                          │
                          ▼
                   E01–E22 进阶案例
```

### 第一批（上手）

| 课 | 文档 | 脚本 | 数据 | 门面 |
| --- | --- | --- | --- | --- |
| 01 | [01_ready_and_eda.md](01_ready_and_eda.md) | `examples/tutorials/01_ready_and_eda.py` | sklearn Iris + seaborn tips | `runtime` + `analyze` |
| 02 | [02_compete_baseline.md](02_compete_baseline.md) | `examples/tutorials/02_compete_baseline.py` | seaborn Titanic | `compete` |
| 03 | [03_features_encode.md](03_features_encode.md) | `examples/tutorials/03_features_encode.py` | 合成高基数表 | `features` |
| 04 | [04_analytics_stats.md](04_analytics_stats.md) | `examples/tutorials/04_analytics_stats.py` | 合成两组样本 | `analytics` |
| 05 | [05_business_pipeline.md](05_business_pipeline.md) | `examples/tutorials/05_business_pipeline.py` | 合成订单 CSV | `business` |
| 06 | [06_ml_preprocess.md](06_ml_preprocess.md) | `examples/tutorials/06_ml_preprocess.py` | sklearn Breast cancer | `ml` |

### 第二批（进阶门面）

| 课 | 文档 | 脚本 | 数据 | 门面 |
| --- | --- | --- | --- | --- |
| 07 | [07_leakage_and_cv.md](07_leakage_and_cv.md) | `examples/tutorials/07_leakage_and_cv.py` | seaborn Titanic | `compete` |
| 08 | [08_timeseries_tools.md](08_timeseries_tools.md) | `examples/tutorials/08_timeseries_tools.py` | 合成季节序列 | `analytics` |
| 09 | [09_calibration_fairness.md](09_calibration_fairness.md) | `examples/tutorials/09_calibration_fairness.py` | Breast cancer + 合成分组 | `analytics` |
| 10 | [10_testing_repro.md](10_testing_repro.md) | `examples/tutorials/10_testing_repro.py` | Iris + 临时台账 | `_testing` |

## 一键烟测

```bash
# 仓库根目录
PYTHONPATH=. python scripts/smoke_docs_examples.py
# 或只跑教程脚本
PYTHONPATH=. python examples/tutorials/01_ready_and_eda.py
```

## 约定

- 只教门面导入：`from wanglaoshi.analyze import DataAnalyzer` 等
- 优先离线可跑（sklearn / seaborn 内置；业务用合成表）
- 教程讲「最短可用路径」；竞赛纪律与深水区仍以 E0x / 精讲课为准
