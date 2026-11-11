```{note}
本页从仓库 `docs/handbook/README.md` 同步生成；请以仓库文件为编辑源。
```

# 使用手册（类与方法 · 带案例）

本目录是相对 [`docs/API_REFERENCE.md`](../reference/api)（查签名）与 [`docs/tutorials/`](../tutorials/)（公开接口短教程）的**第三层文档**：按**类 / 方法**讲「什么时候用、怎么用、一个完整小案例」。

| 手册 | 核心类 / 模块 | 案例数据 |
| --- | --- | --- |
| [01_data_analyzer.md](01_data_analyzer.md) | `DataAnalyzer` | seaborn tips + sklearn Iris |
| [02_feature_engineer.md](02_feature_engineer.md) | `AdvancedFeatureEngineer` | 合成订单（时间/文本/类别/数值） |
| [03_compete_toolkit.md](03_compete_toolkit.md) | `compete.*` | seaborn Titanic |
| [04_analytics_toolkit.md](04_analytics_toolkit.md) | `analytics.*` | 合成序列 / 异常 / 删失 / 排序 |
| [05_ml_guide.md](05_ml_guide.md) | `ml.*` | Breast cancer |
| [06_business_guide.md](06_business_guide.md) | `business.*` | 合成多表订单 |

可执行脚本：[`examples/handbook/`](https://github.com/WangLaoShi/wanglaoshi-pypi/blob/main/examples/handbook/)

```bash
PYTHONPATH=. python examples/handbook/01_data_analyzer.py
PYTHONPATH=. python scripts/smoke_docs_examples.py
```

阅读建议：先 tutorials 上手 → 本手册按类深挖 → API_REFERENCE 查签名 → advanced E0x 做完整课。
