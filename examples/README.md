# 示例入口

| 示例 | 类型 | 说明 | 是否可离线跑 |
| --- | --- | --- | --- |
| [analyzer_demo.py](analyzer_demo.py) | 脚本 | `DataAnalyzer` EDA 与报告 | 是（合成数据） |
| [business_intake_demo.ipynb](business_intake_demo.ipynb) | Notebook | 业务流水线 D01–D11 | 是（合成订单） |
| [advanced/](advanced/) | 12 Notebook | 竞赛/学习进阶案例 | 部分需 OpenML/本地 CSV；有 DEMO 回退 |

## 建议学习顺序

1. `analyzer_demo.py` — 熟悉 EDA
2. `advanced/01_iris.ipynb` → `02_titanic.ipynb` → `04_breast_cancer.ipynb`
3. `business_intake_demo.ipynb` — 业务 Skill 主链
4. 其余 advanced 案例按 [advanced/README.md](advanced/README.md)

## 一键验证

```bash
# 从仓库根目录
python scripts/smoke_docs_examples.py
python scripts/run_advanced_evidence.py --only 01_iris,02_titanic,04_breast_cancer
```

跑通证据汇总：`docs/evidence/ADVANCED_EVIDENCE.md`。
