# 示例入口

新代码只教门面导入（`wanglaoshi.analyze` / `compete` / `features` / `ml` / `runtime` / `business` 等），见 [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)。

| 示例 | 类型 | 说明 | 是否可离线跑 |
| --- | --- | --- | --- |
| [tutorials/](tutorials/) | 脚本 | **门面短教程**（对应 [docs/tutorials/](../docs/tutorials/)） | 是（sklearn / seaborn / 合成） |
| [analyzer_demo.py](analyzer_demo.py) | 脚本 | `DataAnalyzer` EDA 与报告 | 是（合成数据） |
| [business_intake_demo.ipynb](business_intake_demo.ipynb) | Notebook | 业务流水线短 demo | 是（合成订单） |
| [business_pipeline_course.ipynb](business_pipeline_course.ipynb) | Notebook | 业务流水线**教材专章**（阻断/交付/脱敏） | 是（合成订单） |
| [advanced/](advanced/) | E01–E22 | 竞赛/统计/对账/推荐等全覆盖 | 部分需 OpenML/本地 CSV；E13–E22 多合成/DEMO |

## 建议学习顺序

1. [docs/tutorials/](../docs/tutorials/) — 按门面 + 公开数据上手  
2. `analyzer_demo.py` — 熟悉 EDA  
3. `advanced/01_iris.ipynb` → `02_titanic.ipynb` → `04_breast_cancer.ipynb`  
4. `business_intake_demo.ipynb` — 业务 Skill 主链  
5. 其余 advanced 案例按 [advanced/README.md](advanced/README.md)

## 一键验证

```bash
# 从仓库根目录
python scripts/smoke_docs_examples.py
python scripts/run_advanced_evidence.py --only 01_iris,02_titanic,04_breast_cancer
```

跑通证据汇总：`docs/evidence/ADVANCED_EVIDENCE.md`。
