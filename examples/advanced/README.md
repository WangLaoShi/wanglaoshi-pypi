# 进阶演示（12 个案例）

对应 GitHub Issue：[#9](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/9)  
总跟踪：[#6](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/6)  
上级索引：[examples/README.md](../README.md)

本目录用于在 Jupyter 中验证 `wanglaoshi` 的 EDA、特征工程、基线模型与提交生成能力。  
这些 Notebook 同时是 **基线工坊**（`book/`，标题：基线工坊 · 从 EDA 到提交）的正文来源；构建：`python scripts/build_book.py`。
**演示跑通 ≠ 单元测试通过**；回归测试见仓库根目录 `tests/`。  
批量证据：`docs/evidence/ADVANCED_EVIDENCE.md`（最近一次 **12/12 passed**）。

## 学习顺序（建议）

| 顺序 | Notebook | 数据集 | 核心难点 |
| --- | --- | --- | --- |
| 1 | `01_iris.ipynb` | Iris | 小样本分层、标签提交 |
| 2 | `02_titanic.ipynb` | Titanic | 缺失/文本、训练验证隔离 |
| 3 | `04_breast_cancer.ipynb` | Breast Cancer | 缩放/PCA、召回权衡 |
| 4 | `12_digit_recognizer.ipynb` | Digits | 高维 EDA、错分分析 |
| 5 | `06_california_housing.ipynb` | California Housing | 空间分组 vs 随机划分 |
| 6 | `03_wine_quality.ipynb` | Wine Quality | 回归/有序分类、不均衡 |
| 7 | `07_abalone.ipynb` | Abalone | 混合编码、目标变换 |
| 8 | `08_bike_sharing.ipynb` | Bike Sharing | 时间验证、滞后特征 |
| 9 | `09_mall_customers.ipynb` | Mall Customers | 聚类稳定性 |
| 10 | `10_adult_census.ipynb` | Adult | 折内目标编码 |
| 11 | `05_credit_card_fraud.ipynb` | Credit Card Fraud | 极端不均衡、PR-AUC |
| 12 | `11_house_prices.ipynb` | House Prices | 高维混合、集成 |

## 环境准备

依赖分层详见 [docs/DEPENDENCIES.md](../../docs/DEPENDENCIES.md)。

```bash
cd /path/to/wanglaoshi-pypi
python -m pip install -e ".[jupyter]"
# 或：pip install -r requirements/examples.txt && pip install -e .
jupyter notebook examples/advanced/
```

可选深度学习请自行安装 `torch`（见依赖文档），不影响本目录大多数案例。

## 数据说明

- **可直接跑（sklearn / seaborn）**：E01 Iris、E02 Titanic、E04 Breast Cancer、E06 California Housing、E12 Digits
- **需本地下载到 `data/`**：E03 Wine Quality、E05 Fraud、E07 Abalone、E08 Bike、E09 Mall、E10 Adult、E11 House Prices

请遵守各数据集许可与竞赛条款；Notebook 不会自动绕过登录或上传提交。

## 每个 Notebook 课程结构（已加厚至约 3×）

正统数据分析叙事（问题 → 质量 → **协议** → 基线 → 诊断 → 交付），并绑定 `wanglaoshi` API：

1. 问题形式化与经典文献/教材背景（含外链）
2. `ready()` 环境就绪
3. 数据加载（本地 `data/` 或 DEMO 回退）
4. 理论驱动 EDA（`DataAnalyzer` + 关键可视化）
5. **本课专属验证协议**（分层 / 时间 / 空间分组）及错误对照
6. 特征工程纪律（折内 fit；`FeatureEngineering`）
7. `QuickBaseline` / 专项模型与指标选择理论
8. 误差分析、阈值/代价、提交契约（`SubmissionGenerator`）
9. 附录：深度阅读、书面作业、API 地图

重新生成加厚版：

```bash
python examples/advanced/_deepen_curriculum_batch1.py
python examples/advanced/_deepen_curriculum_batch2.py
python examples/advanced/_deepen_curriculum_batch3.py
# 旧版通用加深（较薄）：
# python examples/advanced/_deepen_all.py
```

## 跑通证据（E17）

```bash
python scripts/run_advanced_evidence.py
```

汇总见 `docs/evidence/ADVANCED_EVIDENCE.md`（JSON：`docs/evidence/advanced_evidence.json`）。

## 重新生成

- **详细版（推荐，案例专属）**：

```bash
python examples/advanced/_deepen_all.py
```

- **通用骨架（较薄，仅作模板）**：

```bash
python examples/advanced/_generate_notebooks.py
```

注意：两个脚本都会覆盖同名 `.ipynb`。个性化修改请先备份。
