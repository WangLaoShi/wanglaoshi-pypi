# 进阶案例总览

每课 = 仓库里的真实 Notebook + 课下练习。  
源码目录：`examples/advanced/`（构建时复制进 `book/cases/`）。

## 和 `wanglaoshi` 的强制对应

不要把案例当成「纯 sklearn 演示」。下表是 **本课至少要碰到的包 API**（练习会考）：

| 课 | Notebook | 必用 / 必懂的 wanglaoshi API | 练的纪律 |
| --- | --- | --- | --- |
| 01 Iris | `01_iris.ipynb` | `ready`, `DataAnalyzer`, `QuickBaseline`, `SubmissionGenerator` | 分层划分；标签提交契约 |
| 02 Titanic | `02_titanic.ipynb` | `DataAnalyzer`, `AdvancedFeatureEngineer`, `LeakageDetector` | 缺失填充只在训练折 |
| 04 Breast Cancer | `04_breast_cancer.ipynb` | `QuickBaseline`, `FeatureImportanceAnalyzer` | 缩放进 Pipeline；召回权衡 |
| 12 Digits | `12_digit_recognizer.ipynb` | `DataAnalyzer`, `SubmissionGenerator` | 高维 EDA；错分分析 |
| 06 California | `06_california_housing.ipynb` | `CrossValidator.group_cv` / 分组思想 | 同区块不进两侧 |
| 03 Wine Quality | `03_wine_quality.ipynb` | `QuickBaseline`（回归） | 有序/不均衡指标 |
| 07 Abalone | `07_abalone.ipynb` | `one_hot_encode` / 目标变换 | 混合类型 |
| 08 Bike | `08_bike_sharing.ipynb` | `extract_datetime_features`, `detect_time_leakage` | 滞后不得见未来 |
| 09 Mall | `09_mall_customers.ipynb` | EDA + 稳定性讨论 | 聚类不是提交赛 |
| 10 Adult | `10_adult_census.ipynb` | `target_encode`（折内思想） | 目标编码防泄露 |
| 05 Fraud | `05_credit_card_fraud.ipynb` | 指标选择 + `QuickBaseline` | 别只看 Accuracy |
| 11 House | `11_house_prices.ipynb` | `ModelEnsemble`, `SubmissionGenerator` | 集成与校验 |

业务专章另计：{doc}`business_pipeline` → `BusinessPipeline` 全阶段。

## 怎么学才不算「太简单」

1. **先读精讲** {doc}`../lessons/overview`（泄露 / 划分 / 提交 / 折内拟合）  
2. 打开 Notebook，每碰到一个 `wanglaoshi` 调用，对照 {doc}`../reference/api`  
3. 合上 Notebook 做练习：要写出**方法名**和**一段可运行最小代码**  
4. 故意写坏一次（打乱 ID、全表先标准化再划分），确认校验器或分数变化  

## 数据可得性

- **可直接跑（sklearn / seaborn）**：Iris、Titanic、Breast Cancer、California、Digits  
- **可能需下载到 `data/`**：Wine、Fraud、Abalone、Bike、Mall、Adult、House（Notebook 内常有 DEMO 回退）  

## 本地打开

```bash
pip install -e ".[jupyter]"
jupyter lab examples/advanced/
# 发版检查（含案例证据）：
./scripts/release_check.sh full
```

## 批量证据

{doc}`../reference/evidence`
