"""一次性生成 12 个进阶演示 Notebook 骨架（可重复运行覆盖）。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CASES = [
    {
        "id": "E01",
        "slug": "01_iris",
        "title": "E01 Iris：小样本分层验证与标签提交",
        "dataset": "sklearn iris",
        "task": "多分类",
        "hard": [
            "小样本下分层交叉验证的方差很大",
            "字符串标签 vs 整数标签的提交格式差异",
            "用 wanglaoshi CompetitionTools 生成并校验 submission",
        ],
        "load": """
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame.copy()
df['target_name'] = data.target_names[data.target]
print(df.shape)
df.head()
""",
    },
    {
        "id": "E02",
        "slug": "02_titanic",
        "title": "E02 Titanic：缺失、文本特征与训练验证隔离",
        "dataset": "seaborn titanic（或本地 CSV）",
        "task": "二分类",
        "hard": [
            "缺失值策略必须只在训练集拟合",
            "姓名/舱位等文本衍生特征",
            "泄漏：用全量统计填充会抬高线下分",
        ],
        "load": """
import seaborn as sns
df = sns.load_dataset('titanic')
print(df.shape)
df.head()
""",
    },
    {
        "id": "E03",
        "slug": "03_wine_quality",
        "title": "E03 Wine Quality：回归/有序分类与不均衡",
        "dataset": "UCI Wine Quality（注意：不是 sklearn load_wine）",
        "task": "回归或有序分类",
        "hard": [
            "quality 是有序整数，可当回归也可当分类",
            "类别不均衡：极端分数样本很少",
            "红/白酒是否分开建模",
        ],
        "load": """
# Wine Quality 需从 UCI / OpenML 下载；此处先给占位流程
# 推荐：pip install ucimlrepo 或手动下载 winequality-red.csv
import pandas as pd
from pathlib import Path

path = Path('../../data/winequality-red.csv')
if path.exists():
    df = pd.read_csv(path, sep=';')
    print(df.shape)
    display(df.head())
else:
    print('请先下载 Wine Quality 到 data/winequality-red.csv')
    print('参考：https://archive.ics.uci.edu/dataset/186/wine+quality')
""",
    },
    {
        "id": "E04",
        "slug": "04_breast_cancer",
        "title": "E04 Breast Cancer：缩放/PCA 与召回权衡",
        "dataset": "sklearn breast_cancer",
        "task": "二分类（医疗场景）",
        "hard": [
            "特征量纲差异大，需要缩放",
            "PCA 降维后可解释性下降",
            "医疗场景更关注召回率/漏诊成本",
        ],
        "load": """
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.shape)
df.head()
""",
    },
    {
        "id": "E05",
        "slug": "05_credit_card_fraud",
        "title": "E05 Credit Card Fraud：时间划分、不均衡与 PR-AUC",
        "dataset": "Kaggle Credit Card Fraud（需本地下载）",
        "task": "极端不均衡二分类",
        "hard": [
            "正类极稀，Accuracy 几乎无意义",
            "应按时间切分，避免未来信息",
            "关注 PR-AUC / Recall@Precision",
        ],
        "load": """
from pathlib import Path
import pandas as pd

path = Path('../../data/creditcard.csv')
if path.exists():
    df = pd.read_csv(path)
    print(df.shape, df['Class'].value_counts(normalize=True))
    df.head()
else:
    print('请从 Kaggle 下载 creditcard.csv 到 data/')
""",
    },
    {
        "id": "E06",
        "slug": "06_california_housing",
        "title": "E06 California Housing：空间分组 vs 随机划分",
        "dataset": "sklearn fetch_california_housing",
        "task": "回归",
        "hard": [
            "地理邻近样本相关，随机划分会泄漏空间信息",
            "用 GroupKFold / 按区域划分对比",
            "分层误差：不同房价区间误差不同",
        ],
        "load": """
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.shape)
df.head()
""",
    },
    {
        "id": "E07",
        "slug": "07_abalone",
        "title": "E07 Abalone：混合编码、目标变换与残差",
        "dataset": "OpenML / UCI Abalone",
        "task": "回归（年龄 rings）",
        "hard": [
            "Sex 类别编码方式影响很大",
            "目标可做 log1p 变换再还原",
            "残差分析看是否系统偏差",
        ],
        "load": """
from pathlib import Path
import pandas as pd

path = Path('../../data/abalone.csv')
if path.exists():
    df = pd.read_csv(path)
    print(df.shape)
    df.head()
else:
    print('请下载 Abalone 到 data/abalone.csv')
""",
    },
    {
        "id": "E08",
        "slug": "08_bike_sharing",
        "title": "E08 Bike Sharing：时间验证与可用滞后特征",
        "dataset": "UCI Bike Sharing",
        "task": "时间序列回归",
        "hard": [
            "必须用时间切分，不能随机打乱",
            "滞后特征只能用预测时刻已知信息",
            "常用 RMSLE 作为指标",
        ],
        "load": """
from pathlib import Path
import pandas as pd

path = Path('../../data/bike_sharing_hour.csv')
if path.exists():
    df = pd.read_csv(path)
    print(df.shape)
    df.head()
else:
    print('请下载 Bike Sharing hour.csv 到 data/bike_sharing_hour.csv')
""",
    },
    {
        "id": "E09",
        "slug": "09_mall_customers",
        "title": "E09 Mall Customers：缩放、聚类数与稳定性",
        "dataset": "Mall Customers（常见教学 CSV）",
        "task": "聚类 / 用户画像",
        "hard": [
            "聚类前必须缩放",
            "k 的选择：肘部法 + silhouette",
            "多次随机种子检查簇稳定性",
        ],
        "load": """
from pathlib import Path
import pandas as pd

path = Path('../../data/Mall_Customers.csv')
if path.exists():
    df = pd.read_csv(path)
    print(df.shape)
    df.head()
else:
    print('请准备 Mall_Customers.csv 到 data/')
""",
    },
    {
        "id": "E10",
        "slug": "10_adult_census",
        "title": "E10 Adult Census：折内目标编码与分组表现",
        "dataset": "OpenML Adult / Census Income",
        "task": "二分类",
        "hard": [
            "目标编码必须在折内拟合，否则泄漏",
            "不同人群（性别/种族）表现差异需单独评估",
            "高基数类别处理",
        ],
        "load": """
from pathlib import Path
import pandas as pd

path = Path('../../data/adult.csv')
if path.exists():
    df = pd.read_csv(path)
    print(df.shape)
    df.head()
else:
    print('请下载 Adult 数据集到 data/adult.csv')
""",
    },
    {
        "id": "E11",
        "slug": "11_house_prices",
        "title": "E11 House Prices：高维混合、缺失含义与集成",
        "dataset": "Kaggle House Prices",
        "task": "回归",
        "hard": [
            "缺失本身可能有业务含义（没有车库）",
            "目标常用 log1p",
            "树模型 + 线性模型 stacking",
        ],
        "load": """
from pathlib import Path
import pandas as pd

train_path = Path('../../data/house_prices_train.csv')
if train_path.exists():
    train = pd.read_csv(train_path)
    print(train.shape)
    train.head()
else:
    print('请从 Kaggle House Prices 下载 train.csv 到 data/house_prices_train.csv')
""",
    },
    {
        "id": "E12",
        "slug": "12_digit_recognizer",
        "title": "E12 Digit Recognizer：高维 EDA、PCA 与错分分析",
        "dataset": "sklearn digits 或 Kaggle Digit Recognizer",
        "task": "多分类（图像像素）",
        "hard": [
            "784 维像素的 EDA 方式不同",
            "PCA 可视化与压缩",
            "混淆矩阵定位易混数字对",
        ],
        "load": """
from sklearn.datasets import load_digits
import pandas as pd

digits = load_digits()
X = pd.DataFrame(digits.data)
y = pd.Series(digits.target, name='label')
print(X.shape, y.value_counts().sort_index().to_dict())
""",
    },
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip("\n").split("\n")]}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def build_notebook(case: dict) -> dict:
    hard_list = "\n".join(f"- {h}" for h in case["hard"])
    cells = [
        md(
            f"""# {case['title']}

> 对应路线图任务：**{case['id']}**（Issue #9）

## 目标
- 数据集：`{case['dataset']}`
- 任务类型：`{case['task']}`
- 用 `wanglaoshi` 完成：环境检查 → EDA → 特征/划分注意点 → 基线 →（如适用）提交生成与校验

## 本案例技术难点
{hard_list}

## 运行模式
- `QUICK=True`：小样本/少折/少模型，适合课堂演示
- `QUICK=False`：更完整对比（耗时更长）
"""
        ),
        code(
            """
# ===== 全局配置 =====
QUICK = True
RANDOM_STATE = 42

from wanglaoshi.runtime import ready
from wanglaoshi.runtime import VERSIONS as V

ready(seed=RANDOM_STATE)  # 关警告 + 环境信息；勿再叠 filterwarnings / JE.no_warning
V.check_all_versions(version='simple', show_latest=False)
"""
        ),
        md("## 1. 数据加载"),
        code(case["load"]),
        md(
            """## 2. 快速 EDA（wanglaoshi.Analyzer）

先用 Analyzer 看缺失、分布、相关性；大数据集请先抽样。"""
        ),
        code(
            """
from wanglaoshi.analyze import DataAnalyzer

# 若上面成功得到 df，则继续；否则请先解决数据路径问题
try:
    eda_df = df.copy()
except NameError:
    raise RuntimeError('请先在上一格成功加载为变量 df')

# 过大时抽样，避免 Notebook 卡死
if QUICK and len(eda_df) > 5000:
    eda_df = eda_df.sample(5000, random_state=RANDOM_STATE)

analyzer = DataAnalyzer(eda_df)
print(analyzer.get_basic_info())
display(analyzer.missing_value_analysis().head(20))
display(analyzer.basic_statistics().head(20))
"""
        ),
        md(
            """## 3. 训练/验证划分（本案例最容易踩坑的地方）

请根据案例类型选择：
- 分类：`StratifiedShuffleSplit` / `StratifiedKFold`
- 时间：按时间切，禁止随机打乱
- 空间/分组：`GroupKFold`
- 不均衡：关注 PR 曲线，不只看 Accuracy
"""
        ),
        code(
            """
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# ---- 请按数据集修改 target_col / feature_cols ----
# 下面给一个通用探测：优先常见目标列名

candidates = ['target', 'target_name', 'Survived', 'survived', 'Class', 'quality',
              'MedHouseVal', 'Rings', 'cnt', 'income', 'SalePrice', 'label']
target_col = None
for c in candidates:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    # digits 案例可能把 y 分开了
    if 'y' in globals():
        target_col = None
        print('检测到独立 y，将使用 X/y 变量')
    else:
        raise RuntimeError(f'请手动指定 target_col。当前列: {list(df.columns)}')

if target_col is not None:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df[target_col].copy()
    print('target =', target_col, 'numeric features =', len(X.columns))
else:
    X = X.copy()
    y = y.copy()

# 简单划分（时间/分组案例请在后面覆盖）
stratify = y if (y.dtype == 'object' or y.nunique() < 20) else None
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
)
print(X_train.shape, X_valid.shape)
"""
        ),
        md("## 4. 快速基线（wanglaoshi.CompetitionTools.QuickBaseline）"),
        code(
            """
from wanglaoshi import compete as CT

# 仅数值特征的快速基线；含类别特征的案例请先做编码（见 FeatureEngineering）
baseline = CT.QuickBaseline()

# Quick 模式：只用部分模型会更快——这里仍跑全量小模型，但数据已可能抽样
results = baseline.run_all_models(
    X_train=X_train.fillna(X_train.median(numeric_only=True)),
    y_train=y_train,
    X_test=X_valid.fillna(X_train.median(numeric_only=True)),
    y_test=y_valid,
    task_type='auto',
)
comparison = baseline.compare_models(results, top_n=5)
comparison
"""
        ),
        md(
            """## 5. 特征工程尝试（可选增强）

使用 `FeatureEngineering.AdvancedFeatureEngineer`：时间特征、目标编码、变换等。  
**提醒：目标编码 / 缺失填充统计必须只在训练集拟合。**
"""
        ),
        code(
            """
from wanglaoshi import features as FE

engineer = FE.AdvancedFeatureEngineer()
print('FeatureEngineering 已加载。请按本案例难点选择：')
print('- extract_datetime_features')
print('- target_encode / frequency_encode / one_hot_encode')
print('- log_transform / bin_features')
print('- select_features_by_importance')
"""
        ),
        md(
            """## 6. 提交文件生成与校验（若为竞赛格式）

对 Iris / Titanic / Digits 等可本地模拟 sample_submission。
"""
        ),
        code(
            f"""
from pathlib import Path
import pandas as pd
import numpy as np
from wanglaoshi import compete as CT

# 用验证集预测模拟“测试提交”
best_name = comparison.iloc[0]['Model']
best_model = results[best_name]['model']
pred = results[best_name].get('predictions')
if pred is None:
    pred = best_model.predict(X_valid.fillna(X_train.median(numeric_only=True)))

out_dir = Path('outputs') / '{case["slug"]}'
out_dir.mkdir(parents=True, exist_ok=True)

# 构造样本提交：兼容数值标签与字符串标签
sample = pd.DataFrame({{
    'Id': np.arange(len(pred)),
    'Target': y_valid.to_numpy()[:len(pred)] if hasattr(y_valid, 'to_numpy') else list(y_valid)[:len(pred)]
}})
sample_path = out_dir / 'sample_submission.csv'
sub_path = out_dir / 'submission.csv'
sample.to_csv(sample_path, index=False)

gen = CT.SubmissionGenerator()
sub = gen.create_submission(
    predictions=pred,
    sample_submission_path=str(sample_path),
    output_path=str(sub_path),
    prediction_type='label',
)
val = gen.validate_submission(str(sub_path), str(sample_path))
print(val)
sub.head()
"""
        ),
        md(
            """## 7. 误差 / 错分分析与局限

请记录：
1. 哪个模型最好？为什么？
2. 主要错误来自哪一类样本？
3. 本 Notebook 的简化假设（抽样、只用数值特征等）会如何影响结论？
4. 若上线/提交，还缺哪些验证（时间切分、分组公平性、阈值调优）？
"""
        ),
        code(
            """
# 简单错误分析模板（分类时）
from sklearn.metrics import classification_report, confusion_matrix

if str(comparison.columns[1]).lower().find('accuracy') >= 0 or 'Test Accuracy' in comparison.columns:
    y_hat = results[best_name]['predictions']
    print(classification_report(y_valid, y_hat))
    print('confusion matrix:\\n', confusion_matrix(y_valid, y_hat))
else:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_hat = results[best_name]['predictions']
    print('MAE', mean_absolute_error(y_valid, y_hat))
    print('RMSE', mean_squared_error(y_valid, y_hat, squared=False))
"""
        ),
        md(
            f"""## 8. 检查清单（{case['id']}）

- [ ] 数据加载成功，并记录来源/许可
- [ ] EDA 看过缺失与目标分布
- [ ] 划分方式符合任务（随机/分层/时间/分组）
- [ ] 基线对比已跑通
- [ ] 提交生成与校验通过（如适用）
- [ ] 写下至少 2 条可改进方向

---
下一步：打开下一个 Notebook，或回到 `examples/advanced/README.md` 查看学习顺序。
"""
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    for case in CASES:
        path = ROOT / f"{case['slug']}.ipynb"
        nb = build_notebook(case)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
