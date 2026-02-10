# 王老师 WangLaoShi

## 项目介绍

`wanglaoshi` 是一个面向 **数据分析入门学生** 和 **数据科学竞赛选手（如 Kaggle）** 的学习与实战工具包，主要特点：

- **一键EDA**：`Analyzer` 模块可以对数据集做全面的探索性分析（缺失值、异常值、相关性、PCA、报告生成等）
- **比赛工具链**：`CompetitionTools` 模块提供提交文件生成、快速基线模型、模型集成、超参数搜索、实验追踪等功能
- **特征工程增强**：`FeatureEngineering` 模块支持时间特征、文本特征、编码、数值变换、特征选择等
- **环境辅助**：`VERSIONS`、`JupyterEnv`、`JupyterFont`、`Useful` 等模块帮助检查环境、管理内核、配置字体和常用工具

适用人群：

- 正在学习 Python 数据分析 / 机器学习的同学
- 想要参加 Kaggle 等比赛、需要一套快速 EDA + 基线建模工具的选手
- 想要复用一套「可解释、带注释」代码模板做课程作业或项目 Demo 的同学

## 项目结构（简要）

```text
wanglaoshi-pypi
├── README.md                     # 项目文档（本文件）
├── setup.py                      # 安装配置
├── wanglaoshi/
│   ├── Analyzer.py               # 高级 EDA 与报告生成（HTML + 图表 + 解释）
│   ├── Analyzer_Plain.py         # 简化版文本 EDA（输出到 txt + 基本图表）
│   ├── CompetitionTools.py       # 比赛工具：提交文件生成、基线模型、集成、调参、实验追踪
│   ├── FeatureEngineering.py     # 增强特征工程：时间/文本/编码/变换/特征选择
│   ├── VERSIONS.py               # 环境与库版本检查工具（简单/完整库表，支持 PyPI 最新版本比对）
│   ├── JupyterEnv.py             # Jupyter 内核管理与环境信息
│   ├── JupyterFont.py            # Matplotlib 中文字体自动下载与配置
│   ├── Useful.py                 # 常用工具函数（pip 源、Jupyter 样式等）
│   ├── WebGetter.py              # 通用文件下载工具（带进度条）
│   ├── MLDL.py                   # 机器学习 + 深度学习基础封装（预处理、特征工程、ML/DL 模型）
│   ├── utils.py                  # 小工具（对齐、中文检测等）
│   ├── __init__.py               # 包导出入口（根据版本逐步完善）
│   └── ...（static/templates 等，用于报告渲染）
```

## 项目版本

- 0.0.1 初始化版本，项目开始
- 0.0.2 增加列表输出
- 0.0.3 增加字典输出,使用 Rich 输出
- 0.0.4 实现 JupyterNotebook 环境创建
- 0.0.5 增加几个有用的库
- 0.0.6 修改获取 version 的方法
- 0.0.7 增加获取当前安装包的版本号，增加获取当前每一个安装包最新版本的方法
- 0.0.8 增加对数据文件的基本分析的部分
- 0.0.9 增加 jinja2 的模板输出的 Analyzer
- 0.10.0 增加 no_waring,字体获取，安装字体
- 0.10.6 增加 Analyzer 的使用部分(需要 statsmodels)
- 0.10.7 增加 MLDL 部分(需要 sklearn,torch)
- 0.10.10 增加分析结果 Render notebook 部分
- 0.10.13 修复分析结果
- 0.11.1 增加 static 和 template 修改 html 报告生成
- 0.11.2 添加 js 的引用
- 0.12.0 【阶段一：比赛核心功能】新增 CompetitionTools 模块，包含提交文件生成器、快速基线模型、数据泄露检测三大核心功能，专为数据科学比赛优化
- 0.13.0 【阶段二：增强功能】新增 FeatureEngineering 模块（时间特征、文本特征、目标编码、数值变换、特征选择），CompetitionTools 增强（模型集成、特征重要性分析、交叉验证增强、超参数优化、实验追踪）

## 安装方式

### 1. 源码安装方式

- 检出项目
- 进入项目目录
- 执行`python setup.py install`
- 安装完成

### 2. pip安装方式

```shell
pip install wanglaoshi
```

## 使用方法

下面给出几个典型使用场景，你可以按需选用。

### 0. 典型比赛工作流示例（从 0 到 baseline）

```python
import pandas as pd
from wanglaoshi import VERSIONS as V
from wanglaoshi import Analyzer
from wanglaoshi import CompetitionTools as CT
from wanglaoshi import FeatureEngineering as FE

# 1）检查环境和常用库
V.show_env_info()
V.check_all_versions(version='simple', show_latest=True, only_problems=True)

# 2）读取数据并做快速 EDA
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
analyzer = Analyzer.DataAnalyzer(train)
analyzer.generate_report('train_eda_report.html')  # 生成详细 HTML 报告

# 3）特征工程示例
engineer = FE.AdvancedFeatureEngineer()
if 'date' in train.columns:
    train = engineer.extract_datetime_features(train, 'date')
    test = engineer.extract_datetime_features(test, 'date')

# 假设 target 是监督任务的目标列
target_col = 'target'
feature_cols = [c for c in train.columns if c != target_col]
X_train = train[feature_cols]
y_train = train[target_col]

# 4）快速基线模型
baseline = CT.QuickBaseline()
results = baseline.run_all_models(X_train, y_train, task_type='auto')
baseline.compare_models(results, top_n=5)

# 5）选择一个模型，生成预测和提交文件（示例）
best_model_name = list(results.keys())[0]
best_model = results[best_model_name]['model']
best_model.fit(X_train, y_train)
X_test = test[feature_cols]
y_pred = best_model.predict(X_test)

sub_gen = CT.SubmissionGenerator()
sub_gen.create_submission(
    predictions=y_pred,
    sample_submission_path='sample_submission.csv',
    output_path='submission.csv'
)
```

> 上面的代码展示了：**环境检查 → EDA 报告 → 特征工程 → 快速基线 → 生成 submission** 的一条完整路径，非常适合作为 Kaggle 比赛的起步模板。

---

### 1. 创建新的环境

```python
from wanglaoshi import JupyterEnv as JE
JE.jupyter_kernel_list()
JE.install_kernel()
# 按照提示输入环境名称
```

### 2. 获取当前环境常用库版本

VERSIONS 模块提供了两个版本的库列表：**简单版本**（9个核心库）和**复杂版本**（60+个库）。

#### 2.1 显示简单版本（推荐初学者）

简单版本只包含最核心、最常用的库，适合快速检查基础环境：

```python
from wanglaoshi import VERSIONS as V

# 显示简单版本（只包含9个核心库）
V.check_all_versions(version='simple')
```

简单版本包含的库：

- **数据处理**：numpy, pandas
- **数据可视化**：matplotlib, seaborn
- **机器学习**：scikit-learn
- **深度学习**：tensorflow, pytorch
- **科学计算**：scipy
- **工具库**：tqdm

#### 2.2 显示复杂版本（默认）

复杂版本包含所有重要的机器学习和深度学习库，适合完整环境检查：

```python
from wanglaoshi import VERSIONS as V

# 显示复杂版本（包含60+个库，默认）
V.check_all_versions()
# 或者显式指定
V.check_all_versions(version='full')
```

复杂版本包含的库类别：

- 数据处理（numpy, pandas, polars, scipy, statsmodels）
- 数据可视化（matplotlib, seaborn, plotly, bokeh）
- 机器学习（scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn）
- 深度学习（tensorflow, keras, pytorch, torchvision, torchaudio, onnx, onnxruntime）
- 自然语言处理（nltk, spacy, transformers, sentence-transformers）
- 计算机视觉（opencv-python, Pillow, scikit-image, albumentations, imageio）
- 强化学习（gym, gymnasium, stable-baselines3）
- 分布式计算（ray, dask, joblib）
- 机器学习管理（mlflow, wandb, optuna, hydra, pycaret, tensorboard）
- 音频处理（librosa, mir_eval, soundfile）
- Web应用（streamlit, fastapi, gradio）
- 开发工具（jupyter, ipython, ipywidgets）
- 工具库（tqdm, requests, python-dotenv）

#### 2.3 显示所有列（包括网站和分类）

```python
from wanglaoshi import VERSIONS as V

# 显示所有列（包括网站链接和分类信息）
V.check_all_versions(all_columns=True, version='full')
```

#### 2.4 其他版本检查功能

```python
from wanglaoshi import VERSIONS as V

# 获取所有已安装的库（不限于ML/DL库）
V.check_all_installed()

# 对比已安装版本和最新版本（需要网络连接）
V.check_all_installed_with_latest()

# 检查指定库的版本（返回字典）
versions = V.check_versions(version='simple')
print(versions['numpy'])  # 输出: '1.21.0' 或 'Not installed'
```

### 5. 得到一个数据文件的基本的分析页面

#### 示例调用

```python
"""
DataAnalyzer 使用示例
这个示例展示了如何使用 DataAnalyzer 进行数据分析并生成报告
"""

import pandas as pd
import numpy as np
from wanglaoshi.Analyzer import DataAnalyzer
import os

def create_sample_data():
    """创建示例数据"""
    # 创建随机数据
    np.random.seed(42)
    n_samples = 1000
    
    # 数值型数据
    data = {
        'age': np.random.normal(35, 10, n_samples),  # 年龄
        'income': np.random.lognormal(10, 1, n_samples),  # 收入
        'height': np.random.normal(170, 10, n_samples),  # 身高
        'weight': np.random.normal(65, 15, n_samples),  # 体重
        'satisfaction': np.random.randint(1, 6, n_samples),  # 满意度评分
    }
    
    # 添加一些缺失值
    for col in ['age', 'income', 'height', 'weight']:
        mask = np.random.random(n_samples) < 0.05  # 5%的缺失值
        data[col][mask] = np.nan
    
    # 添加一些异常值
    data['income'][np.random.choice(n_samples, 5)] = data['income'].max() * 2
    
    # 创建分类数据
    data['gender'] = np.random.choice(['男', '女'], n_samples)
    data['education'] = np.random.choice(['高中', '本科', '硕士', '博士'], n_samples)
    data['occupation'] = np.random.choice(['工程师', '教师', '医生', '销售', '其他'], n_samples)
    
    # 创建时间数据
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    data['date'] = dates
    
    return pd.DataFrame(data)

def basic_analysis_demo():
    """基础分析示例"""
    print("=== 基础分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    print("\n数据预览:")
    print(df.head())
    
    # 创建分析器实例
    analyzer = DataAnalyzer(df)
    
    # 基本统计分析
    print("\n基本统计信息:")
    basic_stats = analyzer.basic_statistics()
    print(basic_stats)
    
    # 正态性检验
    print("\n正态性检验结果:")
    normality_test = analyzer.normality_test()
    print(normality_test)
    
    # 缺失值分析
    print("\n缺失值分析:")
    missing_analysis = analyzer.missing_value_analysis()
    print(missing_analysis)
    
    # 异常值分析
    print("\n异常值分析:")
    outlier_analysis = analyzer.outlier_analysis()
    print(outlier_analysis)
    
    # 重复值分析
    print("\n重复值分析:")
    duplicate_analysis = analyzer.duplicate_analysis()
    print(duplicate_analysis)

def advanced_analysis_demo():
    """高级分析示例"""
    print("\n=== 高级分析示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 相关性分析
    print("\n相关性分析:")
    correlation_matrix = analyzer.correlation_analysis()
    print(correlation_matrix)
    
    # 多重共线性分析
    print("\n多重共线性分析:")
    multicollinearity = analyzer.multicollinearity_analysis()
    print(multicollinearity)
    
    # 主成分分析
    print("\n主成分分析:")
    pca_analysis = analyzer.pca_analysis()
    print(pca_analysis)

def visualization_demo():
    """可视化示例"""
    print("\n=== 可视化示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 分布图
    print("\n生成分布图...")
    for column in ['age', 'income', 'height', 'weight']:
        print(f"\n{column} 的分布图:")
        img_base64 = analyzer.plot_distribution(column)
        print(f"图片已生成，base64长度: {len(img_base64)}")
    
    # 相关性热图
    print("\n生成相关性热图...")
    heatmap_base64 = analyzer.plot_correlation_heatmap()
    print(f"热图已生成，base64长度: {len(heatmap_base64)}")

def report_generation_demo():
    """报告生成示例"""
    print("\n=== 报告生成示例 ===")
    
    # 创建示例数据
    df = create_sample_data()
    analyzer = DataAnalyzer(df)
    
    # 生成HTML报告
    print("\n生成分析报告...")
    analyzer.generate_report("analysis_report.html")
    print("报告已生成: analysis_report.html")

def file_analysis_demo():
    """文件分析示例"""
    print("\n=== 文件分析示例 ===")
    
    # 创建示例数据文件
    print("\n创建示例数据文件...")
    data_dir = "example_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建CSV文件
    df1 = create_sample_data()
    df1.to_csv(os.path.join(data_dir, "sample_data1.csv"), index=False)
    
    # 创建另一个CSV文件，使用不同的数据
    df2 = create_sample_data()  # 使用不同的随机种子
    df2.to_csv(os.path.join(data_dir, "sample_data2.csv"), index=False)
    
    # 创建Excel文件
    df3 = create_sample_data()  # 使用不同的随机种子
    df3.to_excel(os.path.join(data_dir, "sample_data3.xlsx"), index=False)
    
    print(f"示例数据文件已创建在 {data_dir} 目录下")
    
    # 分析单个文件
    print("\n分析单个文件示例:")
    print("分析 sample_data1.csv...")
    from wanglaoshi import analyze_data
    analyze_data(
        os.path.join(data_dir, "sample_data1.csv"),
        "analysis_report_single.html"
    )
    print("单个文件分析报告已生成: analysis_report_single.html")
    
    # 分析多个文件
    print("\n分析多个文件示例:")
    print("分析目录下的所有数据文件...")
    from wanglaoshi import analyze_multiple_files
    analyze_multiple_files(data_dir, "reports")
    print("多个文件的分析报告已生成在 reports 目录下")
    
    # 清理示例文件
    print("\n清理示例文件...")
    import shutil
    shutil.rmtree(data_dir)
    print("示例数据文件已清理")

def notebook_demo():
    """Jupyter Notebook示例"""
    print("\n=== Jupyter Notebook示例 ===")
    print("""
    在Jupyter Notebook中使用:
    
    ```python
    import pandas as pd
    from wanglaoshi import DataAnalyzer
    
    # 创建或加载数据
    df = pd.DataFrame(...)
    
    # 创建分析器实例
    analyzer = DataAnalyzer(df)
    
    # 在notebook中显示分析报告
    analyzer.analyze_notebook()
    ```
    """)

def main():
    """主函数"""
    print("DataAnalyzer 使用示例\n")
    
    # 运行各个示例
    basic_analysis_demo()
    advanced_analysis_demo()
    visualization_demo()
    report_generation_demo()
    file_analysis_demo()
    notebook_demo()

if __name__ == "__main__":
    main() 
```

#### 数据分析示例

`analyzer_demo.py` 提供了完整的数据分析示例，包含以下功能：

1. **基础分析示例**：
   - 数据预览和基本统计信息
   - 正态性检验
   - 缺失值分析
   - 异常值分析
   - 重复值分析

2. **高级分析示例**：
   - 相关性分析：计算变量间的相关系数矩阵
   - 多重共线性分析：使用VIF检测变量间的多重共线性
   - 主成分分析：降维和特征提取

3. **可视化示例**：
   - 分布图：展示数值变量的分布情况
   - 相关性热图：直观展示变量间的相关关系

4. **报告生成示例**：
   - 生成HTML格式的分析报告
   - 包含所有分析结果和可视化图表

5. **文件分析示例**：
   - 支持分析单个数据文件
   - 支持批量分析多个数据文件
   - 支持CSV、Excel、JSON等多种格式

6. **Jupyter Notebook示例**：
   - 在Jupyter环境中使用DataAnalyzer
   - 交互式数据分析和可视化

### 6. 取消错误输出

```python
from wanglaoshi import JupyterEnv as JE
JE.no_warning()
```

### 7. Wget 功能

基本功能：

- 支持从 URL 下载文件
- 自动从 URL 提取文件名
- 支持指定输出目录和自定义文件名
- 显示下载进度条

使用方法：

```python
from WebGetter import Wget

# 创建下载器实例
downloader = Wget(
    url='https://example.com/file.zip',
    output_dir='./downloads',
    filename='custom_name.zip'
)

# 开始下载
downloader.download()
```

## 8. 字体安装

```python
# 这里用的是 SimHei 字体，可以根据自己的需要更改
from wanglaoshi import JupyterFont as JF
JF.matplotlib_font_init()
```

## 9. 批量数据分析（适合比赛）

```python
from wanglaoshi import Analyzer as A
import seaborn as sns
import pandas as pd

# 获取示例数据集
# 方法1：使用seaborn自带的数据集
tips = sns.load_dataset('tips')  # 餐厅小费数据集
tips.to_csv('tips.csv', index=False)

# 方法2：使用sklearn自带的数据集
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.to_csv('iris.csv', index=False)

# 创建测试文件夹
import os
os.makedirs('test_data', exist_ok=True)

# 将数据集移动到测试文件夹
import shutil
shutil.move('tips.csv', 'test_data/tips.csv')
shutil.move('iris.csv', 'test_data/iris.csv')

# 分析数据集
A.analyze_multiple_files('test_data', output_dir='reports')
```

批量分析功能特点：

- 支持多种数据格式（CSV、Excel、JSON）
- 自动生成每个数据文件的详细分析报告
- 异常值分析包含：
  - Z-score方法：识别极端和中度异常值
  - IQR方法：提供数据分布特征和异常值范围
  - 综合建议：基于两种方法的结果给出处理建议
- 报告包含可视化图表和详细的解释说明

分析完成后，您可以在 `reports` 目录下找到生成的分析报告：

- `tips_report.html`：餐厅小费数据集的分析报告
- `iris_report.html`：鸢尾花数据集的分析报告

## 10. MLDL (单独安装 torch，pip install torch)

```python
"""使用示例"""
from MLDL import *
# 1. 数据预处理
preprocessor = DataPreprocessor()
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', 'a', 'c', 'b']
})
df_processed = preprocessor.handle_missing_values(df, method='mean')
df_encoded = preprocessor.encode_categorical(df_processed, ['B'])

# 2. 特征工程
engineer = FeatureEngineer()
df_features = engineer.create_polynomial_features(df_encoded, ['A'], degree=2)

# 3. 机器学习模型
ml_model = MLModel('logistic')
X = df_features[['A', 'A_power_2']]
y = df_features['B']
ml_model.train(X, y)
metrics = ml_model.evaluate()
print("ML模型评估结果:", metrics)

# 4. 深度学习模型
dl_model = DLModel(input_size=2, hidden_size=4, output_size=3)
X_tensor = torch.FloatTensor(X.values)
y_tensor = torch.LongTensor(y.values)
dl_model.train(X_tensor, y_tensor, epochs=100)

# 5. 模型评估
evaluator = ModelEvaluator()
y_pred = ml_model.predict(X)
evaluator.plot_confusion_matrix(y, y_pred)
```

## 11. render notebook

```python
# 方法1：使用工具函数
from wanglaoshi import analyze_notebook
import pandas as pd

df = pd.read_csv('your_data.csv')
analyze_notebook(df)

# 方法2：使用类方法
from wanglaoshi import DataAnalyzer
import pandas as pd

df = pd.read_csv('your_data.csv')
analyzer = DataAnalyzer(df)
analyzer.analyze_notebook()
```

## 12. 比赛工具 (CompetitionTools) - 专为数据科学比赛设计

CompetitionTools 模块提供了专门用于数据科学比赛（如Kaggle）的核心工具，包括提交文件生成、快速基线模型和数据泄露检测。

### 12.1 提交文件生成器

自动生成符合比赛格式要求的提交文件，支持单模型提交和多模型融合。

```python
from wanglaoshi import CompetitionTools as CT
import pandas as pd
import numpy as np

# 创建生成器实例
generator = CT.SubmissionGenerator()

# 假设你已经有了预测结果
y_pred = model.predict(X_test)

# 生成提交文件
submission = generator.create_submission(
    predictions=y_pred,
    sample_submission_path='sample_submission.csv',
    output_path='my_submission.csv'
)

# 多模型融合提交（加权平均）
ensemble = generator.ensemble_submissions(
    submission_files=['sub1.csv', 'sub2.csv', 'sub3.csv'],
    weights=[0.4, 0.3, 0.3],  # 权重
    method='weighted_average',
    output_path='ensemble_submission.csv'
)

# 验证提交文件格式
validation = generator.validate_submission(
    submission_path='my_submission.csv',
    sample_submission_path='sample_submission.csv'
)
print(f"提交文件是否有效: {validation['is_valid']}")
```

**功能特点**：

- 自动识别ID列和目标列
- 支持单列和多列目标（多分类、多输出）
- 多种融合方法：加权平均、简单平均、排名平均、中位数
- 提交文件格式验证

### 12.2 快速基线模型

一键运行多个基础模型，快速获得baseline分数，适合比赛初期快速评估。

```python
from wanglaoshi import CompetitionTools as CT
import pandas as pd

# 创建快速基线工具
baseline = CT.QuickBaseline()

# 自动运行所有基础模型
results = baseline.run_all_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    task_type='auto'  # 自动检测分类或回归
)

# 对比模型性能（显示前5名）
comparison_df = baseline.compare_models(results, top_n=5)

# 获取最佳模型
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
```

**支持的模型**：

- **分类任务**：逻辑回归、随机森林、决策树、SVM、KNN、朴素贝叶斯
- **回归任务**：线性回归、随机森林、决策树、SVM、KNN

**功能特点**：

- 自动检测任务类型（分类/回归）
- 支持交叉验证和测试集评估
- 自动模型性能对比
- 返回训练好的模型供后续使用

### 12.3 数据泄露检测

检测常见的数据泄露问题，这是数据科学比赛中最容易犯的错误之一。

```python
from wanglaoshi import CompetitionTools as CT
import pandas as pd

# 创建泄露检测器
detector = CT.LeakageDetector()

# 检测目标泄露（特征中是否包含目标信息）
leakage_features = detector.detect_target_leakage(
    X=X_train,
    y=y_train,
    threshold=0.9,  # 相关性阈值
    method='correlation'  # 或 'mutual_info'
)
print(f"可能存在泄露的特征: {leakage_features}")

# 检测时间泄露（是否使用了未来信息）
time_leakage = detector.detect_time_leakage(
    df=df,
    date_col='date',
    target_col='target',
    check_future=True
)
if time_leakage['has_leakage']:
    print("警告：检测到时间泄露！")
    for warning in time_leakage['warnings']:
        print(f"  - {warning}")

# 检测分布泄露（训练集和测试集分布是否一致）
distribution_leakage = detector.detect_distribution_leakage(
    train_df=train_df,
    test_df=test_df,
    threshold=0.1
)
            if distribution_leakage['has_leakage']:
    print("警告：训练集和测试集分布不一致！")
```

### 12.4 模型集成工具

提供多种模型集成方法，包括投票、堆叠、Blending等。

```python
from wanglaoshi import CompetitionTools as CT
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 创建集成工具
ensemble = CT.ModelEnsemble()

# 准备多个模型
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = LogisticRegression(random_state=42)
model3 = RandomForestClassifier(n_estimators=200, random_state=42)

# 方法1：投票集成
voting_model = ensemble.voting(
    models=[model1, model2, model3],
    X=X_train,
    y=y_train,
    voting='soft',  # 或 'hard'
    weights=[0.4, 0.3, 0.3]  # 可选权重
)

# 方法2：堆叠集成（Stacking）
from sklearn.linear_model import LogisticRegression as MetaModel
meta_model = MetaModel()

stacked_result = ensemble.stacking(
    models=[model1, model2, model3],
    meta_model=meta_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,  # 可选，用于生成预测
    cv=5
)

# 方法3：Blending集成
blended_model = ensemble.blending(
    models=[model1, model2, model3],
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    meta_model=meta_model  # 或使用加权平均（不提供meta_model）
)
```

**功能特点**：

- 支持投票、堆叠、Blending三种集成方法
- 自动处理分类和回归任务
- 支持模型权重设置
- 堆叠使用交叉验证避免过拟合

### 12.5 特征重要性分析

分析特征重要性，帮助理解模型和选择特征。

```python
from wanglaoshi import CompetitionTools as CT

# 创建分析器
analyzer = CT.FeatureImportanceAnalyzer()

# 计算特征重要性（多种方法）
# 方法1：排列重要性（最可靠）
importance = analyzer.calculate_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='permutation',
    n_repeats=10
)

# 方法2：SHAP值（需要安装shap）
importance_shap = analyzer.calculate_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='shap'
)

# 方法3：模型内置重要性（仅树模型）
importance_builtin = analyzer.calculate_importance(
    model=trained_model,
    X=X_test,
    y=y_test,
    method='builtin'
)

# 可视化特征重要性
analyzer.plot_importance(importance, top_n=20)
```

### 12.6 交叉验证增强

提供时间序列、分组、分层等高级交叉验证方法。

```python
from wanglaoshi import CompetitionTools as CT

validator = CT.CrossValidator()

# 时间序列交叉验证
ts_result = validator.time_series_cv(
    model=model,
    X=X,
    y=y,
    n_splits=5
)

# 分组交叉验证（确保同一组数据不分开）
group_result = validator.group_cv(
    model=model,
    X=X,
    y=y,
    groups=group_column,  # 分组变量
    n_splits=5
)

# 分层交叉验证（适合不平衡数据）
stratified_result = validator.stratified_cv(
    model=model,
    X=X,
    y=y,
    n_splits=5
)
```

### 12.7 超参数优化

使用贝叶斯优化进行高效的超参数搜索。

```python
from wanglaoshi import CompetitionTools as CT
from sklearn.ensemble import RandomForestClassifier

optimizer = CT.HyperparameterOptimizer()

# 定义参数空间
param_space = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20}
}

# 贝叶斯优化
best_result = optimizer.bayesian_optimize(
    model_class=RandomForestClassifier,
    param_space=param_space,
    X=X_train,
    y=y_train,
    n_trials=100,
    cv=5
)

print(f"最佳参数: {best_result['best_params']}")
print(f"最佳分数: {best_result['best_score']:.4f}")

# 自动调参（自动选择参数范围）
auto_result = optimizer.auto_tune(
    model_class=RandomForestClassifier,
    X=X_train,
    y=y_train
)
```

**注意**：需要安装Optuna：`pip install optuna`

### 12.8 实验追踪

记录和管理机器学习实验，追踪不同实验的参数和结果。

```python
from wanglaoshi import CompetitionTools as CT

# 创建追踪器
tracker = CT.ExperimentTracker('my_experiments.json')

# 记录实验
tracker.log_experiment(
    experiment_name='exp_001',
    model_name='RandomForest',
    features=['feature1', 'feature2', 'feature3'],
    params={'n_estimators': 100, 'max_depth': 10},
    score=0.85,
    metrics={'f1': 0.83, 'precision': 0.87, 'recall': 0.81},
    notes='第一次实验，使用基础特征'
)

# 查看实验历史
history = tracker.get_experiment_history(sort_by='score', top_n=10)
print(history)

# 对比实验
comparison = tracker.compare_experiments(['exp_001', 'exp_002', 'exp_003'])
```

## 13. 增强特征工程 (FeatureEngineering)

FeatureEngineering 模块提供了丰富的特征工程功能，专门为数据科学比赛和学习设计。

### 13.1 时间特征提取

从日期时间列提取各种时间特征。

```python
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()

# 提取时间特征
df = engineer.extract_datetime_features(
    df=df,
    datetime_col='date',
    features=['year', 'month', 'day', 'weekday', 'quarter', 'is_weekend']
)

# 自动提取所有可用特征
df = engineer.extract_datetime_features(df, 'date')
```

**提取的特征**：

- 基础时间：年、月、日、小时、分钟、秒
- 周期特征：星期几、一年中的第几天、第几周、季度
- 布尔特征：是否周末、是否月初/月末

### 13.2 文本特征提取

从文本列提取特征，支持基础统计和TF-IDF。

```python
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()

# 方法1：基础统计特征
df = engineer.extract_text_features(
    df=df,
    text_col='text_column',
    method='basic'
)
# 生成：文本长度、词数、字符数、大写字母数、数字数、特殊字符数

# 方法2：TF-IDF特征（需要sklearn）
df = engineer.extract_text_features(
    df=df,
    text_col='text_column',
    method='tfidf',
    max_features=100
)
```

### 13.3 类别特征编码

提供多种强大的类别编码方法。

```python
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()

# 目标编码（Target Encoding）- 最强大的方法
df = engineer.target_encode(
    df=df,
    categorical_col='category_col',
    target_col='target',
    smoothing=1.0,  # 平滑参数，避免过拟合
    min_samples_leaf=1
)

# 频率编码（Frequency Encoding）
df = engineer.frequency_encode(df, 'category_col')

# One-Hot编码
df = engineer.one_hot_encode(
    df=df,
    categorical_cols=['cat1', 'cat2'],
    drop_first=False  # 是否删除第一个类别
)
```

### 13.4 数值特征变换

处理偏斜分布，使数据更接近正态分布。

```python
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()

# 对数变换（处理右偏分布）
df = engineer.log_transform(
    df=df,
    numeric_cols=['income', 'price'],
    add_one=True  # 先加1，处理0值
)

# Box-Cox变换（更强大的变换）
df = engineer.boxcox_transform(
    df=df,
    numeric_cols=['income', 'price']
)

# 特征分箱（将连续值转为离散值）
df = engineer.bin_features(
    df=df,
    numeric_cols=['age', 'income'],
    n_bins=5,
    strategy='quantile'  # 或 'uniform'
)
```

### 13.5 特征选择

基于多种方法选择重要特征。

```python
from wanglaoshi import FeatureEngineering as FE

engineer = FE.AdvancedFeatureEngineer()

# 方法1：基于重要性（使用树模型）
selected_features = engineer.select_features_by_importance(
    X=X_train,
    y=y_train,
    n_features=20,
    model_type='random_forest'  # 或 'xgboost', 'lightgbm'
)

# 方法2：基于相关性
selected_features = engineer.select_features_by_correlation(
    X=X_train,
    y=y_train,
    threshold=0.1  # 相关性阈值
)

# 方法3：基于互信息
selected_features = engineer.select_features_by_mutual_info(
    X=X_train,
    y=y_train,
    n_features=20
)
```

**检测类型**：

1. **目标泄露检测**：检测特征中是否包含目标变量的信息
2. **时间泄露检测**：检测是否使用了未来信息
3. **分布泄露检测**：检测训练集和测试集的分布差异

**功能特点**：

- 多种检测方法（相关性、互信息、KS检验）
- 详细的警告信息和建议
- 帮助避免比赛中的常见错误

## 对外接口一览

这一节汇总了常用的对外 API，方便你快速找到「应该从哪里导入什么」。

### 顶层导入（推荐）

这些是你在大多数脚本 / Notebook 里最常用的入口：

| 类型         | 说明                         | 导入方式 / 示例 |
|--------------|------------------------------|-----------------|
| 环境与版本   | 检查 Python / 库版本         | `from wanglaoshi import VERSIONS as V` |
| EDA 分析器   | 高级 EDA 与报告生成          | `from wanglaoshi import Analyzer, DataAnalyzer` |
| 比赛工具     | 提交文件、基线模型、集成等   | `from wanglaoshi import CompetitionTools as CT` |
| 特征工程     | 时间 / 文本 / 编码 / 选择等  | `from wanglaoshi import FeatureEngineering as FE` |
| ML/DL 基础   | 预处理、特征工程、ML/DL 模型 | `from wanglaoshi import MLDL` |
| Jupyter 环境 | 管理内核、查看环境信息       | `from wanglaoshi import JupyterEnv as JE` |
| 字体配置     | Matplotlib 中文字体支持      | `from wanglaoshi import JupyterFont as JF` |
| 常用工具     | pip 源、样式等               | `from wanglaoshi import Useful` |
| 下载工具     | 带进度条的文件下载           | `from wanglaoshi import WebGetter` |

常用函数 / 类可以直接从包顶层导入（少打一层模块前缀）：

| 名称                    | 来自模块   | 作用                            | 导入示例 |
|-------------------------|------------|---------------------------------|----------|
| `DataAnalyzer`          | `Analyzer` | 高级 EDA 主类                   | `from wanglaoshi import DataAnalyzer` |
| `analyze_data`          | `Analyzer` | 对单个数据文件做 EDA 并生成报告 | `from wanglaoshi import analyze_data` |
| `analyze_multiple_files`| `Analyzer` | 批量分析文件夹下的多个数据文件  | `from wanglaoshi import analyze_multiple_files` |
| `analyze_notebook`      | `Analyzer` | 在 Jupyter Notebook 中展示分析  | `from wanglaoshi import analyze_notebook` |

> 说明：上面这些函数 / 类都已经在 `wanglaoshi/__init__.py` 中导出，可以直接从 `wanglaoshi` 顶层导入，不需要经过子模块。

### 模块级 API 一览

| 模块名              | 主要内容 / 用途                                             | 常见入口 / 示例 |
|---------------------|--------------------------------------------------------------|-----------------|
| `VERSIONS`          | 环境与库版本检查，支持简单 / 完整库表、PyPI 最新版本对比     | `V.show_env_info()`、`V.check_all_versions(...)` |
| `Analyzer`          | 高级 EDA：缺失值、异常值、相关性、PCA、HTML 报告            | `analyzer = DataAnalyzer(df)`、`analyzer.generate_report()` |
| `Analyzer_Plain`    | 简化版文本 EDA：统计 + 文本报告 + 基本图表                  | `from wanglaoshi import Analyzer_Plain` |
| `CompetitionTools`  | 比赛工具：提交文件、快速基线、模型集成、调参、实验追踪      | `CT.SubmissionGenerator`、`CT.QuickBaseline`、`CT.ModelEnsemble` 等 |
| `FeatureEngineering`| 增强特征工程：时间特征、文本特征、编码、变换、特征选择      | `engineer = FE.AdvancedFeatureEngineer()` |
| `MLDL`              | ML/DL 封装：预处理、特征工程、传统 ML 模型、PyTorch DL 模型  | `from wanglaoshi import MLDL` |
| `JupyterEnv`        | Jupyter 内核管理、环境信息                                   | `JE.jupyter_kernel_list()`、`JE.install_kernel()` |
| `JupyterFont`       | Matplotlib 中文字体自动下载与配置                            | `JF.matplotlib_font_init()` |
| `Useful`            | 常用小工具：pip 镜像源、Jupyter Markdown 样式示例等          | `from wanglaoshi import Useful` |
| `WebGetter`         | 通用文件下载工具（支持进度条）                               | `from wanglaoshi import WebGetter` |

> 建议：做比赛 / 项目时，可以把这一小节当成「速查表」，需要某个功能时先看对应模块，再点进具体示例。

## 建议的版本对照关系

1. numpy [`https://numpy.org/news/`](https://numpy.org/news/)
2. pandas [`https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html`](https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html)
3. sklearn [`https://scikit-learn.org/stable/whats_new.html`](https://scikit-learn.org/stable/whats_new.html)