# 王老师 WangLaoShi

## 项目介绍

总结一些在学习过程中的知识点，以及一些学习资料。

## 项目结构

```
WangLaoShi
├── README.md
├── wanglaoshi
│   ├── version.py
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

## 安装方式

### 1. 源码安装方式

* 检出项目
* 进入项目目录
* 执行`python setup.py install`
* 安装完成

### 2. pip安装方式

```shell
pip install wanglaoshi
```

## 使用方法

### 1. 创建新的环境
    
```python
from wanglaoshi import JupyterEnv as JE
JE.jupyter_kernel_list()
JE.install_kernel()
# 按照提示输入环境名称
```
### 2. 获取当前环境常用库版本
    
```python
from wanglaoshi import VERSIONS as V
V.check_all_versions()
```
### 3. 获取当前环境所有库

```python
from wanglaoshi import VERSIONS as V
V.check_all_installed()
```
### 4. 获取当前环境所有库最新版本

```python
from wanglaoshi import VERSIONS as V
V.check_all_installed_with_latest()
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
from wanglaoshi import DataAnalyzer
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


## 建议的版本对照关系

1. numpy https://numpy.org/news/
2. pandas https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html
3. sklearn https://scikit-learn.org/stable/whats_new.html