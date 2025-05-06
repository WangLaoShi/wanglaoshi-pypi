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

```python
from wanglaoshi import Analyzer as A
A.analyze_data('data.csv')  # 生成HTML格式的分析报告
```

分析报告包含以下内容：
- 基础统计分析：描述性统计、缺失值分析等
- 异常值分析：
  - Z-score方法：识别极端异常值（|Z-score| > 3）和中度异常值（2 < |Z-score| ≤ 3）
  - IQR方法：基于四分位距的异常值检测，提供数据分布特征和异常值范围
  - 详细的异常值解释和处理建议
- 数据质量分析：重复值、缺失值等
- 高级统计分析：相关性分析、主成分分析等

如果不需要 HTML 页面也可以使用下面的方法

```python
from wanglaoshi import Analyzer_Plain as A
A.analyze_data('data.csv')
```

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


## 建议的版本对照关系

1. numpy https://numpy.org/news/
2. pandas https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html
3. sklearn https://scikit-learn.org/stable/whats_new.html