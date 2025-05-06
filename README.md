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
A.analyze_data_to_html('data.csv')
```

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

## 9. 数据的基本分析（适合比赛）

```python
from wanglaoshi import Analyzer as A
folder_path = "your_data_folder"  # 替换为实际的文件夹路径
target_dir = "reports"            # 替换为实际的报告保存目录
A.load_and_explore_csvs(folder_path, report=True, target_dir=target_dir)
```

## 建议的版本对照关系

1. numpy https://numpy.org/news/
2. pandas https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html
3. sklearn https://scikit-learn.org/stable/whats_new.html