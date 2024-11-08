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

## 安装方式

1. 源码安装方式

* 检出项目
* 进入项目目录
* 执行`python setup.py install`
* 安装完成

2. pip安装方式

```shell
pip install wanglaoshi
```

## 使用方法

1. 创建新的环境
    
```python
from wanglaoshi import JupyterEnv as JE
JE.jupyter_kernel_list()
JE.install_kernel()
# 按照提示输入环境名称
```
2. 获取当前环境常用库版本
    
```python
from wanglaoshi import VERSIONS as V
V.check_all_versions()
```
3. 获取当前环境所有库
```python
from wanglaoshi import VERSIONS as V
V.check_all_installed()
```
4. 获取当前环境所有库最新版本
```python
from wanglaoshi import VERSIONS as V
V.check_all_installed_with_latest()
```

## 建议的版本对照关系

1. numpy https://numpy.org/news/
2. pandas https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html
3. sklearn https://scikit-learn.org/stable/whats_new.html