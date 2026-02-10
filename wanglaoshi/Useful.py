"""
实用工具函数模块 (Useful.py)

这个模块包含一些常用的辅助函数，主要用于：
1. 显示PyPI镜像源信息（加速pip安装）
2. 显示Jupyter Notebook的Markdown样式示例
3. 提供帮助信息

使用场景：
- 在Jupyter Notebook中快速查看常用信息
- 学习Markdown样式
- 配置pip镜像源

主要函数：
- pypis(): 显示国内PyPI镜像源地址和使用方法
- dep(): 显示依赖库版本信息获取方法
- styles(): 显示Jupyter Notebook支持的Markdown样式
- helps(): 显示帮助信息
"""

# ==================== PyPI镜像源相关 ====================

def pypis():
    """
    显示国内PyPI镜像源地址和使用方法
    
    这个函数会打印：
    1. 为什么需要使用镜像源（解决下载慢的问题）
    2. 常用的国内镜像源地址
    3. 如何使用镜像源（临时和永久两种方式）
    
    使用场景：
    - 当pip安装包速度慢时，可以使用镜像源加速
    - 在国内使用pip时特别有用
    
    示例:
        pypis()  # 查看所有镜像源信息
    """
    # 说明为什么需要镜像源
    print("在国内使用 pip 安装 Python 包时，经常会遇到下载速度慢的问题。")
    print("这是因为 pip 默认从国外的 PyPI 服务器下载包，速度比较慢。")
    print("可以通过更换国内的源地址来解决这个问题。")
    print("以下是一些常用的国内源地址：")
    # 列出常用的国内镜像源
    print("阿里云 https://mirrors.aliyun.com/pypi/simple/")
    print("清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/")
    print("中国科学技术大学 https://pypi.mirrors.ustc.edu.cn/simple/")
    print("豆瓣 https://pypi.doubanio.com/simple/")
    print("网易 https://mirrors.163.com/pypi/simple/")
    print("腾讯云 https://mirrors.cloud.tencent.com/pypi/simple")
    print("\n","-"*40)
    # 打印详细的使用说明
    print("""
使用说明

一、临时使用
使用pip的时候在后面加上-i参数，指定pip源：

pip install xxx -i https://mirrors.163.com/pypi/simple/
替换"xxx"为你需要安装的模块名称。

二、永久修改使用
Linux/Unix中使用：

~/.pip/pip.conf
添加或修改pip.conf（如果不存在，创建一个）

[global]
index-url = https://mirrors.163.com/pypi/simple/

Windows中使用：

%APPDATA%/pip/pip.ini
1.打开此电脑，在最上面的的文件夹窗口输入：%APPDATA%

2.按回车跳转进入目录，并新建一个文件夹：pip

3.创建文件：pip.ini

添加或修改pip.ini（如果不存在，创建一个）

[global]
index-url = https://mirrors.163.com/pypi/simple/
    """)

def dep():
    """
    显示获取项目依赖库版本信息的方法
    
    这个函数会显示如何使用pipdeptree工具来查看项目的依赖关系。
    
    pipdeptree是什么？
    - 一个Python工具，可以显示已安装包的依赖树
    - 帮助了解项目的依赖关系
    - 可以导出requirements.txt文件
    """
    print("获取项目依赖库的版本信息")
    print("使用方法：")
    print("https://github.com/WangLaoShi/pipdeptree")

def styles():
    """
    显示Jupyter Notebook支持的Markdown样式示例
    
    这个函数会打印所有可以在Jupyter Notebook的Markdown单元格中使用的样式，
    包括：
    - 标题（1-6级）
    - 列表（有序、无序）
    - 链接和图片
    - 代码块
    - 表格
    - 文本格式（加粗、斜体、删除线）
    - HTML样式（背景色、符号、彩色文本等）
    
    使用场景：
    - 学习如何在Jupyter Notebook中使用Markdown
    - 美化Notebook的显示效果
    - 创建格式化的文档
    
    注意：
    - 这些样式需要在Markdown单元格中使用
    - HTML样式需要在Markdown单元格中直接写HTML代码
    - 彩色文本需要在代码单元格中使用print()函数
    """
    print("*"*60)
    print("JupyterNotebook 支持的 Markdown 样式")
    
    # ========== 基础Markdown语法 ==========
    print("标题")
    print("# 一级标题")
    print("## 二级标题")
    print("### 三级标题")
    print("#### 四级标题")
    print("##### 五级标题")
    print("###### 六级标题")
    print("\n")
    
    print("列表")
    print("- 无序列表")
    print("1. 有序列表")
    print("\n")
    
    print("链接")
    print("[链接名称](链接地址)")
    print("\n")
    
    print("图片")
    print("![图片名称](图片地址)")
    print("\n")
    
    print("引用")
    print("> 引用内容")
    print("\n")
    
    print("代码")
    print("```python")
    print("print('Hello World!')")
    print("```")
    print("\n")
    
    print("表格")
    print("| 表头1 | 表头2 |")
    print("| --- | --- |")
    print("| 内容1 | 内容2 |")
    print("\n")
    
    print("加粗")
    print("**加粗内容**")
    print("\n")
    
    print("斜体")
    print("*斜体内容*")
    print("\n")
    
    print("删除线")
    print("~~删除线内容~~")
    print("\n")
    
    print("分割线")
    print("---")
    print("\n")
    
    print("脚注")
    print("脚注[^1]")
    print("[^1]: 脚注内容")
    print("\n")
    
    # ========== HTML样式（在Markdown单元格中使用）==========
    print("格式化-背景颜色")
    print("# <div style='background-color:skyblue'><center> TEXT WITH BACKGROUND COLOR </center></div>")
    print("\n")
    
    print("格式化-背景颜色（使用Bootstrap样式类）")
    print("""
# Blue Background
<div class="alert alert-info"> Example text highlighted in blue background </div>
# Green Background
<div class="alert alert-success">Example text highlighted in green background.</div>
# Yellow Background
<div class="alert alert-warning">Example text highlighted in yellow background.</div>
# Red Background
<div class="alert alert-danger">Example text highlighted in red background.</div>
    """)
    print("\n")

    print("格式化-符号（使用HTML实体）")
    print("""
&#10148; Bullet point one</br>
&#10143; Bullet point two</br>
&#10147; Bullet point three</br>
&#10145; Bullet point four</br>
&#10144; Bullet point five</br>
&#10142; Bullet point six</br>
&#10141; Bullet point seven</br>
&#10140; Bullet point eight</br>
    """)
    print("\n")
    
    # ========== 终端彩色文本（在代码单元格中使用）==========
    print("格式化-有色文本（在代码单元格中使用print()）")
    print("""
print('\033[31;3m This is red\033[0m')
print('\033[32;3m This is green\033[0m')
print('\033[33;3m This is yellow\033[0m')
print('\033[34;3m This is blue\033[0m')
print('\033[35;3m This is pink\033[0m')
print('\033[36;3m This is skyblue\033[0m')
print('\033[37;3m This is grey\033[0m')
    """)
    print("\n")
    
    print("格式化-黑体文字（在代码单元格中使用print()）")
    print("""
print('\033[1;31m This is bold red \033[0m')
print('\033[1;32m This is bold green\033[0m')
print('\033[1;33m This is bold yellow\033[0m')
print('\033[1;34m This is bold blue\033[0m')
print('\033[1;35m This is bold purple\033[0m')
print('\033[1;36m This is bold teal\033[0m')
print('\033[1;37m This is bold grey\033[0m')
    """)
    print("\n")
    
    print("格式化-背景颜色（在代码单元格中使用print()）")
    print("""
print('\033[1;40mBlack background - Bold text\033[0m')
print('\033[1;41mRed background - Bold text\033[0m')
print('\033[1;42mGreen background - Bold text\033[0m')
print('\033[1;43mYellow background - Bold text\033[0m')
print('\033[1;44mBlue background - Bold text\033[0m')
print('\033[1;45mPink background - Bold text\033[0m')
print('\033[1;46mLight Blue background - Bold text\033[0m')
print('\033[1;47mLight Grey background - Bold text\033[0m')    
    """)
    print("*"*60)
    print("\n")
    print("*"*60)

def helps():
    """
    显示本模块的帮助信息
    
    这个函数会列出本模块中所有可用的函数及其用途。
    """
    print("本模块包含一些常用的函数，可以直接调用。")
    print("pypis()  # 常用的 pypi 更新源地址")
    print("dep()  # 获取项目依赖库的版本信息")
    print("helps()  # 查看帮助信息")
    print("styles()  # 查看 JupyterNotebook 支持的样式")

# 模块加载时自动显示样式信息
styles()