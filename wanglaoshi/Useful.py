# 放置一个经常用到的函数

def pypis():
    # 常用的 pypi 更新源地址
    print("在国内使用 pip 安装 Python 包时，经常会遇到下载速度慢的问题。")
    print("这是因为 pip 默认从国外的 PyPI 服务器下载包，速度比较慢。")
    print("可以通过更换国内的源地址来解决这个问题。")
    print("以下是一些常用的国内源地址：")
    print("阿里云 https://mirrors.aliyun.com/pypi/simple/")
    print("清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/")
    print("中国科学技术大学 https://pypi.mirrors.ustc.edu.cn/simple/")
    print("豆瓣 https://pypi.doubanio.com/simple/")
    print("网易 https://mirrors.163.com/pypi/simple/")
    print("腾讯云 https://mirrors.cloud.tencent.com/pypi/simple")
    print("\n","-"*40)
    print("""
使用说明

一、临时使用
使用pip的时候在后面加上-i参数，指定pip源：

pip install xxx -i https://mirrors.163.com/pypi/simple/
替换“xxx”为你需要安装的模块名称。

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
    print("获取项目依赖库的版本信息")
    print("使用方法：")
    print("https://github.com/WangLaoShi/pipdeptree")
# pypis()

def styles():
    print("*"*60)
    print("JupyterNotebook 支持的 Markdown 样式")
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
    print("格式化-背景颜色")
    print("# <div style='background-color:skyblue'><center> TEXT WITH BACKGROUND COLOR </center></div>")
    print("\n")
    print("格式化-背景颜色")
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

    print("格式化-符号")
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
    print("格式化-有色文本")
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
    print("格式化-黑体文字")
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
    print("格式化-背景颜色")
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
    print("本模块包含一些常用的函数，可以直接调用。")
    print("pypis()  # 常用的 pypi 更新源地址")
    print("dep()  # 获取项目依赖库的版本信息")
    print("helps()  # 查看帮助信息")
    print("styles()  # 查看 JupyterNotebook 支持的样式")
styles()