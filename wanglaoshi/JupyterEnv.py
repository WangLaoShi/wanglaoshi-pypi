"""
Jupyter环境管理模块 (JupyterEnv.py)

这个模块提供了Jupyter Notebook环境相关的工具函数，包括：
1. 系统命令执行
2. 运行环境检测（操作系统、Python版本等）
3. Jupyter内核管理（安装、列出等）

使用场景：
- 在Jupyter Notebook中检查运行环境
- 安装和配置Jupyter内核
- 获取Python环境信息
"""

import sys  # 系统相关参数和函数
import os   # 操作系统接口

def run_cmd(cmd):
    """
    在系统中执行命令
    
    这个函数用于执行系统命令（如pip、jupyter等），
    并将命令的输出显示在控制台。
    
    参数:
        cmd: 要执行的命令字符串
            例如: "pip install numpy" 或 "jupyter kernelspec list"
            
    返回:
        None（无返回值，直接打印输出）
        
    示例:
        run_cmd("pip list")  # 列出已安装的包
    """
    # 打印要执行的命令
    print('Running command:', cmd)
    # os.popen() 打开一个管道，执行命令并返回文件对象
    # 这样可以读取命令的输出
    p = os.popen(cmd)
    # 使用read()获取命令的完整输出（一个字符串）
    results = p.read()
    # 将输出按行分割成列表
    # split('\n') 按换行符分割字符串
    results_lst = results.split('\n')
    # 逐行打印输出
    for line in results_lst:
        print(line)

def running_os():
    """
    检测当前运行的操作系统
    
    通过sys.platform获取系统平台标识符，并转换为易读的操作系统名称。
    
    返回:
        操作系统名称字符串：
        - 'Windows': Windows系统
        - 'Linux': Linux系统
        - 'MacOS': macOS系统（苹果系统）
        - 'Unknown': 未知系统
        
    示例:
        os_name = running_os()  # 返回 'Windows' 或 'Linux' 或 'MacOS'
    """
    # sys.platform 返回系统平台标识符
    # 'win32' 表示Windows（包括64位）
    # 'linux' 表示Linux
    # 'darwin' 表示macOS
    env = sys.platform
    if env == 'win32':
        return 'Windows'
    elif env == 'linux':
        return 'Linux'
    elif env == 'darwin':
        return 'MacOS'
    else:
        return 'Unknown'

def running_python_version():
    """
    获取当前运行的Python版本
    
    返回:
        Python版本字符串，例如: "3.9.7 (default, Sep 3 2021, 06:20:32)"
        
    示例:
        version = running_python_version()
        print(version)  # 输出: "3.9.7 (default, Sep 3 2021, 06:20:32)"
    """
    # sys.version 返回Python解释器的版本信息字符串
    return sys.version

def running_python_path():
    """
    获取当前运行的Python解释器路径
    
    返回:
        Python解释器的完整路径字符串
        例如: "/usr/bin/python3" 或 "C:\\Python39\\python.exe"
        
    示例:
        path = running_python_path()
        print(path)  # 输出Python解释器的路径
    """
    # sys.executable 返回Python解释器的可执行文件路径
    return sys.executable

def running_python_version_info():
    """
    获取当前运行的Python版本详细信息
    
    返回:
        sys.version_info对象，包含版本号的各个组成部分
        例如: sys.version_info(major=3, minor=9, micro=7, ...)
        
    示例:
        version_info = running_python_version_info()
        print(version_info.major)  # 输出主版本号，如 3
        print(version_info.minor)  # 输出次版本号，如 9
    """
    # sys.version_info 返回版本信息的命名元组
    return sys.version_info

def running():
    """
    显示完整的运行环境信息
    
    这个函数会打印并返回以下信息：
    1. 操作系统类型
    2. Python版本
    3. Python解释器路径
    4. Python版本详细信息
    
    返回:
        一个元组，包含：
        (操作系统名称, Python版本字符串, Python路径, Python版本信息对象)
        
    示例:
        os_name, version, path, version_info = running()
        # 同时会在控制台打印这些信息
    """
    # 打印分隔线（40个星号）
    print("*"*40)
    # 打印并获取各项信息
    print('Running environment:', running_os())
    print('Running Python version:', running_python_version())
    print('Running Python path:', running_python_path())
    print('Running Python version info:', running_python_version_info())
    print("*" * 40)
    # 返回所有信息组成的元组
    return running_os(), running_python_version(), running_python_path(), running_python_version_info()

def install_ipykernel():
    """
    安装ipykernel包
    
    ipykernel是什么？
    - ipykernel是Jupyter Notebook的内核包
    - 它允许Jupyter Notebook使用Python解释器来执行代码
    - 必须安装ipykernel才能创建自定义的Jupyter内核
    
    这个函数会执行命令: pip install ipykernel
    """
    # 使用pip安装ipykernel包
    # 注释中的 !pip install ipykernel 是Jupyter Notebook中的魔法命令语法
    run_cmd('pip install ipykernel')

def jupyter_kernel_list():
    """
    列出所有已安装的Jupyter内核
    
    这个函数会显示当前系统中所有可用的Jupyter内核，
    包括内核名称、显示名称和路径等信息。
    
    执行命令: jupyter kernelspec list
    """
    # 列出所有已安装的Jupyter内核
    # 注释中的 !jupyter kernelspec list 是Jupyter Notebook中的魔法命令语法
    run_cmd('jupyter kernelspec list')

def install_kernel():
    """
    安装新的Jupyter内核（交互式安装）
    
    这个函数会：
    1. 列出当前已安装的内核
    2. 安装ipykernel（如果未安装）
    3. 显示运行环境信息
    4. 提示用户输入新内核的名称和显示名称
    5. 安装新内核
    
    使用场景：
    - 为特定的Python环境创建Jupyter内核
    - 创建多个不同名称的Python内核
    
    注意：
    - 这个函数会要求用户交互输入（使用input()）
    - 不适合在非交互式环境中使用
    """
    # 步骤1：列出当前已安装的内核
    jupyter_kernel_list()
    # 步骤2：安装ipykernel（如果未安装）
    install_ipykernel()
    # 步骤3：显示运行环境信息
    running()
    # 步骤4：获取Python解释器路径
    python_path = running_python_path()
    # 步骤5：提示用户输入新内核的信息
    print("Please input the new kernel name and display name.")
    # 获取内核名称（用于内部标识）
    kernel_name = input("Please input the kernel name:")
    # 获取显示名称（在Jupyter Notebook中显示的名称）
    kernel_display_name = input("Please input the kernel display name:")
    # 步骤6：安装新内核
    # -m ipykernel install: 使用ipykernel模块安装内核
    # --user: 安装到用户目录（不需要管理员权限）
    # --name: 指定内核名称
    # --display-name: 指定显示名称
    run_cmd(python_path + ' -m ipykernel install --user --name=' + kernel_name + ' --display-name=' + kernel_display_name)

def no_warning():
    """
    禁用Python警告信息
    
    这个函数会忽略所有Python警告（warnings），
    让代码运行时不会显示警告信息。
    
    使用场景：
    - 当代码会产生大量警告但不想看到时
    - 在演示或报告中希望输出更简洁时
    
    注意：
    - 禁用警告可能会隐藏重要的问题提示
    - 建议只在确认警告不影响功能时使用
    """
    import warnings
    # filterwarnings('ignore') 忽略所有警告
    warnings.filterwarnings('ignore')
