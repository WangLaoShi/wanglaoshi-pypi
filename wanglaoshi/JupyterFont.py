"""
Jupyter字体管理模块 (JupyterFont.py)

这个模块提供了Jupyter Notebook中matplotlib图表的中文字体支持功能，包括：
1. 检测系统字体是否支持中文
2. 列出所有支持中文的字体
3. 自动下载和配置SimHei（黑体）字体
4. 初始化matplotlib的中文字体设置

使用场景：
- 在Jupyter Notebook中绘制包含中文的图表
- 解决matplotlib中文显示为方块的问题
- 自动配置中文字体环境

主要功能：
- fonts(): 列出所有支持中文的字体
- simhei(): 下载SimHei字体
- matplotlib_font_init(): 初始化matplotlib中文字体
"""

import matplotlib  # matplotlib主模块
import matplotlib.font_manager as fm  # 字体管理器
import matplotlib.pyplot as plt  # 绘图模块
import sys  # 系统相关
import os   # 操作系统接口

# 将当前文件所在目录添加到Python路径
# 这样可以从当前目录导入WebGetter模块
sys.path.append(os.path.dirname(__file__))
from WebGetter import Wget  # 导入文件下载工具类

def is_chinese_font(font_path):
    """
    测试字体文件是否支持中文显示
    
    这个函数通过实际渲染中文字符来测试字体是否支持中文。
    如果字体不支持中文，渲染时会出错或显示为方块。
    
    参数:
        font_path: 字体文件的完整路径
                  例如: "/path/to/font.ttf"
        
    返回:
        True: 字体支持中文
        False: 字体不支持中文或测试失败
        
    工作原理：
    1. 加载字体文件
    2. 尝试在图表中渲染"中文测试"这几个字
    3. 如果成功渲染，说明字体支持中文
    """
    try:
        # 创建字体属性对象，指定字体文件路径
        font = fm.FontProperties(fname=font_path)
        # 创建一个测试图表（不显示，只用于测试）
        fig, ax = plt.subplots()
        # 尝试在图表中渲染中文字符"中文测试"
        # 如果字体不支持中文，这里可能会出错或显示为方块
        ax.text(0.5, 0.5, '中文测试', fontproperties=font)
        # 关闭测试图表（不显示）
        plt.close(fig)
        # 如果执行到这里没有出错，说明字体支持中文
        return True
    except:
        # 如果出现任何错误（字体不支持中文、文件损坏等），返回False
        return False

def fonts(show_demo = False):
    """
    查找并列出系统中所有支持中文的字体
    
    这个函数会：
    1. 扫描系统中的所有字体文件
    2. 测试每个字体是否支持中文
    3. 列出所有支持中文的字体及其路径
    4. 可选：显示字体效果演示
    
    参数:
        show_demo: 是否显示字体效果演示
                  - True: 显示一个图表，展示所有支持中文的字体效果
                  - False: 只打印字体列表（默认）
                  
    返回:
        None（直接打印结果）
        
    使用场景：
    - 查找系统中可用的中文字体
    - 选择合适的字体用于图表
    - 检查字体是否正确安装
    """
    # 步骤1：获取系统中所有字体文件的路径
    # findSystemFonts() 会扫描系统字体目录，返回所有字体文件的路径列表
    all_fonts = fm.findSystemFonts()
    # 存储支持中文的字体信息
    chinese_fonts = []

    # 步骤2：测试每个字体是否支持中文
    for font_path in all_fonts:
        try:
            # 使用is_chinese_font()函数测试字体
            if is_chinese_font(font_path):
                # 如果支持中文，获取字体名称
                font = fm.FontProperties(fname=font_path)
                font_name = font.get_name()
                # 将字体信息添加到列表
                chinese_fonts.append({
                    'name': font_name,      # 字体名称
                    'path': font_path      # 字体文件路径
                })
        except:
            # 如果测试过程中出错（字体损坏等），跳过这个字体
            continue

    # 步骤3：打印支持中文的字体列表
    print("\n支持中文的字体:")
    for font in chinese_fonts:
        print(f"字体名称: {font['name']}")
        print(f"字体路径: {font['path']}\n")

    # 步骤4：可选 - 显示字体效果演示
    if chinese_fonts and show_demo == True:
        # 创建一个图表，高度根据字体数量调整
        plt.figure(figsize=(15, len(chinese_fonts)))
        # 为每个字体显示一行测试文本
        for i, font in enumerate(chinese_fonts):
            # 在图表上添加文本，使用对应的字体
            # 位置：x=0.1, y=1-(i*0.1)（从上到下排列）
            plt.text(0.1, 1 - (i * 0.1),
                     f'这是{font["name"]}字体的中文显示测试',
                     fontproperties=fm.FontProperties(fname=font['path']),
                     fontsize=12)
        # 隐藏坐标轴（只显示文本）
        plt.axis('off')
        # 显示图表
        plt.show()

def simhei():
    """
    下载SimHei（黑体）字体文件
    
    SimHei是什么？
    - SimHei是Windows系统自带的中文字体（黑体）
    - 支持中文显示，常用于数据可视化
    - 如果系统没有这个字体，可以从网上下载
    
    这个函数会：
    1. 从Gitee下载SimHei.ttf字体文件
    2. 保存到当前目录
    
    注意：
    - 需要网络连接
    - 下载的文件会保存为当前目录下的SimHei.ttf
    """
    # 创建文件下载器实例
    # Wget是WebGetter模块中的下载工具类
    downloader = Wget(
        url='https://gitee.com/lincoln/fonts/raw/master/SimHei.ttf',  # 字体文件下载地址
        output_dir='.',  # 输出目录：当前目录
        filename='SimHei.ttf',  # 文件名
    )
    # 执行下载
    downloader.download()

def matplotlib_font_init(show_demo = False):
    """
    初始化matplotlib的中文字体设置
    
    这个函数是使用matplotlib绘制中文图表的关键函数。
    它会：
    1. 检查SimHei字体是否存在，如果不存在则自动下载
    2. 将SimHei字体添加到matplotlib的字体管理器
    3. 设置matplotlib使用SimHei字体
    4. 解决负号显示问题
    5. 可选：显示测试图表验证中文显示
    
    参数:
        show_demo: 是否显示测试图表
                  - True: 显示一个包含中文的测试图表
                  - False: 只配置字体，不显示图表（默认）
                  
    使用场景：
    - 在Jupyter Notebook中绘制包含中文的图表前调用
    - 解决matplotlib中文显示为方块的问题
    
    示例:
        # 在Jupyter Notebook中
        import matplotlib.pyplot as plt
        from wanglaoshi import JupyterFont
        
        JupyterFont.matplotlib_font_init()  # 初始化中文字体
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title('这是中文标题')  # 现在可以正常显示中文了
        plt.show()
    """
    # 步骤1：检查SimHei字体文件是否存在
    # 如果不存在，自动下载
    if not os.path.exists('./SimHei.ttf'):
        simhei()  # 调用下载函数
    
    # 步骤2：将SimHei字体添加到matplotlib的字体管理器
    # addfont() 将字体文件添加到字体管理器，使其可以被matplotlib使用
    matplotlib.font_manager.fontManager.addfont('./SimHei.ttf')
    
    # 步骤3：设置matplotlib的默认字体为SimHei
    # rc() 用于设置matplotlib的运行时配置
    # family='SimHei' 设置字体族为SimHei
    matplotlib.rc('font', family='SimHei')
    
    # 注释掉的代码是另一种设置方式（功能相同）
    # plt.rcParams['font.sans-serif'] = ['simhei']
    
    # 步骤4：解决负号显示问题
    # 默认情况下，matplotlib可能将负号显示为方块
    # 设置axes.unicode_minus = False 可以解决这个问题
    plt.rcParams['axes.unicode_minus'] = False

    # 步骤5：可选 - 显示测试图表
    if show_demo == True:
        # 创建一个测试图表
        plt.figure(figsize=(8, 6))
        # 在图表中央显示中文测试文本
        plt.text(0.5, 0.5, '这是中文测试', fontsize=20)
        # 显示图表（验证中文是否正常显示）
        plt.show()