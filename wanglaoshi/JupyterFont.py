import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import sys
import os
# Add the directory containing WebGetter.py to the Python path
sys.path.append(os.path.dirname(__file__))
from WebGetter import Wget

def is_chinese_font(font_path):
    """测试字体是否支持中文"""
    try:
        font = fm.FontProperties(fname=font_path)
        # 尝试渲染中文字符
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '中文测试', fontproperties=font)
        plt.close(fig)
        return True
    except:
        return False

def fonts(show_demo = False):
    # 获取所有字体
    all_fonts = fm.findSystemFonts()
    chinese_fonts = []

    # 测试每个字体
    for font_path in all_fonts:
        try:
            if is_chinese_font(font_path):
                font = fm.FontProperties(fname=font_path)
                font_name = font.get_name()
                chinese_fonts.append({
                    'name': font_name,
                    'path': font_path
                })
        except:
            continue

    # 打印支持中文的字体
    print("\n支持中文的字体:")
    for font in chinese_fonts:
        print(f"字体名称: {font['name']}")
        print(f"字体路径: {font['path']}\n")

    # 测试显示
    if chinese_fonts and show_demo == True:
        plt.figure(figsize=(15, len(chinese_fonts)))
        for i, font in enumerate(chinese_fonts):
            plt.text(0.1, 1 - (i * 0.1),
                     f'这是{font["name"]}字体的中文显示测试',
                     fontproperties=fm.FontProperties(fname=font['path']),
                     fontsize=12)
        plt.axis('off')
        plt.show()

def simhei():
    # 创建下载器实例
    downloader = Wget(
        url='https://gitee.com/lincoln/fonts/raw/master/SimHei.ttf',
        output_dir='.',
        filename='SimHei.ttf',
    )
    downloader.download()

def matplotlib_font_init(show_demo = False):
    # 判断，同级文件夹下是否存在 SimHei.ttf 文件，如果没有，执行 simhei() 函数下载
    if not os.path.exists('./SimHei.ttf'):
        simhei()
    # 设置中文字体和样式
    matplotlib.font_manager.fontManager.addfont('./SimHei.ttf')
    matplotlib.rc('font', family='SimHei')
    # plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    # 测试显示
    if show_demo == True:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, '这是中文测试', fontsize=20)
        plt.show()