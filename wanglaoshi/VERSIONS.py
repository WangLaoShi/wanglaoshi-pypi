import importlib
import pkg_resources
ml_dl_libraries = {
    "numpy": "用于数值计算",
    "pandas": "数据处理与操作",
    "scikit-learn": "经典的机器学习库",
    "tensorflow": "Google 的深度学习框架",
    "keras": "基于 TensorFlow 的高级深度学习 API",
    "pytorch": "Facebook 开发的深度学习框架",
    "xgboost": "高效的梯度提升库，常用于比赛",
    "lightgbm": "高效的梯度提升决策树库",
    "catboost": "适合处理分类数据的梯度提升库",
    "matplotlib": "数据可视化",
    "seaborn": "基于 Matplotlib 的数据可视化库",
    "plotly": "交互式图表库",
    "scipy": "科学计算库",
    "statsmodels": "统计建模和计量经济学",
    "nltk": "自然语言处理库",
    "spacy": "高效的自然语言处理库",
    "transformers": "用于使用预训练的自然语言处理模型",
    "opencv-python": "图像处理库",
    "Pillow": "图像操作库",
    "gym": "强化学习环境",
    "ray": "分布式计算，用于加速训练",
    "joblib": "并行计算与模型持久化"
}

def check_versions(libraries=ml_dl_libraries):
    """显示指定库的版本信息"""
    versions = {}
    for lib in ml_dl_libraries.keys():
        try:
            module = importlib.import_module(lib)
            version = pkg_resources.get_distribution(lib).version
            versions[lib] = version
        except ImportError:
            versions[lib] = 'Not installed'
    return versions
def check_all_versions():
    """显示所有常用库的版本信息"""
    results = check_versions(ml_dl_libraries.keys())
    from rich.console import Console
    from rich.table import Column, Table

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=12)
    table.add_column("Description", justify="left")
    table.add_column("Version", justify="right")
    for key, value in results.items():
        desc = ml_dl_libraries[key]
        if not value == 'Not installed':
            value = ':smiley: ' + '[red]' + value + '[/red]'
        table.add_row(key, desc, value)
    console.print(table)

