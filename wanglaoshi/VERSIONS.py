from rich.console import Console
from rich.table import Column, Table
from importlib.metadata import version, PackageNotFoundError,distributions  # Python 3.8+
import requests
ml_dl_libraries = {
    "numpy": {
        "url": "https://numpy.org/",
        "description": "用于数值计算",
        "category": "数据处理与可视化"
    },
    "pandas": {
        "url": "https://pandas.pydata.org/",
        "description": "数据处理与操作",
        "category": "数据处理与可视化"
    },
    "scikit-learn": {
        "url": "https://scikit-learn.org/",
        "description": "scikit-learn 经典的机器学习库",
        "category": "机器学习"
    },
    "tensorflow": {
        "url": "https://www.tensorflow.org/",
        "description": "Google 的深度学习框架",
        "category": "深度学习"
    },
    "keras": {
        "url": "https://keras.io/",
        "description": "基于 TensorFlow 的高级深度学习 API",
        "category": "深度学习"
    },
    "pytorch": {
        "url": "https://pytorch.org/",
        "description": "Facebook 开发的深度学习框架",
        "category": "深度学习"
    },
    "xgboost": {
        "url": "https://xgboost.readthedocs.io/",
        "description": "高效的梯度提升库，常用于比赛",
        "category": "机器学习"
    },
    "lightgbm": {
        "url": "https://lightgbm.readthedocs.io/",
        "description": "高效的梯度提升决策树库",
        "category": "机器学习"
    },
    "catboost": {
        "url": "https://catboost.ai/",
        "description": "适合处理分类数据的梯度提升库",
        "category": "机器学习"
    },
    "matplotlib": {
        "url": "https://matplotlib.org/",
        "description": "数据可视化",
        "category": "数据处理与可视化"
    },
    "seaborn": {
        "url": "https://seaborn.pydata.org/",
        "description": "基于 Matplotlib 的数据可视化库",
        "category": "数据处理与可视化"
    },
    "plotly": {
        "url": "https://plotly.com/",
        "description": "交互式图表库",
        "category": "数据处理与可视化"
    },
    "scipy": {
        "url": "https://scipy.org/",
        "description": "科学计算库",
        "category": "数据处理与可视化"
    },
    "statsmodels": {
        "url": "https://www.statsmodels.org/",
        "description": "统计建模和计量经济学",
        "category": "数据处理与可视化"
    },
    "nltk": {
        "url": "https://www.nltk.org/",
        "description": "自然语言处理库",
        "category": "自然语言处理"
    },
    "spacy": {
        "url": "https://spacy.io/",
        "description": "高效的自然语言处理库",
        "category": "自然语言处理"
    },
    "transformers": {
        "url": "https://huggingface.co/transformers/",
        "description": "用于使用预训练的自然语言处理模型",
        "category": "自然语言处理"
    },
    "cv2": {
        "url": "https://opencv.org/",
        "description": "opencv-python 图像处理库",
        "category": "计算机视觉"
    },
    "PIL": {
        "url": "https://python-pillow.org/",
        "description": "Pillow 图像操作库",
        "category": "计算机视觉"
    },
    "gym": {
        "url": "https://www.gymlibrary.ml/",
        "description": "强化学习环境",
        "category": "强化学习"
    },
    "ray": {
        "url": "https://www.ray.io/",
        "description": "分布式计算，用于加速训练",
        "category": "分布式计算"
    },
    "joblib": {
        "url": "https://joblib.readthedocs.io/",
        "description": "并行计算与模型持久化",
        "category": "分布式计算"
    },
    "dask": {
        "url": "https://dask.org/",
        "description": "并行计算库，支持大规模数据处理",
        "category": "分布式计算"
    },
    "mlflow": {
        "url": "https://mlflow.org/",
        "description": "机器学习生命周期管理",
        "category": "机器学习管理"
    },
    "wandb": {
        "url": "https://wandb.ai/",
        "description": "实验跟踪与可视化",
        "category": "机器学习管理"
    },
    "hydra": {
        "url": "https://hydra.cc/",
        "description": "配置管理工具",
        "category": "机器学习管理"
    },
    "optuna": {
        "url": "https://optuna.org/",
        "description": "超参数优化库",
        "category": "机器学习管理"
    },
    "pycaret": {
        "url": "https://pycaret.org/",
        "description": "低代码机器学习库",
        "category": "机器学习管理"
    },
    "onnx": {
        "url": "https://onnx.ai/",
        "description": "用于深度学习模型的开放式神经网络交换格式",
        "category": "深度学习"
    },
    "albumentations": {
        "url": "https://albumentations.ai/",
        "description": "图像增强库",
        "category": "计算机视觉"
    },
    "tqdm": {
        "url": "https://tqdm.github.io/",
        "description": "进度条显示库",
        "category": "其他"
    },
    "librosa": {
        "url": "https://librosa.org/",
        "description": "用于音频和音乐信号分析的 Python 包，提供功能丰富的音频处理工具",
        "category": "音频处理"
    },
    "mir_eval": {
        "url": "https://craffel.github.io/mir_eval/",
        "description": "用于音乐信息检索 (MIR) 任务的评估库，支持一系列音乐分析基准的评估功能",
        "category": "音乐信息检索"
    }
}

def check_versions(libraries=ml_dl_libraries):
    """显示指定库的版本信息"""
    versions = {}
    for lib in libraries.keys():
        try:
            # module = importlib.import_module(lib)
            lib_version = version(lib)
            versions[lib] = lib_version
        except PackageNotFoundError:
            versions[lib] = 'Not installed'
        except ModuleNotFoundError:
            versions[lib] = 'Not installed'
    return versions

def check_all_versions(all_columns=False):
    """显示所有常用库的版本信息"""
    results = check_versions(ml_dl_libraries)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=16)
    table.add_column("Description", justify="left",width=30)
    if all_columns:
        table.add_column("Website", justify="left",width=30)
        table.add_column("Category", justify="left",width=20)
    table.add_column("Version", justify="right",width=20)
    for key, value in results.items():
        desc = ml_dl_libraries[key]["description"]
        site = ml_dl_libraries[key]["url"]
        category = ml_dl_libraries[key]["category"]
        if not value == 'Not installed':
            value = ':smiley: ' + '[red]' + value + '[/red]'
        if all_columns:
            table.add_row(key, desc, site,category, value)
        else:
            table.add_row(key, desc,  value)
    console.print(table)


def check_all_installed():
    """检查并显示所有已安装的 Python 库和版本信息"""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=30)
    table.add_column("Version", justify="right", width=15)

    # 获取所有已安装库的信息
    installed_packages = sorted((dist.metadata["Name"], dist.version) for dist in distributions())

    for name, version in installed_packages:
        table.add_row(name, version)

    console.print(table)


def get_latest_version(package_name):
    """查询 PyPI 获取指定包的最新版本"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()["info"]["version"]
        else:
            return "Not found"
    except requests.RequestException:
        return "Error"


def check_all_installed_with_latest():
    """显示所有已安装库的当前版本和最新版本"""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=30)
    table.add_column("Installed Version", justify="right", width=15)
    table.add_column("Latest Version", justify="right", width=15)

    # 获取所有已安装库的名称和版本
    installed_packages = sorted((dist.metadata["Name"], dist.version) for dist in distributions())

    for name, installed_version in installed_packages:
        latest_version = get_latest_version(name)
        table.add_row(name, installed_version, latest_version)

    console.print(table)


# 调用函数显示所有已安装库的当前版本和最新版本
# check_all_installed_with_latest()
# 调用函数显示所有已安装库和版本
# check_all_installed()
# check_all_versions()
