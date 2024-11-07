import importlib
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # Python 3.7 及以下
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
    }
}

def check_versions(libraries=ml_dl_libraries):
    """显示指定库的版本信息"""
    versions = {}
    for lib in libraries.keys():
        try:
            # module = importlib.import_module(lib)
            lib_version = importlib.metadata.version(lib)
            versions[lib] = lib_version
        except PackageNotFoundError:
            versions[lib] = 'Not installed'
        except ModuleNotFoundError:
            versions[lib] = 'Not installed'
    return versions

def check_all_versions(all_columns=False):
    """显示所有常用库的版本信息"""
    results = check_versions(ml_dl_libraries)
    from rich.console import Console
    from rich.table import Column, Table

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

# check_all_versions()
