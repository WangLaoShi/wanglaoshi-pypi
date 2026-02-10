"""
Python库版本管理模块 (VERSIONS.py)

这个模块提供了机器学习和深度学习相关Python库的版本检查功能，包括：
1. 检查常用ML/DL库的安装状态和版本
2. 显示所有已安装库的版本信息
3. 查询PyPI获取库的最新版本
4. 对比已安装版本和最新版本

主要功能：
- check_all_versions(): 检查常用ML/DL库的版本
- check_all_installed(): 列出所有已安装的库
- check_all_installed_with_latest(): 对比已安装版本和最新版本

使用场景：
- 检查项目依赖库的版本
- 了解环境中安装了哪些库
- 检查是否有可用的库更新
"""

from rich.console import Console  # 富文本控制台输出（美化显示）
from rich.table import Table  # 表格显示工具
from importlib.metadata import version as get_package_version, PackageNotFoundError, distributions  # Python 3.8+ 的包元数据API
import requests  # HTTP请求库，用于查询PyPI
import sys
import platform
import os

# ==================== 简单版本：核心库（最常用的基础库）====================
# 这个版本只包含最核心、最常用的库，适合初学者和快速检查
ml_dl_libraries_simple = {
    # 数据处理基础
    "numpy": {
        "url": "https://numpy.org/",
        "description": "Python数值计算基础库，提供多维数组和数学函数",
        "category": "数据处理"
    },
    "pandas": {
        "url": "https://pandas.pydata.org/",
        "description": "强大的数据分析和处理库，提供DataFrame数据结构",
        "category": "数据处理"
    },
    # 数据可视化
    "matplotlib": {
        "url": "https://matplotlib.org/",
        "description": "Python最流行的绘图库，用于创建各种静态图表",
        "category": "数据可视化"
    },
    "seaborn": {
        "url": "https://seaborn.pydata.org/",
        "description": "基于Matplotlib的统计可视化库，提供更美观的图表",
        "category": "数据可视化"
    },
    # 机器学习
    "scikit-learn": {
        "url": "https://scikit-learn.org/",
        "description": "经典的机器学习库，包含各种分类、回归、聚类算法",
        "category": "机器学习"
    },
    # 深度学习
    "tensorflow": {
        "url": "https://www.tensorflow.org/",
        "description": "Google开发的深度学习框架，工业级应用广泛",
        "category": "深度学习"
    },
    "pytorch": {
        "url": "https://pytorch.org/",
        "description": "Facebook开发的深度学习框架，研究领域主流",
        "category": "深度学习"
    },
    # 科学计算
    "scipy": {
        "url": "https://scipy.org/",
        "description": "科学计算库，提供优化、统计、信号处理等功能",
        "category": "科学计算"
    },
    # 工具库
    "tqdm": {
        "url": "https://tqdm.github.io/",
        "description": "进度条显示库，让循环过程可视化",
        "category": "工具库"
    }
}

# ==================== 复杂版本：完整库列表（包含所有重要库）====================
# 这个版本包含更全面的库列表，适合专业开发者和完整环境检查
ml_dl_libraries = {
    # ========== 数据处理 ==========
    "numpy": {
        "url": "https://numpy.org/",
        "description": "Python数值计算基础库，提供多维数组和数学函数",
        "category": "数据处理"
    },
    "pandas": {
        "url": "https://pandas.pydata.org/",
        "description": "强大的数据分析和处理库，提供DataFrame数据结构",
        "category": "数据处理"
    },
    "polars": {
        "url": "https://www.pola.rs/",
        "description": "高性能数据处理库，比Pandas更快，适合大数据",
        "category": "数据处理"
    },
    "scipy": {
        "url": "https://scipy.org/",
        "description": "科学计算库，提供优化、统计、信号处理等功能",
        "category": "数据处理"
    },
    "statsmodels": {
        "url": "https://www.statsmodels.org/",
        "description": "统计建模和计量经济学库，提供回归分析和时间序列分析",
        "category": "数据处理"
    },
    
    # ========== 数据可视化 ==========
    "matplotlib": {
        "url": "https://matplotlib.org/",
        "description": "Python最流行的绘图库，用于创建各种静态图表",
        "category": "数据可视化"
    },
    "seaborn": {
        "url": "https://seaborn.pydata.org/",
        "description": "基于Matplotlib的统计可视化库，提供更美观的图表",
        "category": "数据可视化"
    },
    "plotly": {
        "url": "https://plotly.com/python/",
        "description": "交互式图表库，支持Web交互和3D可视化",
        "category": "数据可视化"
    },
    "bokeh": {
        "url": "https://bokeh.org/",
        "description": "交互式Web可视化库，适合创建仪表板",
        "category": "数据可视化"
    },
    
    # ========== 机器学习 ==========
    "scikit-learn": {
        "url": "https://scikit-learn.org/",
        "description": "经典的机器学习库，包含各种分类、回归、聚类算法",
        "category": "机器学习"
    },
    "xgboost": {
        "url": "https://xgboost.readthedocs.io/",
        "description": "高效的梯度提升库，在Kaggle等比赛中广泛使用",
        "category": "机器学习"
    },
    "lightgbm": {
        "url": "https://lightgbm.readthedocs.io/",
        "description": "微软开发的高效梯度提升决策树库，速度快内存占用小",
        "category": "机器学习"
    },
    "catboost": {
        "url": "https://catboost.ai/",
        "description": "Yandex开发的梯度提升库，特别适合处理分类特征",
        "category": "机器学习"
    },
    "imbalanced-learn": {
        "url": "https://imbalanced-learn.org/",
        "description": "处理不平衡数据集的库，提供过采样和欠采样方法",
        "category": "机器学习"
    },
    
    # ========== 深度学习 ==========
    "tensorflow": {
        "url": "https://www.tensorflow.org/",
        "description": "Google开发的深度学习框架，工业级应用广泛",
        "category": "深度学习"
    },
    "keras": {
        "url": "https://keras.io/",
        "description": "基于TensorFlow的高级深度学习API，简单易用",
        "category": "深度学习"
    },
    "pytorch": {
        "url": "https://pytorch.org/",
        "description": "Facebook开发的深度学习框架，研究领域主流",
        "category": "深度学习"
    },
    "torchvision": {
        "url": "https://pytorch.org/vision/",
        "description": "PyTorch的计算机视觉工具库，提供数据集和预训练模型",
        "category": "深度学习"
    },
    "torchaudio": {
        "url": "https://pytorch.org/audio/",
        "description": "PyTorch的音频处理库，提供音频数据集和工具",
        "category": "深度学习"
    },
    "onnx": {
        "url": "https://onnx.ai/",
        "description": "开放式神经网络交换格式，用于模型转换和部署",
        "category": "深度学习"
    },
    "onnxruntime": {
        "url": "https://onnxruntime.ai/",
        "description": "ONNX模型推理运行时，支持多平台部署",
        "category": "深度学习"
    },
    
    # ========== 自然语言处理 ==========
    "nltk": {
        "url": "https://www.nltk.org/",
        "description": "自然语言处理工具包，提供文本处理和语料库",
        "category": "自然语言处理"
    },
    "spacy": {
        "url": "https://spacy.io/",
        "description": "高效的自然语言处理库，提供预训练模型和NLP管道",
        "category": "自然语言处理"
    },
    "transformers": {
        "url": "https://huggingface.co/docs/transformers",
        "description": "Hugging Face的预训练模型库，支持BERT、GPT等模型",
        "category": "自然语言处理"
    },
    "sentence-transformers": {
        "url": "https://www.sbert.net/",
        "description": "用于句子和文本嵌入的库，基于Transformers",
        "category": "自然语言处理"
    },
    
    # ========== 计算机视觉 ==========
    "opencv-python": {
        "url": "https://opencv.org/",
        "description": "OpenCV的Python绑定，强大的图像和视频处理库",
        "category": "计算机视觉"
    },
    "Pillow": {
        "url": "https://python-pillow.org/",
        "description": "PIL的现代替代品，Python图像处理标准库",
        "category": "计算机视觉"
    },
    "scikit-image": {
        "url": "https://scikit-image.org/",
        "description": "基于scipy的图像处理库，提供图像算法和工具",
        "category": "计算机视觉"
    },
    "albumentations": {
        "url": "https://albumentations.ai/",
        "description": "快速灵活的图像增强库，支持多种数据增强技术",
        "category": "计算机视觉"
    },
    "imageio": {
        "url": "https://imageio.readthedocs.io/",
        "description": "图像I/O库，支持多种图像格式的读写",
        "category": "计算机视觉"
    },
    
    # ========== 强化学习 ==========
    "gym": {
        "url": "https://www.gymlibrary.ml/",
        "description": "OpenAI开发的强化学习环境标准库",
        "category": "强化学习"
    },
    "gymnasium": {
        "url": "https://gymnasium.farama.org/",
        "description": "Gym的维护版本，提供更多环境和功能",
        "category": "强化学习"
    },
    "stable-baselines3": {
        "url": "https://stable-baselines3.readthedocs.io/",
        "description": "强化学习算法实现库，提供多种RL算法",
        "category": "强化学习"
    },
    
    # ========== 分布式计算 ==========
    "ray": {
        "url": "https://www.ray.io/",
        "description": "分布式计算框架，用于加速ML训练和分布式应用",
        "category": "分布式计算"
    },
    "dask": {
        "url": "https://dask.org/",
        "description": "并行计算库，支持大规模数据处理，类似分布式Pandas",
        "category": "分布式计算"
    },
    "joblib": {
        "url": "https://joblib.readthedocs.io/",
        "description": "并行计算与模型持久化库，scikit-learn的依赖",
        "category": "分布式计算"
    },
    
    # ========== 机器学习管理 ==========
    "mlflow": {
        "url": "https://mlflow.org/",
        "description": "机器学习生命周期管理平台，支持实验跟踪和模型部署",
        "category": "机器学习管理"
    },
    "wandb": {
        "url": "https://wandb.ai/",
        "description": "Weights & Biases实验跟踪工具，提供可视化仪表板",
        "category": "机器学习管理"
    },
    "optuna": {
        "url": "https://optuna.org/",
        "description": "自动超参数优化框架，支持多种优化算法",
        "category": "机器学习管理"
    },
    "hydra": {
        "url": "https://hydra.cc/",
        "description": "配置管理框架，简化复杂应用的配置",
        "category": "机器学习管理"
    },
    "pycaret": {
        "url": "https://pycaret.org/",
        "description": "低代码机器学习库，自动化ML工作流程",
        "category": "机器学习管理"
    },
    "tensorboard": {
        "url": "https://www.tensorflow.org/tensorboard",
        "description": "TensorFlow的可视化工具，用于监控训练过程",
        "category": "机器学习管理"
    },
    
    # ========== 音频处理 ==========
    "librosa": {
        "url": "https://librosa.org/",
        "description": "音频和音乐信号分析库，提供丰富的音频处理工具",
        "category": "音频处理"
    },
    "mir_eval": {
        "url": "https://craffel.github.io/mir_eval/",
        "description": "音乐信息检索评估库，支持多种MIR任务的评估",
        "category": "音频处理"
    },
    "soundfile": {
        "url": "https://pysoundfile.readthedocs.io/",
        "description": "音频文件读写库，支持多种音频格式",
        "category": "音频处理"
    },
    
    # ========== Web应用和API ==========
    "streamlit": {
        "url": "https://streamlit.io/",
        "description": "快速构建数据科学Web应用的框架",
        "category": "Web应用"
    },
    "fastapi": {
        "url": "https://fastapi.tiangolo.com/",
        "description": "现代高性能Web框架，用于构建API",
        "category": "Web应用"
    },
    "gradio": {
        "url": "https://www.gradio.app/",
        "description": "快速创建机器学习模型演示界面的库",
        "category": "Web应用"
    },
    
    # ========== Jupyter和开发工具 ==========
    "jupyter": {
        "url": "https://jupyter.org/",
        "description": "Jupyter Notebook和JupyterLab，交互式开发环境",
        "category": "开发工具"
    },
    "ipython": {
        "url": "https://ipython.org/",
        "description": "增强的Python交互式shell，Jupyter的核心",
        "category": "开发工具"
    },
    "ipywidgets": {
        "url": "https://ipywidgets.readthedocs.io/",
        "description": "Jupyter交互式小部件，用于创建交互式界面",
        "category": "开发工具"
    },
    
    # ========== 工具库 ==========
    "tqdm": {
        "url": "https://tqdm.github.io/",
        "description": "进度条显示库，让循环过程可视化",
        "category": "工具库"
    },
    "requests": {
        "url": "https://requests.readthedocs.io/",
        "description": "HTTP库，用于发送网络请求",
        "category": "工具库"
    },
    "python-dotenv": {
        "url": "https://github.com/theskumar/python-dotenv",
        "description": "环境变量管理库，从.env文件加载配置",
        "category": "工具库"
    }
}

# ==================== 默认使用复杂版本 ====================
# 为了保持向后兼容，ml_dl_libraries默认指向复杂版本
# 可以通过参数选择使用简单版本或复杂版本

def get_libraries(version='full'):
    """
    获取库字典（简单版本或复杂版本）
    
    参数:
        version: 版本选择
                - 'simple': 返回简单版本（只包含核心库）
                - 'full': 返回复杂版本（包含所有库，默认）
        
    返回:
        库字典
    """
    if version == 'simple':
        return ml_dl_libraries_simple
    else:
        return ml_dl_libraries

def check_versions(libraries=None, version='full'):
    """
    检查指定库的版本信息
    
    这个函数会遍历库字典，检查每个库是否已安装，并获取其版本号。
    注意：字典中的键应该是PyPI包名（pip install时使用的名称），
    因为version()函数使用PyPI包名来查找版本。
    
    参数:
        libraries: 要检查的库字典
                   - None: 使用默认库字典（根据version参数选择）
                   - dict: 自定义库字典
        version: 版本选择（当libraries为None时生效）
                - 'simple': 使用简单版本（只包含核心库，约9个）
                - 'full': 使用复杂版本（包含所有库，约60+个，默认）
        
    返回:
        一个字典，键为库名，值为版本号字符串或'Not installed'
        
    示例:
        # 使用复杂版本（默认）
        versions = check_versions()
        
        # 使用简单版本
        versions = check_versions(version='simple')
        
        # 使用自定义库字典
        custom_libs = {'numpy': {...}, 'pandas': {...}}
        versions = check_versions(libraries=custom_libs)
    """
    # 如果没有提供库字典，根据version参数选择
    if libraries is None:
        libraries = get_libraries(version)
    versions = {}
    # 遍历库字典中的所有库名
    for lib in libraries.keys():
        try:
            # 使用importlib.metadata.version()获取库的版本号
            # 这个方法使用PyPI包名（pip install时使用的名称）
            # 例如：'opencv-python'（不是'cv2'），'Pillow'（不是'PIL'）
            lib_version = get_package_version(lib)
            versions[lib] = lib_version
        except PackageNotFoundError:
            # 如果包未安装，标记为'Not installed'
            versions[lib] = 'Not installed'
        except ModuleNotFoundError:
            # 如果模块未找到，也标记为'Not installed'
            versions[lib] = 'Not installed'
    return versions


def show_env_info() -> None:
    """
    显示当前Python环境的关键信息（操作系统、Python版本等）

    这个函数不会检查任何第三方库，只用于了解当前运行环境。

    显示内容包含：
    - 操作系统及版本
    - Python版本与实现
    - Python可执行文件路径
    - 是否在虚拟环境/conda环境中
    """
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Item", style="bold", width=24)
    table.add_column("Value", style="dim")

    # 操作系统信息
    table.add_row("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
    table.add_row("Machine", platform.machine())

    # Python信息
    table.add_row("Python Version", platform.python_version())
    table.add_row("Python Implementation", platform.python_implementation())
    table.add_row("Python Executable", sys.executable)

    # 虚拟环境 / Conda
    venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    table.add_row("Virtual Env", "Yes" if venv else "No")

    if conda_env:
        table.add_row("Conda Env", conda_env)

    console.print(table)


def check_all_versions(
    all_columns: bool = False,
    version: str = 'full',
    show_latest: bool = False,
    only_problems: bool = False
) -> None:
    """
    以表格形式显示所有常用ML/DL库的版本信息
    
    这个函数会创建一个美观的表格，显示常用机器学习和深度学习库的：
    - 库名称
    - 描述
    - 版本号（如果已安装）
    - 可选：网站链接和分类
    
    参数:
        all_columns: 是否显示所有列（包括网站和分类）
                    - True: 显示库名、描述、网站、分类、版本
                    - False: 只显示库名、描述、版本（默认）
        version: 版本选择
                - 'simple': 使用简单版本（只包含核心库，约9个）
                - 'full': 使用复杂版本（包含所有库，约60+个，默认）
        show_latest: 是否同时显示PyPI上的最新版本并给出状态
                    - True: 额外显示Latest和Status两列（需要网络）
                    - False: 只显示本地安装版本
        only_problems: 仅显示有问题的库
                    - True: 只显示未安装或不是最新版本的库
                    - False: 显示所有库
                    
    返回:
        None（直接打印表格）
        
    使用场景：
    - 快速查看环境中安装了哪些ML/DL库
    - 检查库的版本是否符合要求
    - 了解库的用途和分类
    - 对比本地版本和PyPI最新版本，发现需要更新的库
    
    示例:
        # 显示复杂版本的所有库
        check_all_versions()
        
        # 显示简单版本的核心库
        check_all_versions(version='simple')
        
        # 显示所有列（包括网站和分类）
        check_all_versions(all_columns=True)

        # 显示最新版本并标记状态
        check_all_versions(show_latest=True)

        # 只显示未安装或需要更新的库
        check_all_versions(show_latest=True, only_problems=True)
    """
    # 根据version参数获取对应的库字典
    libraries = get_libraries(version)
    # 获取所有库的版本信息（本地安装情况）
    installed_versions = check_versions(libraries=libraries)

    # 如果需要最新版本信息，额外从PyPI获取
    latest_versions = {}
    status_map = {}
    if show_latest:
        for lib, installed in installed_versions.items():
            latest = get_latest_version(lib)
            latest_versions[lib] = latest
            if installed == 'Not installed':
                status = 'Not installed'
            elif latest in ('Error', 'Not found'):
                status = 'Unknown'
            elif installed == latest:
                status = 'Up-to-date'
            else:
                status = 'Outdated'
            status_map[lib] = status

    # 创建Rich控制台对象（用于美化输出）
    console = Console()
    # 创建表格对象
    table = Table(show_header=True, header_style="bold magenta")
    
    # 添加表格列
    table.add_column("Library", style="bold", width=20)  # 库名
    table.add_column("Description", justify="left", width=30)  # 描述
    
    # 如果all_columns为True，添加额外的列
    if all_columns:
        table.add_column("Website", justify="left", width=32)  # 网站链接
        table.add_column("Category", justify="left", width=18)  # 分类
    
    table.add_column("Installed", justify="right", width=14)  # 已安装版本
    if show_latest:
        table.add_column("Latest", justify="right", width=14)  # 最新版本
        table.add_column("Status", justify="left", width=14)  # 状态
    
    # 遍历库，构造行
    for lib, installed in installed_versions.items():
        desc = libraries[lib]["description"]
        site = libraries[lib]["url"]
        category = libraries[lib]["category"]

        # 计算状态（如果需要）
        latest = latest_versions.get(lib, '') if show_latest else ''
        status = status_map.get(lib, '') if show_latest else ''

        # 只显示有问题的条目
        if only_problems and show_latest:
            if status not in ('Not installed', 'Outdated'):
                continue

        # 美化版本显示
        installed_display = installed
        latest_display = latest
        status_display = status

        if installed != 'Not installed':
            installed_display = f"[green]{installed}[/green]"
        else:
            installed_display = f"[red]{installed}[/red]"

        if show_latest and latest not in ('Error', 'Not found', ''):
            latest_display = f"[cyan]{latest}[/cyan]"

        if show_latest:
            if status == 'Up-to-date':
                status_display = "[green]Up-to-date[/green]"
            elif status == 'Outdated':
                status_display = "[yellow]Outdated[/yellow]"
            elif status == 'Not installed':
                status_display = "[red]Not installed[/red]"
            else:
                status_display = status or ''

        # 添加行
        if all_columns:
            if show_latest:
                table.add_row(lib, desc, site, category, installed_display, latest_display, status_display)
            else:
                table.add_row(lib, desc, site, category, installed_display)
        else:
            if show_latest:
                table.add_row(lib, desc, installed_display, latest_display, status_display)
            else:
                table.add_row(lib, desc, installed_display)
    
    # 打印表格
    console.print(table)


def check_all_installed():
    """
    检查并显示所有已安装的Python库和版本信息
    
    这个函数会列出当前Python环境中所有已安装的包及其版本号。
    与check_all_versions()不同，这个函数会显示所有包，而不仅仅是ML/DL库。
    
    返回:
        None（直接打印表格）
        
    使用场景：
    - 查看环境中安装了哪些包
    - 导出依赖列表
    - 检查包的安装情况
    
    注意：
    - 这个函数会列出所有已安装的包，可能数量很多
    - 输出会按包名排序
    """
    # 创建Rich控制台和表格对象
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=30)  # 库名
    table.add_column("Version", justify="right", width=15)  # 版本

    # 获取所有已安装包的信息
    # distributions() 返回所有已安装包的Distribution对象
    # dist.metadata["Name"] 获取包名
    # dist.version 获取版本号
    # sorted() 按包名排序
    installed_packages = sorted((dist.metadata["Name"], dist.version) for dist in distributions())

    # 将每个包的信息添加到表格中
    for name, pkg_version in installed_packages:
        table.add_row(name, pkg_version)

    # 打印表格
    console.print(table)


def get_latest_version(package_name):
    """
    查询PyPI获取指定包的最新版本
    
    这个函数会访问PyPI（Python Package Index）的API，
    查询指定包的最新版本号。
    
    参数:
        package_name: 要查询的包名（例如: "numpy", "pandas"）
        
    返回:
        版本号字符串，例如: "1.21.0"
        "Not found": 包不存在
        "Error": 网络错误或其他错误
        
    工作原理：
    - PyPI提供了JSON API：https://pypi.org/pypi/{package_name}/json
    - 发送HTTP GET请求获取包的元数据
    - 从JSON响应中提取版本信息
    
    示例:
        version = get_latest_version("numpy")
        print(version)  # 输出: "1.21.0"
    """
    # 构建PyPI API的URL
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        # 发送GET请求，设置5秒超时
        response = requests.get(url, timeout=5)
        # 检查响应状态码
        if response.status_code == 200:
            # 解析JSON响应，提取版本信息
            return response.json()["info"]["version"]
        else:
            # 如果状态码不是200，说明包不存在
            return "Not found"
    except requests.RequestException:
        # 捕获网络请求异常（连接超时、网络错误等）
        return "Error"


def check_all_installed_with_latest():
    """
    显示所有已安装库的当前版本和最新版本（对比）
    
    这个函数会：
    1. 列出所有已安装的包及其当前版本
    2. 查询PyPI获取每个包的最新版本
    3. 以表格形式对比显示
    
    返回:
        None（直接打印表格）
        
    使用场景：
    - 检查哪些包需要更新
    - 了解包的版本情况
    - 决定是否更新包
    
    注意：
    - 这个函数会查询PyPI，需要网络连接
    - 查询所有包可能需要较长时间
    - 某些包可能查询失败（网络问题、包不存在等）
    """
    # 创建Rich控制台和表格对象
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Library", style="dim", width=30)  # 库名
    table.add_column("Installed Version", justify="right", width=15)  # 已安装版本
    table.add_column("Latest Version", justify="right", width=15)  # 最新版本

    # 获取所有已安装包的名称和版本
    installed_packages = sorted((dist.metadata["Name"], dist.version) for dist in distributions())

    # 遍历每个已安装的包
    for name, installed_version in installed_packages:
        # 查询PyPI获取最新版本
        latest_version = get_latest_version(name)
        # 添加到表格中
        table.add_row(name, installed_version, latest_version)

    # 打印表格
    console.print(table)


# ==================== 使用示例 ====================
# 
# 1. 显示简单版本（只包含核心库，约9个）
# check_all_versions(version='simple')
#
# 2. 显示复杂版本（包含所有库，约60+个，默认）
# check_all_versions()
# check_all_versions(version='full')
#
# 3. 显示所有列（包括网站和分类）
# check_all_versions(all_columns=True)
#
# 4. 显示所有已安装的库
# check_all_installed()
#
# 5. 对比已安装版本和最新版本
# check_all_installed_with_latest()
