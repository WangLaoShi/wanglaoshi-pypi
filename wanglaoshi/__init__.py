"""
wanglaoshi 顶层导出 (API)

为了让 README 中的用法与实际安装后的体验一致，这里整理了推荐的对外 API。

常见用法示例：

    from wanglaoshi import JupyterEnv as JE
    from wanglaoshi import VERSIONS as V
    from wanglaoshi import Analyzer
    from wanglaoshi import CompetitionTools as CT
    from wanglaoshi import FeatureEngineering as FE

    from wanglaoshi import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook

模块级别导出：
- JupyterEnv
- VERSIONS
- Analyzer
- Analyzer_Plain
- CompetitionTools
- FeatureEngineering
- MLDL
- JupyterFont
- Useful
- WebGetter

函数 / 类级别导出：
- DataAnalyzer
- analyze_data
- analyze_multiple_files
- analyze_notebook
"""

# 模块导出（用于 from wanglaoshi import VERSIONS as V 这类用法）
from . import JupyterEnv
from . import VERSIONS
from . import Analyzer
from . import Analyzer_Plain
from . import CompetitionTools
from . import FeatureEngineering
from . import MLDL
from . import JupyterFont
from . import Useful
from . import WebGetter

# 直接导出常用类和函数（方便用户少写一层模块前缀）
from .Analyzer import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook

__all__ = [
    # 模块
    "JupyterEnv",
    "VERSIONS",
    "Analyzer",
    "Analyzer_Plain",
    "CompetitionTools",
    "FeatureEngineering",
    "MLDL",
    "JupyterFont",
    "Useful",
    "WebGetter",
    # 类 / 函数
    "DataAnalyzer",
    "analyze_data",
    "analyze_multiple_files",
    "analyze_notebook",
]
