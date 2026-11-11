"""统计分析与试验设计（位于 analytics 公开接口下）。

推荐::

    from wanglaoshi.analytics import StatsToolkit, ExperimentDesign
    # 或
    from wanglaoshi.stats import StatsToolkit, ExperimentDesign
"""

from .toolkit import StatsToolkit
from .experiment import ExperimentDesign

__all__ = ["StatsToolkit", "ExperimentDesign"]
