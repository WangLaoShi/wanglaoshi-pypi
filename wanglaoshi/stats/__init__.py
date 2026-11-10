"""统计分析与试验设计最小工具（P0）。

推荐用法::

    from wanglaoshi.stats import StatsToolkit, ExperimentDesign
    StatsToolkit.ttest_ind(a, b)
    ExperimentDesign.two_proportion_test(x1, n1, x2, n2)
"""

from .toolkit import StatsToolkit
from .experiment import ExperimentDesign

__all__ = ["StatsToolkit", "ExperimentDesign"]
