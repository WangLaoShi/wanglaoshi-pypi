"""EDA / 报告分析门面（零破坏：主实现仍在 ``Analyzer``）。

推荐::

    from wanglaoshi.analyze import DataAnalyzer, analyze_data

说明：
- ``Analyzer_Plain`` / ``wqe`` 为历史并行路径；导入会发 DeprecationWarning
  （``WANGLAOSHI_SILENCE_LEGACY=1`` 可关）。新代码请优先本门面。
"""

from __future__ import annotations

from ..Analyzer import (
    DataAnalyzer,
    DataProcessor,
    ReportGenerator,
    Visualizer,
    analyze_data,
    analyze_multiple_files,
    analyze_notebook,
    load_data,
)

__all__ = [
    "DataAnalyzer",
    "DataProcessor",
    "Visualizer",
    "ReportGenerator",
    "load_data",
    "analyze_data",
    "analyze_multiple_files",
    "analyze_notebook",
]
