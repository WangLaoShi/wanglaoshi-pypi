"""EDA / 报告分析门面。

推荐::

    from wanglaoshi.analyze import DataAnalyzer, analyze_data

说明：
- legacy 实现：``analyze.plain`` / ``analyze.wqe``（根上 ``Analyzer_Plain`` / ``wqe`` 为 shim）；
  导入会发 DeprecationWarning（``WANGLAOSHI_SILENCE_LEGACY=1`` 可关）。
"""

from __future__ import annotations

from .core import (
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
