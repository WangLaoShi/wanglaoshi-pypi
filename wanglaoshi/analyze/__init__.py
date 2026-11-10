"""EDA / 报告分析门面。

推荐::

    from wanglaoshi.analyze import DataAnalyzer, analyze_data

说明：legacy ``analyze.plain`` / ``analyze.wqe`` 仅过渡保留（导入有 DeprecationWarning）。
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
