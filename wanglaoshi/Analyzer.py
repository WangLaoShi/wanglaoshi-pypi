"""兼容 shim：实现已迁至 ``wanglaoshi.analyze.core``。

推荐::

    from wanglaoshi.analyze import DataAnalyzer
"""
from __future__ import annotations

import sys

from .analyze import core as _impl

sys.modules[__name__] = _impl
