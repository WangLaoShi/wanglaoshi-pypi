"""兼容 shim：实现已迁至 ``wanglaoshi.analyze.wqe``（legacy）。"""
from __future__ import annotations

import sys

from .analyze import wqe as _impl

sys.modules[__name__] = _impl
