"""兼容 shim：实现已迁至 ``wanglaoshi.analytics.timeseries``。"""
from __future__ import annotations

import sys

from .analytics import timeseries as _impl

sys.modules[__name__] = _impl
