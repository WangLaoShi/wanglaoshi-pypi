"""兼容 shim：实现已迁至 ``wanglaoshi.analytics.anomaly``。"""
from __future__ import annotations

import sys

from .analytics import anomaly as _impl

sys.modules[__name__] = _impl
