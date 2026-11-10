"""兼容 shim：实现已迁至 ``wanglaoshi.analytics.ranking``。"""
from __future__ import annotations

import sys

from .analytics import ranking as _impl

sys.modules[__name__] = _impl
