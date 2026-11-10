"""兼容 shim：实现已迁至 ``wanglaoshi.runtime.web_getter``。"""
from __future__ import annotations

import sys

from .runtime import web_getter as _impl

sys.modules[__name__] = _impl
