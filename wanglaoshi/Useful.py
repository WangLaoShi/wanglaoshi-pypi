"""兼容 shim：实现已迁至 ``wanglaoshi.runtime.useful``。"""
from __future__ import annotations

import sys

from .runtime import useful as _impl

sys.modules[__name__] = _impl
