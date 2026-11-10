"""兼容 shim：实现已迁至 ``wanglaoshi.runtime.jupyter_env``。"""
from __future__ import annotations

import sys

from .runtime import jupyter_env as _impl

sys.modules[__name__] = _impl
