"""兼容 shim：实现已迁至 ``wanglaoshi.runtime.ready``。"""

from __future__ import annotations

from .runtime.ready import ready

__all__ = ["ready"]
