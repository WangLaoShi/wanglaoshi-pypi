"""兼容 shim：实现已迁至 ``wanglaoshi._testing.data_registry``。"""
from __future__ import annotations

import sys

from ._testing import data_registry as _impl

sys.modules[__name__] = _impl
