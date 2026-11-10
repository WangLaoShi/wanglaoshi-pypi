"""兼容 shim：实现已迁至 ``wanglaoshi._testing.asserting``。"""
from __future__ import annotations

import sys

from ._testing import asserting as _impl

sys.modules[__name__] = _impl
