"""兼容 shim：实现已迁至 ``wanglaoshi._testing.regression_summary``。"""
from __future__ import annotations

import sys

from ._testing import regression_summary as _impl

sys.modules[__name__] = _impl
