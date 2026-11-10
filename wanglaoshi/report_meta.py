"""兼容 shim：实现已迁至 ``wanglaoshi._testing.report_meta``。"""
from __future__ import annotations

import sys

from ._testing import report_meta as _impl

sys.modules[__name__] = _impl
