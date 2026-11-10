"""兼容 shim：实现已迁至 ``wanglaoshi._testing.score_floors``。"""
from __future__ import annotations

import sys

from ._testing import score_floors as _impl

sys.modules[__name__] = _impl
