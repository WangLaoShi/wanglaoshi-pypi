"""兼容 shim：实现已迁至 ``wanglaoshi._testing.testing_utils``。"""
from __future__ import annotations

import sys

from ._testing import testing_utils as _impl

sys.modules[__name__] = _impl
