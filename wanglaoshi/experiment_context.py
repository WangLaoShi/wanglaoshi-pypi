"""兼容 shim：实现已迁至 ``wanglaoshi._testing.experiment_context``。"""
from __future__ import annotations

import sys

from ._testing import experiment_context as _impl

sys.modules[__name__] = _impl
