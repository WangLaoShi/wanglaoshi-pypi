"""兼容 shim：实现已迁至 ``wanglaoshi.features.engineering``。

推荐::

    from wanglaoshi.features import AdvancedFeatureEngineer
"""
from __future__ import annotations

import sys

from .features import engineering as _impl

sys.modules[__name__] = _impl
