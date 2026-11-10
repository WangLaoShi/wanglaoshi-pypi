"""兼容 shim：实现已迁至 ``wanglaoshi.ml.core``。

推荐::

    from wanglaoshi.ml import DataPreprocessor, MLModel
"""
from __future__ import annotations

import sys

from .ml import core as _impl

sys.modules[__name__] = _impl
