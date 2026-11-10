"""兼容 shim：实现已迁至 ``wanglaoshi.analytics.stats``。"""

from __future__ import annotations

import sys

from ..analytics import stats as _impl

# 将本包命名空间换成真实实现包
sys.modules[__name__] = _impl
