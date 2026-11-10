"""Legacy / soft-deprecate helpers（Phase 3）。

默认发出 ``DeprecationWarning``；设环境变量关闭刷屏::

    WANGLAOSHI_SILENCE_LEGACY=1
"""

from __future__ import annotations

import os
import warnings

_ENV = "WANGLAOSHI_SILENCE_LEGACY"


def silence_legacy() -> bool:
    """True 表示关闭 legacy 弃用警告。"""
    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def warn_legacy(name: str, *, replacement: str, stacklevel: int = 2) -> None:
    """对 legacy 模块/路径发出一次可过滤的 DeprecationWarning。"""
    if silence_legacy():
        return
    warnings.warn(
        f"{name} 为历史路径（legacy）。新代码请改用 {replacement}。"
        f" 关闭警告：export {_ENV}=1。"
        f" 详见 docs/ARCHITECTURE.md。",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
