"""C11：缺数据策略——路径不存在则明确跳过（非假绿）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest


def require_data_path(path: str | Path, reason: Optional[str] = None) -> Path:
    """若文件/目录不存在，pytest.skip 并说明原因。"""
    p = Path(path)
    if not p.exists():
        msg = reason or f"optional data missing (download separately): {p}"
        pytest.skip(msg)
    return p


def optional_data_available(path: str | Path) -> bool:
    return Path(path).exists()
