"""测试与校验辅助（C04）：数值容差 + 结构精确断言。"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def assert_close(
    actual: float,
    expected: float,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    name: str = "value",
) -> None:
    """数值接近断言（明确容差）。"""
    a, e = float(actual), float(expected)
    if not np.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(f"{name}: {a} !~ {e} (atol={atol}, rtol={rtol})")


def assert_metrics_close(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    """指标字典按键比对；缺失键直接失败。"""
    for k, exp in expected.items():
        if k not in actual:
            raise AssertionError(f"missing metric key: {k}")
        assert_close(actual[k], exp, atol=atol, rtol=rtol, name=k)


def assert_frame_contract(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    n_rows: Optional[int] = None,
    ordered_columns: bool = True,
) -> None:
    """字段 / 行数 /（可选）列顺序的精确断言。"""
    if n_rows is not None and len(df) != n_rows:
        raise AssertionError(f"n_rows: {len(df)} != {n_rows}")
    if columns is not None:
        cols = list(columns)
        if ordered_columns:
            if list(df.columns) != cols:
                raise AssertionError(f"columns order/content: {list(df.columns)} != {cols}")
        else:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise AssertionError(f"missing columns: {missing}")
