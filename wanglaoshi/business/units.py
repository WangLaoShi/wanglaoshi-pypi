"""单位换算（S1 / P1）。

用显式因子表做列级换算，避免「万元/元」混用却直接加总。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

from .contracts import StageResult

# (from, to) -> multiply factor to convert from → to
DEFAULT_FACTORS: Dict[Tuple[str, str], float] = {
    ("yuan", "wan"): 1.0 / 10_000.0,
    ("wan", "yuan"): 10_000.0,
    ("fen", "yuan"): 1.0 / 100.0,
    ("yuan", "fen"): 100.0,
    ("g", "kg"): 1.0 / 1000.0,
    ("kg", "g"): 1000.0,
    ("ml", "l"): 1.0 / 1000.0,
    ("l", "ml"): 1000.0,
}


def convert_units(
    df: pd.DataFrame,
    col: str,
    *,
    from_unit: str,
    to_unit: str,
    out_col: Optional[str] = None,
    factors: Optional[Mapping[Tuple[str, str], float]] = None,
    stage_id: str = "D02_units",
) -> StageResult:
    """将 ``col`` 从 ``from_unit`` 换算到 ``to_unit``。

    新列默认 ``{col}__{to_unit}``；同单位则复制。
    """
    if col not in df.columns:
        return StageResult(
            stage_id=stage_id,
            status="blocked",
            block_reasons=[f"column missing: {col}"],
        )
    fu, tu = from_unit.strip().lower(), to_unit.strip().lower()
    table = dict(DEFAULT_FACTORS)
    if factors:
        table.update({(a.lower(), b.lower()): float(v) for (a, b), v in factors.items()})

    if fu == tu:
        factor = 1.0
    elif (fu, tu) in table:
        factor = table[(fu, tu)]
    else:
        return StageResult(
            stage_id=stage_id,
            status="blocked",
            block_reasons=[f"no conversion factor for {fu!r} -> {tu!r}"],
            data={"known_pairs": sorted(f"{a}->{b}" for a, b in table)},
        )

    work = df.copy()
    target = out_col or f"{col}__{tu}"
    work[target] = pd.to_numeric(work[col], errors="coerce") * factor
    warnings: List[str] = []
    n_na = int(work[target].isna().sum())
    if n_na:
        warnings.append(f"{n_na} values became NaN after numeric coerce")

    return StageResult(
        stage_id=stage_id,
        status="warning" if warnings else "passed",
        data={
            "from_unit": fu,
            "to_unit": tu,
            "factor": factor,
            "out_col": target,
            "frame": work,
        },
        rules=["D02.convert_units"],
        warnings=warnings,
    )
