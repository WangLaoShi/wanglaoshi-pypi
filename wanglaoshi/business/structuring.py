"""D02 data-structuring：列名标准化与基础类型规整（MVP）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .contracts import StageResult


def _normalize_column(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^0-9a-z_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def structure_dataframe(
    df: pd.DataFrame,
    *,
    column_map: Optional[Mapping[str, str]] = None,
    dtype_map: Optional[Mapping[str, str]] = None,
    output_path: Optional[str | Path] = None,
) -> StageResult:
    """标准化列名、可选重命名与类型转换。

    - column_map: 原始列名 → 目标列名（在 normalize 之前应用）
    - dtype_map: 目标列名 → dtype 字符串（如 float64 / int64 / string）
    """
    warnings: List[str] = []
    block_reasons: List[str] = []
    rules = ["D02.normalize_columns"]
    work = df.copy()
    original_columns = list(work.columns)

    if column_map:
        rules.append("D02.column_map")
        missing = [k for k in column_map if k not in work.columns]
        if missing:
            warnings.append(f"column_map keys not found: {missing}")
        work = work.rename(columns={k: v for k, v in column_map.items() if k in work.columns})

    new_cols = []
    seen: Dict[str, int] = {}
    for c in work.columns:
        base = _normalize_column(c)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        new_cols.append(base)
    mapping = dict(zip(original_columns, new_cols)) if not column_map else dict(zip(list(df.columns), new_cols))
    work.columns = new_cols

    converted: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    if dtype_map:
        rules.append("D02.dtype_map")
        for col, dtype in dtype_map.items():
            if col not in work.columns:
                warnings.append(f"dtype_map column missing: {col}")
                continue
            try:
                if dtype in {"string", "str"}:
                    work[col] = work[col].astype("string")
                elif dtype.startswith("datetime"):
                    work[col] = pd.to_datetime(work[col], errors="coerce")
                else:
                    work[col] = work[col].astype(dtype)
                converted[col] = dtype
            except Exception as e:  # noqa: BLE001
                failed[col] = str(e)
                warnings.append(f"dtype convert failed for {col}: {e}")

    outputs: List[str] = []
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out, index=False)
        outputs.append(str(out))

    status = "blocked" if block_reasons else ("warning" if warnings else "passed")
    result = StageResult(
        stage_id="D02_structuring",
        status=status,
        data={
            "n_rows": int(len(work)),
            "n_cols": int(work.shape[1]),
            "column_mapping": mapping,
            "converted_dtypes": converted,
            "failed_dtypes": failed,
            "columns": list(work.columns),
        },
        rules=rules,
        warnings=warnings,
        block_reasons=block_reasons,
        output_paths=outputs,
    )
    # DataFrame 挂在结果对象上，避免塞进可序列化 data
    result.frame = work  # type: ignore[attr-defined]
    return result


def structure_file(
    source_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> StageResult:
    path = Path(source_path)
    if not path.exists():
        return StageResult(
            stage_id="D02_structuring",
            status="blocked",
            block_reasons=[f"file not found: {path}"],
            input_paths=[str(path)],
        )
    df = pd.read_csv(path)
    result = structure_dataframe(df, output_path=output_path, **kwargs)
    result.input_paths = [str(path)]
    # 去掉不可 JSON 序列化的 frame 再写 sidecar 时由调用方处理
    return result
