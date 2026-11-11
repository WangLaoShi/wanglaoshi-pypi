"""D02 data-structuring：列名标准化与基础类型规整（MVP）。

常见痛点：Excel/业务导出的列名带空格、括号、中英文混杂。
本模块会把列名尽量变成稳定标识，并支持按配置做类型转换。
注意：单位换算、多表关联膨胀检查仍属后续增强项。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import pandas as pd

from .contracts import StageResult

_SUPPORTED_TABULAR = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}


def _normalize_column(name: str) -> str:
    """保留 Unicode 字母数字；空白与符号压成下划线。"""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def _unique_name(base: str, used: Set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


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

    used: Set[str] = set()
    new_cols: List[str] = []
    for c in work.columns:
        new_cols.append(_unique_name(_normalize_column(c), used))
    # mapping：原始输入列 → 最终列名（若有 column_map，以 rename 后的列为中间态）
    mapping = dict(zip(original_columns, new_cols))
    work.columns = new_cols
    if len(set(new_cols)) != len(new_cols):
        block_reasons.append(f"duplicate columns after normalize: {new_cols}")

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
    result.frame = work  # type: ignore[attr-defined]
    return result


def _load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise ImportError(
                "reading Excel requires openpyxl; pip install 'wanglaoshi[business]' or openpyxl"
            ) from e
    raise ValueError(f"unsupported file format: {suffix or '(none)'}; supported={sorted(_SUPPORTED_TABULAR)}")


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
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_TABULAR:
        return StageResult(
            stage_id="D02_structuring",
            status="blocked",
            block_reasons=[
                f"unsupported file format: {suffix or '(none)'}; supported={sorted(_SUPPORTED_TABULAR)}"
            ],
            input_paths=[str(path)],
            rules=["D02.format_gate"],
        )
    try:
        df = _load_tabular(path)
    except Exception as e:  # noqa: BLE001
        return StageResult(
            stage_id="D02_structuring",
            status="blocked",
            block_reasons=[f"failed to read {path.name}: {type(e).__name__}: {e}"],
            input_paths=[str(path)],
            rules=["D02.load_tabular"],
        )
    result = structure_dataframe(df, output_path=output_path, **kwargs)
    result.input_paths = [str(path)]
    return result
