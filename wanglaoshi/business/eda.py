"""D12 exploratory-data-analysis 适配 + 统计方法清单（首版）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .contracts import StageResult

# 首版方法清单：仅声明适用条件，不做“全自动选检验”
STAT_METHOD_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "describe",
        "name": "描述统计",
        "requires": ["numeric or categorical columns"],
        "not_for": [],
    },
    {
        "id": "missing_rate",
        "name": "缺失率",
        "requires": ["any columns"],
        "not_for": [],
    },
    {
        "id": "pearson_corr",
        "name": "Pearson 相关",
        "requires": [">=2 numeric columns", "approx linear association"],
        "not_for": ["heavy outliers without robust alternative", "ordinal-only features"],
    },
    {
        "id": "ttest_ind",
        "name": "独立样本 t 检验",
        "requires": ["two groups", "approx normality or large n"],
        "not_for": ["paired design", "severe skew with small n"],
    },
    {
        "id": "mannwhitney",
        "name": "Mann-Whitney U",
        "requires": ["two independent groups", "ordinal/continuous"],
        "not_for": ["paired design"],
    },
    {
        "id": "chi_square",
        "name": "卡方独立性",
        "requires": ["two categorical columns", "expected counts adequate"],
        "not_for": ["tiny expected cell counts without correction"],
    },
]


def run_eda_profile(
    df: pd.DataFrame,
    *,
    field_roles: Optional[Dict[str, Any]] = None,
    sample_n: Optional[int] = 5000,
    random_seed: int = 42,
    use_analyzer: bool = True,
) -> StageResult:
    """统一数据画像：优先复用 DataAnalyzer，失败则退化为轻量画像。"""
    rules = ["D12.eda_profile", "D12.stat_catalog"]
    warnings: List[str] = []
    roles = field_roles or {}

    work = df
    if sample_n is not None and len(df) > sample_n:
        work = df.sample(sample_n, random_state=random_seed)
        warnings.append(f"sampled to {sample_n} rows for EDA")

    profile: Dict[str, Any] = {
        "n_rows": int(len(work)),
        "n_cols": int(work.shape[1]),
        "columns": list(work.columns),
        "dtypes": {c: str(work[c].dtype) for c in work.columns},
        "missing_rate": {c: float(work[c].isna().mean()) for c in work.columns},
        "field_roles": roles,
        "numeric_columns": work.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": work.select_dtypes(exclude="number").columns.tolist(),
    }

    analyzer_info: Dict[str, Any] = {}
    if use_analyzer:
        try:
            from wanglaoshi.Analyzer import DataAnalyzer

            analyzer = DataAnalyzer(work)
            basic = analyzer.basic_statistics()
            missing = analyzer.missing_value_analysis()
            info = analyzer.get_basic_info()
            # 保证可 JSON 序列化
            safe_info = {str(k): (str(v) if not isinstance(v, (str, int, float, bool, list, dict)) and v is not None else v) for k, v in dict(info).items()}
            analyzer_info = {
                "basic_statistics_rows": int(len(basic)) if basic is not None else 0,
                "missing_analysis_rows": int(len(missing)) if missing is not None else 0,
                "get_basic_info": safe_info,
            }
            rules.append("D12.reuse_DataAnalyzer")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"DataAnalyzer unavailable, using lightweight profile: {e}")

    # 适用性提示：根据字段角色标出可能可用的方法
    applicable = []
    n_num = len(profile["numeric_columns"])
    n_cat = len(profile["categorical_columns"])
    for m in STAT_METHOD_CATALOG:
        ok = True
        reason = "ok"
        if m["id"] == "pearson_corr" and n_num < 2:
            ok, reason = False, "need >=2 numeric columns"
        if m["id"] in {"ttest_ind", "mannwhitney"} and not roles.get("segment_cols"):
            ok, reason = False, "need segment_cols in field_roles"
        if m["id"] == "chi_square" and n_cat < 2:
            ok, reason = False, "need >=2 categorical columns"
        applicable.append({"id": m["id"], "applicable": ok, "reason": reason, **{k: m[k] for k in ("name", "requires", "not_for")}})

    status = "warning" if warnings else "passed"
    return StageResult(
        stage_id="D12_eda",
        status=status,
        data={
            "profile": profile,
            "analyzer": analyzer_info,
            "stat_methods": applicable,
        },
        rules=rules,
        warnings=warnings,
        evidence_ids=["eda:profile"],
    )
