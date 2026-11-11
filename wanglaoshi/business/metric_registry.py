"""指标口径版本登记（S5 最小版）。

不做完整公式 DSL，只保证：
- 指标字典带 version
- 必填字段齐全
- 聚合方式合法（与计算器共用）
- 依赖指标存在
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .contracts import StageResult
from .metrics import DEFAULT_METRIC_DICT, SUPPORTED_AGGS


def validate_metric_dict(
    metric_dict: Mapping[str, Any],
    *,
    version: Optional[str] = None,
    stage_id: str = "D06_metric_registry",
) -> StageResult:
    """校验指标字典结构；返回规范化副本（含 version）。

    ``__version__`` 视为元数据：先剥离再校验指标条目。
    """
    warnings: List[str] = []
    block_reasons: List[str] = []
    normalized: Dict[str, Dict[str, Any]] = {}

    # 先解析版本元数据，避免把 __version__ 当成非法指标
    raw = dict(metric_dict)
    incoming_ver = raw.pop("__version__", None)

    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            block_reasons.append(f"{name}: spec must be mapping")
            continue
        item = dict(spec)
        deps = item.get("depends_on")
        if deps:
            missing = [d for d in deps if d not in raw]
            if missing:
                block_reasons.append(f"{name}: missing depends_on {missing}")
        else:
            if "value_col" not in item:
                block_reasons.append(f"{name}: missing value_col")
            if "agg" not in item:
                warnings.append(f"{name}: missing agg, default sum assumed")
                item["agg"] = "sum"
            how = item.get("agg", "sum")
            if how not in SUPPORTED_AGGS:
                block_reasons.append(
                    f"{name}: unsupported agg {how!r}; supported={sorted(SUPPORTED_AGGS)}"
                )
        if "direction" not in item:
            warnings.append(f"{name}: missing direction")
        normalized[name] = item

    ver = version or (str(incoming_ver) if incoming_ver is not None else "v0")
    data = {"metric_dict": normalized, "version": ver, "n_metrics": len(normalized)}

    if block_reasons:
        return StageResult(
            stage_id=stage_id,
            status="blocked",
            data=data,
            block_reasons=block_reasons,
            warnings=warnings,
            rules=["D06.metric_registry"],
        )
    status = "warning" if warnings else "passed"
    return StageResult(
        stage_id=stage_id,
        status=status,
        data=data,
        warnings=warnings,
        rules=["D06.metric_registry"],
    )


def default_metric_registry(version: str = "v1") -> StageResult:
    """包装 DEFAULT_METRIC_DICT 并标注版本。"""
    return validate_metric_dict(DEFAULT_METRIC_DICT, version=version)
