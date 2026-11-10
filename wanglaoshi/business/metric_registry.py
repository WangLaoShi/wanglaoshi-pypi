"""指标口径版本登记（S5 最小版）。

不做完整公式 DSL，只保证：
- 指标字典带 version
- 必填字段齐全
- 依赖指标存在
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .contracts import StageResult
from .metrics import DEFAULT_METRIC_DICT


def validate_metric_dict(
    metric_dict: Mapping[str, Mapping[str, Any]],
    *,
    version: Optional[str] = None,
    stage_id: str = "D06_metric_registry",
) -> StageResult:
    """校验指标字典结构；返回规范化副本（含 version）。"""
    warnings: List[str] = []
    block_reasons: List[str] = []
    normalized: Dict[str, Dict[str, Any]] = {}

    for name, spec in metric_dict.items():
        if not isinstance(spec, Mapping):
            block_reasons.append(f"{name}: spec must be mapping")
            continue
        item = dict(spec)
        deps = item.get("depends_on")
        if deps:
            missing = [d for d in deps if d not in metric_dict]
            if missing:
                block_reasons.append(f"{name}: missing depends_on {missing}")
        else:
            if "value_col" not in item:
                block_reasons.append(f"{name}: missing value_col")
            if "agg" not in item:
                warnings.append(f"{name}: missing agg, default sum assumed")
                item["agg"] = "sum"
        if "direction" not in item:
            warnings.append(f"{name}: missing direction")
        normalized[name] = item

    # 若字典里误带 __version__ 键则剥离，不以指标论
    incoming_ver = metric_dict.get("__version__") if "__version__" in metric_dict else None
    normalized.pop("__version__", None)
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
