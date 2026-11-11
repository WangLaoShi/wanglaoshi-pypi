"""D10 analysis-report + delivery-archive：报告、脱敏、交付索引、缺项阻断。

两种模式：
- formal（默认）：正式交付。除文件齐全外，还要求必需阶段存在、运行未 blocked、
  必需指标有效；否则 deliverable=False 且本阶段 blocked。
- diagnostic：仅导出诊断产物（摘要/报告/索引），用于排障；不宣称可交付。

交付物至少包括：
- report.md / run_summary.json / delivery_index.json
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import RunRecord, StageResult


REQUIRED_DELIVERABLES = (
    "run_summary.json",
    "report.md",
    "delivery_index.json",
)

FORMAL_REQUIRED_STAGES = (
    "D01_intake",
    "D06_metrics",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(obj: Any) -> Any:
    """把 numpy/pandas 等不可直接 JSON 序列化的对象转成基础类型。"""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # noqa: BLE001
        pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _redact(obj: Any, fields: Sequence[str]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in fields:
                out[k] = "***REDACTED***"
            else:
                out[k] = _redact(v, fields)
        return out
    if isinstance(obj, list):
        return [_redact(x, fields) for x in obj]
    return _json_safe(obj)


def _collect_sensitive_values(obj: Any, fields: Sequence[str], acc: Optional[set] = None) -> set:
    """从结构化数据中收集脱敏字段对应的原始值，供文本扫描替换。"""
    found: set = acc if acc is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in fields and v is not None and not isinstance(v, (dict, list)):
                s = str(v)
                if s and s != "***REDACTED***":
                    found.add(s)
            else:
                _collect_sensitive_values(v, fields, found)
    elif isinstance(obj, list):
        for x in obj:
            _collect_sensitive_values(x, fields, found)
    return found


def _redact_text(text: str, fields: Sequence[str], sensitive_values: Sequence[str]) -> str:
    out = text
    for field in fields:
        out = re.sub(
            rf"({re.escape(field)}\s*[:=]\s*)([^\s,;|]+)",
            r"\1***REDACTED***",
            out,
            flags=re.IGNORECASE,
        )
    # 长值优先，避免短串误伤
    for val in sorted({str(v) for v in sensitive_values if v}, key=len, reverse=True):
        if len(val) < 3:
            continue
        out = out.replace(val, "***REDACTED***")
    return out


def _scan_redaction_leaks(text: str, sensitive_values: Sequence[str]) -> List[str]:
    leaks = []
    for val in sensitive_values:
        if val and len(str(val)) >= 3 and str(val) in text:
            leaks.append(str(val))
    return leaks


def _render_report_md(
    record: RunRecord,
    *,
    insights_data: Optional[Dict[str, Any]] = None,
    metrics_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[Sequence[str]] = None,
    sensitive_values: Optional[Sequence[str]] = None,
) -> str:
    insights = insights_data or {}
    metrics = (metrics_data or {}).get("metrics") or {}
    findings = insights.get("findings") or []
    recs = insights.get("recommendations") or []
    fields = list(redact_fields or [])
    safe_config = _redact(record.config, fields) if fields else record.config
    lines = [
        f"# 业务分析报告（run `{record.run_id}`）",
        "",
        f"- 生成时间(UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- blocked: {record.blocked()}",
        f"- random_seed: {record.random_seed}",
        "",
        "## 1. 摘要",
        f"- 阶段数: {len(record.stages)}",
        f"- 指标: {', '.join(f'{k}={v}' for k, v in metrics.items()) or '（无）'}",
        "",
        "## 2. 数据说明",
        f"- 输入哈希: {json.dumps(record.input_hashes, ensure_ascii=False)}",
        f"- 配置: `{json.dumps(safe_config, ensure_ascii=False)}`",
        "",
        "## 3. 阶段状态",
    ]
    for s in record.stages:
        lines.append(f"- `{s.stage_id}`: **{s.status}**")
        if s.warnings:
            lines.append(f"  - warnings: {'; '.join(s.warnings)}")
        if s.block_reasons:
            lines.append(f"  - block_reasons: {'; '.join(s.block_reasons)}")

    lines += ["", "## 4. 发现"]
    if not findings:
        lines.append("- （无）")
    for f in findings:
        lines.append(f"- [{f.get('type')}] **{f.get('title')}**: {f.get('statement')}")
        if f.get("evidence_ids"):
            lines.append(f"  - evidence: {', '.join(f.get('evidence_ids') or [])}")

    lines += ["", "## 5. 建议"]
    if not recs:
        lines.append("- （无）")
    for r in recs:
        lines.append(f"- `{r.get('id')}`: {r.get('action')}（收益: {r.get('expected_uplift')}）")

    views = (analysis_data or {}).get("views") or {}
    lines += ["", "## 6. 局限"]
    lines.append("- 本报告由规则引擎生成，不含 AI 编造原因。")
    if views.get("pareto", {}).get("note"):
        lines.append(f"- {views['pareto']['note']}")
    lines.append("- 缺证据的发现已降级为观察；无对照基线的收益标记为 pending。")
    lines.append("")
    text = "\n".join(lines)
    if fields:
        text = _redact_text(text, fields, list(sensitive_values or []))
    return text


def archive_delivery(
    record: RunRecord,
    output_dir: str | Path,
    *,
    insights_data: Optional[Dict[str, Any]] = None,
    metrics_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[Sequence[str]] = None,
    required: Optional[Sequence[str]] = None,
    mode: str = "formal",
    required_stages: Optional[Sequence[str]] = None,
    require_metrics: Optional[Sequence[str]] = ("gmv",),
) -> StageResult:
    """写出报告与交付索引。

    - mode='formal'：正式交付门槛（阶段 + 指标 + 文件）
    - mode='diagnostic'：只保证诊断文件写出，deliverable 恒为 False
    """
    if mode not in {"formal", "diagnostic"}:
        raise ValueError("mode must be 'formal' or 'diagnostic'")

    rules = ["D10.report", "D10.delivery_index", "D10.required_files", f"D10.mode.{mode}"]
    warnings: List[str] = []
    block_reasons: List[str] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    redact_fields = list(redact_fields or [])
    required = list(required or REQUIRED_DELIVERABLES)
    req_stages = list(required_stages if required_stages is not None else FORMAL_REQUIRED_STAGES)
    req_metrics = list(require_metrics or ())

    sensitive_values = sorted(
        _collect_sensitive_values(record.to_dict(), redact_fields)
        | _collect_sensitive_values(insights_data or {}, redact_fields)
        | _collect_sensitive_values(metrics_data or {}, redact_fields)
    )

    # 1) run summary（可脱敏）
    summary = record.summary()
    summary_path = out_dir / "run_summary.json"
    summary_obj = _json_safe(_redact(summary, redact_fields))
    summary_text = json.dumps(summary_obj, ensure_ascii=False, indent=2)
    if redact_fields:
        summary_text = _redact_text(summary_text, redact_fields, sensitive_values)
    summary_path.write_text(summary_text, encoding="utf-8")
    outputs.append(str(summary_path))

    # 2) markdown report
    report_md = _render_report_md(
        record,
        insights_data=insights_data,
        metrics_data=metrics_data,
        analysis_data=analysis_data,
        redact_fields=redact_fields,
        sensitive_values=sensitive_values,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    outputs.append(str(report_path))

    # 3) full record json
    full_path = out_dir / "run_record.json"
    full_obj = _json_safe(_redact(record.to_dict(), redact_fields))
    full_text = json.dumps(full_obj, ensure_ascii=False, indent=2)
    if redact_fields:
        full_text = _redact_text(full_text, redact_fields, sensitive_values)
    full_path.write_text(full_text, encoding="utf-8")
    outputs.append(str(full_path))

    # 4) formal gates
    if mode == "formal":
        present = {s.stage_id for s in record.stages}
        missing_stages = [s for s in req_stages if s not in present]
        if missing_stages:
            block_reasons.append(f"formal delivery missing stages: {missing_stages}")
        if record.blocked():
            block_reasons.append("upstream run is blocked; formal delivery not allowed")
            warnings.append("deliverable flagged false because run blocked")
        metrics = (metrics_data or {}).get("metrics") or {}
        if req_metrics and metrics_data is None and "D06_metrics" not in present:
            block_reasons.append("formal delivery requires metrics_data or D06_metrics stage")
        for name in req_metrics:
            raw = metrics.get(name)
            if raw is None:
                # try pull from stage
                for s in record.stages:
                    if s.stage_id == "D06_metrics":
                        raw = (s.data.get("metrics") or {}).get(name)
                        break
            try:
                import math

                bad = raw is None or (isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)))
            except Exception:  # noqa: BLE001
                bad = raw is None
            if bad:
                block_reasons.append(f"required metric {name!r} missing or NaN for formal delivery")
    else:
        warnings.append("diagnostic mode: deliverable forced false")

    # 5) redaction leak scan on written texts
    if redact_fields and sensitive_values:
        for label, text in (
            ("run_summary.json", summary_text),
            ("report.md", report_md),
            ("run_record.json", full_text),
        ):
            leaks = _scan_redaction_leaks(text, sensitive_values)
            if leaks:
                block_reasons.append(f"redaction leak in {label}: {leaks[:3]}")

    # 6) delivery index
    files_meta = []
    for p in [summary_path, report_path, full_path]:
        files_meta.append(
            {
                "name": p.name,
                "path": str(p),
                "sha256": _sha256_file(p),
                "size_bytes": p.stat().st_size,
                "readable": p.exists() and p.stat().st_size > 0,
            }
        )

    missing = [name for name in required if not (out_dir / name).exists()]
    deliverable = (
        False
        if mode == "diagnostic"
        else (len(missing) == 0 and not block_reasons and not record.blocked())
    )
    index = {
        "run_id": record.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "files": files_meta,
        "required": required,
        "missing_required": missing,
        "required_stages": req_stages if mode == "formal" else [],
        "require_metrics": req_metrics if mode == "formal" else [],
        "deliverable": deliverable,
    }
    index_path = out_dir / "delivery_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(str(index_path))

    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        block_reasons.append(f"missing required deliverables: {missing}")
        deliverable = False

    index["missing_required"] = missing
    index["deliverable"] = deliverable and len(block_reasons) == 0 and mode == "formal" and not record.blocked()
    # recompute final deliverable consistently
    if mode == "diagnostic":
        index["deliverable"] = False
    else:
        index["deliverable"] = len(missing) == 0 and len(block_reasons) == 0 and not record.blocked()
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "blocked" if block_reasons else ("warning" if warnings else "passed")
    return StageResult(
        stage_id="D10_report_archive",
        status=status,
        data={
            "delivery_index": index,
            "report_chars": len(report_md),
            "redact_fields": redact_fields,
            "mode": mode,
        },
        rules=rules,
        warnings=warnings,
        block_reasons=block_reasons,
        output_paths=outputs,
        evidence_ids=[f"file:{m['name']}:{m['sha256'][:12]}" for m in files_meta],
    )
