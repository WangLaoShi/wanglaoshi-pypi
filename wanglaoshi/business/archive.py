"""D10 analysis-report + delivery-archive：报告、脱敏、交付索引、缺项阻断。

交付物至少包括：
- report.md           给人读的报告
- run_summary.json    阶段摘要
- delivery_index.json 文件清单 + sha256 + 是否可交付

若上游 blocked 或缺少必需文件 → 本阶段 blocked，deliverable=False。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import RunRecord, StageResult


REQUIRED_DELIVERABLES = (
    "run_summary.json",
    "report.md",
    "delivery_index.json",
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
    # numpy scalar / dtype
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


def _render_report_md(
    record: RunRecord,
    *,
    insights_data: Optional[Dict[str, Any]] = None,
    metrics_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[Sequence[str]] = None,
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
            lines.append(f"  - blocked: {'; '.join(s.block_reasons)}")

    lines += ["", "## 4. 发现"]
    if not findings:
        lines.append("- （无）")
    for f in findings:
        lines.append(
            f"- [{f.get('type')}] **{f.get('title')}**: {f.get('statement')} "
            f"(evidence={f.get('evidence_ids')})"
        )

    lines += ["", "## 5. 建议"]
    if not recs:
        lines.append("- （无）")
    for r in recs:
        lines.append(
            f"- `{r.get('id')}` → {r.get('action')} "
            f"[verify={r.get('verify_metric')}, uplift={r.get('expected_uplift')}]"
        )

    views = (analysis_data or {}).get("views") or {}
    lines += ["", "## 6. 局限"]
    lines.append("- 本报告由规则引擎生成，不含 AI 编造原因。")
    if views.get("pareto", {}).get("note"):
        lines.append(f"- {views['pareto']['note']}")
    lines.append("- 缺证据的发现已降级为观察；无对照基线的收益标记为 pending。")
    lines.append("")
    return "\n".join(lines)


def archive_delivery(
    record: RunRecord,
    output_dir: str | Path,
    *,
    insights_data: Optional[Dict[str, Any]] = None,
    metrics_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    redact_fields: Optional[Sequence[str]] = None,
    required: Optional[Sequence[str]] = None,
) -> StageResult:
    """写出报告与交付索引；缺必需文件则 blocked。"""
    rules = ["D10.report", "D10.delivery_index", "D10.required_files"]
    warnings: List[str] = []
    block_reasons: List[str] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    redact_fields = list(redact_fields or [])
    required = list(required or REQUIRED_DELIVERABLES)

    # 1) run summary（可脱敏）
    summary = record.summary()
    summary_path = out_dir / "run_summary.json"
    summary_obj = _json_safe(_redact(summary, redact_fields))
    summary_text = json.dumps(summary_obj, ensure_ascii=False, indent=2)
    summary_path.write_text(summary_text, encoding="utf-8")
    outputs.append(str(summary_path))

    # 2) markdown report
    report_md = _render_report_md(
        record,
        insights_data=insights_data,
        metrics_data=metrics_data,
        analysis_data=analysis_data,
        redact_fields=redact_fields,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    outputs.append(str(report_path))

    # 3) full record json（脱敏 + JSON-safe）
    full_path = out_dir / "run_record.json"
    full_obj = _json_safe(_redact(record.to_dict(), redact_fields))
    full_path.write_text(json.dumps(full_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(str(full_path))

    # 4) delivery index with hashes
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
    index = {
        "run_id": record.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files_meta,
        "required": required,
        "missing_required": missing,
        "deliverable": len(missing) == 0 and not record.blocked(),
    }
    index_path = out_dir / "delivery_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(str(index_path))

    # re-check required after writing index itself
    missing = [name for name in required if not (out_dir / name).exists()]
    if missing:
        block_reasons.append(f"missing required deliverables: {missing}")
    if record.blocked():
        block_reasons.append("upstream run is blocked; formal delivery not allowed")
        warnings.append("deliverable flagged false because run blocked")

    # refresh index deliverable flag if needed
    index["missing_required"] = missing
    index["deliverable"] = len(missing) == 0 and not record.blocked()
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "blocked" if block_reasons else ("warning" if warnings else "passed")
    return StageResult(
        stage_id="D10_report_archive",
        status=status,
        data={
            "delivery_index": index,
            "report_chars": len(report_md),
            "redact_fields": redact_fields,
        },
        rules=rules,
        warnings=warnings,
        block_reasons=block_reasons,
        output_paths=outputs,
        evidence_ids=[f"file:{m['name']}:{m['sha256'][:12]}" for m in files_meta],
    )
