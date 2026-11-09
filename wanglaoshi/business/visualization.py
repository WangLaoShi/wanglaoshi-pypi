"""D08 data-visualization：图表独立归档（标题/单位/周期/来源齐全）。

每张图除了 png，还要有 sidecar json（title/unit/period/source）。
没有元数据的图，后续报告很难追溯「这张图是什么口径」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .contracts import StageResult


def _write_meta(path: Path, meta: Dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_charts(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dim_col: Optional[str] = None,
    period: Optional[str] = None,
    source: str = "business_pipeline",
    unit: str = "currency_unit",
) -> StageResult:
    """从关键字段生成趋势/结构图，并写 sidecar 元数据。"""
    rules = ["D08.chart_archive", "D08.metadata"]
    warnings: List[str] = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: List[Dict[str, Any]] = []
    outputs: List[str] = []

    if value_col and date_col and value_col in df.columns and date_col in df.columns:
        work = df[[date_col, value_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna()
        if len(work) == 0:
            warnings.append("trend chart skipped: no valid rows")
        else:
            daily = work.groupby(work[date_col].dt.date)[value_col].sum().sort_index()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(list(map(str, daily.index)), daily.values, marker="o")
            title = f"Trend of {value_col}"
            ax.set_title(title)
            ax.set_xlabel(date_col)
            ax.set_ylabel(f"{value_col} ({unit})")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            img = out_dir / "trend.png"
            meta_path = out_dir / "trend.json"
            fig.savefig(img, dpi=120)
            plt.close(fig)
            meta = {
                "title": title,
                "unit": unit,
                "period": period,
                "source": source,
                "chart_type": "trend",
                "x": date_col,
                "y": value_col,
                "n_points": int(len(daily)),
                "image": img.name,
            }
            _write_meta(meta_path, meta)
            charts.append(meta)
            outputs.extend([str(img), str(meta_path)])
    else:
        warnings.append("trend chart skipped: need date_col and value_col")

    if value_col and dim_col and value_col in df.columns and dim_col in df.columns:
        work = df[[dim_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        g = work.groupby(dim_col, dropna=False)[value_col].sum().sort_values(ascending=False).head(12)
        if g.empty:
            warnings.append("structure chart skipped: empty group")
        else:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(g.index.astype(str), g.values)
            title = f"Structure by {dim_col}"
            ax.set_title(title)
            ax.set_xlabel(dim_col)
            ax.set_ylabel(f"{value_col} ({unit})")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            img = out_dir / "structure.png"
            meta_path = out_dir / "structure.json"
            fig.savefig(img, dpi=120)
            plt.close(fig)
            meta = {
                "title": title,
                "unit": unit,
                "period": period,
                "source": source,
                "chart_type": "structure",
                "x": dim_col,
                "y": value_col,
                "n_bars": int(len(g)),
                "image": img.name,
            }
            _write_meta(meta_path, meta)
            charts.append(meta)
            outputs.extend([str(img), str(meta_path)])
    else:
        warnings.append("structure chart skipped: need dim_col and value_col")

    manifest = {"charts": charts, "period": period, "source": source, "unit": unit}
    manifest_path = out_dir / "charts_manifest.json"
    _write_meta(manifest_path, manifest)
    outputs.append(str(manifest_path))

    if not charts:
        warnings.append("no charts produced")
        status = "warning"
    else:
        status = "warning" if warnings else "passed"

    return StageResult(
        stage_id="D08_visualization",
        status=status,
        data=manifest,
        rules=rules,
        warnings=warnings,
        output_paths=outputs,
        evidence_ids=[f"chart:{c['chart_type']}" for c in charts],
    )
