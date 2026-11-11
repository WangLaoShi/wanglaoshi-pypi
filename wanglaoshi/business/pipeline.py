"""业务分析总控：按阶段串联，阻断后不生成正式下游结果。

可以把 BusinessPipeline 理解成一条「装配线」：
  收货(D01) → 规整(D02) → 清洗(D03) → EDA(D12) → 衍生(D04)
  → 计划(D05) → 指标(D06) → 多维分析(D07) → 出图(D08)
  → 结论建议(D09) → 报告归档(D10)

重要约定（初学者务必记住）：
1. 任一阶段 blocked 且 stop_on_block=True 时，下游全部 skipped
2. 正式交付（delivery_index.deliverable）要求必需阶段与必需指标有效
3. 字段角色（日期列、金额列、维度列）能显式传入最好；不传时才做有限自动探测
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from .archive import archive_delivery
from .analysis import run_multidimensional_analysis
from .cleaning import clean_dataframe
from .contracts import RunRecord, StageResult
from .derived import derive_metrics
from .eda import run_eda_profile
from .insights import generate_insights
from .intake import intake_file
from .metrics import build_metric_dict, calculate_metrics
from .planning import build_analysis_plan
from .structuring import structure_file
from .visualization import archive_charts

# 主链阶段顺序（用于统一跳过）
_STAGE_ORDER = [
    "D01_intake",
    "D02_structuring",
    "D03_cleaning",
    "D12_eda",
    "D04_derived",
    "D05_planning",
    "D06_metrics",
    "D07_analysis",
    "D08_visualization",
    "D09_insights",
    "D10_report_archive",
]


def _skip_after(
    record: RunRecord,
    blocked_stage_id: str,
    reason: str,
    *,
    until_exclusive: str = "D10_report_archive",
) -> None:
    """把 blocked 之后、归档之前尚未记录的阶段标为 skipped（D10 留给收尾写出）。"""
    present = {s.stage_id for s in record.stages}
    try:
        idx = _STAGE_ORDER.index(blocked_stage_id)
        end = _STAGE_ORDER.index(until_exclusive)
    except ValueError:
        return
    for stage_id in _STAGE_ORDER[idx + 1 : end]:
        if stage_id in present:
            continue
        record.add_stage(
            StageResult(
                stage_id=stage_id,
                status="skipped",
                warnings=[reason],
            )
        )


class BusinessPipeline:
    """业务分析总控（对应路线图 D11）。

    最小用法::

        pipe = BusinessPipeline('archive/', required_columns=['order_id', 'amount'])
        record = pipe.run('orders.csv')
        print(record.summary())
    """

    def __init__(
        self,
        archive_dir: str | Path,
        *,
        required_columns: Optional[Sequence[str]] = None,
        period: Optional[str] = None,
        random_seed: int = 42,
        stop_on_block: bool = True,
        date_col: Optional[str] = None,
        value_col: Optional[str] = None,
        dim_cols: Optional[Sequence[str]] = None,
        segment_cols: Optional[Sequence[str]] = None,
        metric_group_cols: Optional[Sequence[str]] = None,
        order_id_col: str = "order_id",
        metric_dict: Optional[Dict[str, Any]] = None,
        column_map: Optional[Mapping[str, str]] = None,
        required_metrics: Optional[Sequence[str]] = ("gmv",),
        delivery_mode: str = "formal",
        redact_fields: Optional[Sequence[str]] = None,
        chart_unit: str = "currency_unit",
    ) -> None:
        """创建流水线。

        常用参数：
        - archive_dir / required_columns / period / random_seed / stop_on_block
        - date_col/value_col/dim_cols/segment_cols : 字段角色（强烈建议显式传入）
        - order_id_col / metric_dict / required_metrics : 指标口径
        - column_map : 原始列 → 标准列（在 D02 应用）
        - delivery_mode : formal | diagnostic
        - redact_fields / chart_unit
        """
        self.archive_dir = Path(archive_dir)
        self.required_columns = list(required_columns) if required_columns else None
        self.period = period
        self.random_seed = random_seed
        self.stop_on_block = stop_on_block
        self.date_col = date_col
        self.value_col = value_col
        self.dim_cols = list(dim_cols) if dim_cols else None
        self.segment_cols = list(segment_cols) if segment_cols else None
        self.metric_group_cols = list(metric_group_cols) if metric_group_cols else None
        self.order_id_col = order_id_col
        self.metric_dict = metric_dict
        self.column_map = dict(column_map) if column_map else None
        self.required_metrics = list(required_metrics) if required_metrics is not None else None
        self.delivery_mode = delivery_mode
        self.redact_fields = list(redact_fields) if redact_fields else None
        self.chart_unit = chart_unit

    def run(self, source_path: str | Path, run_id: Optional[str] = None) -> RunRecord:
        """对单个输入文件跑完整主链，返回 RunRecord。"""
        rid = run_id or uuid.uuid4().hex[:12]
        record = RunRecord(
            run_id=rid,
            config={
                "archive_dir": str(self.archive_dir),
                "required_columns": self.required_columns,
                "period": self.period,
                "stop_on_block": self.stop_on_block,
                "date_col": self.date_col,
                "value_col": self.value_col,
                "dim_cols": self.dim_cols,
                "segment_cols": self.segment_cols,
                "order_id_col": self.order_id_col,
                "delivery_mode": self.delivery_mode,
                "column_map": self.column_map,
            },
            random_seed=self.random_seed,
        )

        def _finalize_archive(
            *,
            insights_data: Optional[Dict[str, Any]] = None,
            metrics_data: Optional[Dict[str, Any]] = None,
            analysis_data: Optional[Dict[str, Any]] = None,
        ) -> None:
            if any(s.stage_id == "D10_report_archive" for s in record.stages):
                return
            if metrics_data is None:
                for s in record.stages:
                    if s.stage_id == "D06_metrics":
                        metrics_data = s.data
                        break
            delivery_dir = self.archive_dir / f"{rid}__delivery"
            archive = archive_delivery(
                record,
                delivery_dir,
                insights_data=insights_data,
                metrics_data=metrics_data,
                analysis_data=analysis_data,
                redact_fields=self.redact_fields,
                mode=self.delivery_mode,
                require_metrics=self.required_metrics,
            )
            record.add_stage(archive)

        def _stop(stage: StageResult) -> bool:
            record.add_stage(stage)
            if stage.status == "blocked" and self.stop_on_block:
                _skip_after(record, stage.stage_id, "skipped because upstream blocked")
                _finalize_archive()
                return True
            return False

        # ========== D01 接收 ==========
        intake = intake_file(
            source_path,
            self.archive_dir,
            required_columns=self.required_columns,
            period=self.period,
        )
        if intake.data.get("sha256"):
            record.input_hashes[str(source_path)] = intake.data["sha256"]
        if _stop(intake):
            return record

        # ========== D02 结构化 ==========
        structured_path = self.archive_dir / f"{rid}__structured.csv"
        structuring = structure_file(
            source_path,
            structured_path,
            column_map=self.column_map,
        )
        if _stop(structuring):
            return record

        structured_df = getattr(structuring, "frame", None)
        if structured_df is None and structured_path.exists():
            structured_df = pd.read_csv(structured_path)

        # ========== D03 清洗 ==========
        cleaned_path = self.archive_dir / f"{rid}__cleaned.csv"
        log_path = self.archive_dir / f"{rid}__cleaning_log.json"
        cleaning = clean_dataframe(
            structured_df if structured_df is not None else pd.DataFrame(),
            output_path=cleaned_path,
            log_path=log_path,
        )
        if _stop(cleaning):
            return record

        cleaned_df = getattr(cleaning, "frame", None)
        if cleaned_df is None and cleaned_path.exists():
            cleaned_df = pd.read_csv(cleaned_path)
        if cleaned_df is None:
            cleaned_df = pd.DataFrame()

        date_col = self.date_col or next(
            (
                c
                for c in cleaned_df.columns
                if "date" in c or c.endswith("_dt") or c in {"dt", "day", "timestamp"}
            ),
            None,
        )
        value_col = self.value_col or next(
            (c for c in cleaned_df.columns if c in {"amount", "gmv", "sales", "revenue", "cnt", "value"}),
            None,
        )
        dim_cols = self.dim_cols or [c for c in ["channel", "region", "category"] if c in cleaned_df.columns]
        segment_cols = self.segment_cols or [c for c in ["user_tier", "segment", "vip"] if c in cleaned_df.columns]

        # ========== D12 EDA ==========
        eda = run_eda_profile(
            cleaned_df,
            field_roles={
                "date_col": date_col,
                "value_col": value_col,
                "dim_cols": dim_cols,
                "segment_cols": segment_cols,
            },
            random_seed=self.random_seed,
        )
        if _stop(eda):
            return record

        # ========== D04 衍生 ==========
        derived = derive_metrics(
            cleaned_df,
            date_col=date_col,
            value_col=value_col,
            share_group_cols=dim_cols[:1] if dim_cols else None,
            add_wow=bool(date_col and value_col),
        )
        if _stop(derived):
            return record

        derived_df = getattr(derived, "frame", cleaned_df)

        # ========== D05 计划 ==========
        planning = build_analysis_plan(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_cols=dim_cols,
            segment_cols=segment_cols,
            goal="sales_overview",
        )
        if _stop(planning):
            return record

        # ========== D06 指标 ==========
        group_cols = self.metric_group_cols or dim_cols[:1] or None
        if self.metric_dict is not None:
            metric_dict = self.metric_dict
        elif value_col:
            metric_dict = build_metric_dict(
                value_col=value_col,
                order_id_col=self.order_id_col,
            )
        else:
            metric_dict = None
        metrics = calculate_metrics(
            derived_df,
            metric_dict=metric_dict,
            group_cols=group_cols,
            share_of="gmv" if value_col else None,
            required_metrics=self.required_metrics,
        )
        if _stop(metrics):
            return record

        # ========== D07 分析 ==========
        analysis = run_multidimensional_analysis(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_col=dim_cols[0] if dim_cols else None,
            segment_col=segment_cols[0] if segment_cols else None,
        )
        if _stop(analysis):
            return record

        # ========== D08 图表 ==========
        charts_dir = self.archive_dir / f"{rid}__charts"
        charts = archive_charts(
            derived_df,
            charts_dir,
            date_col=date_col,
            value_col=value_col,
            dim_col=dim_cols[0] if dim_cols else None,
            period=self.period,
            source=f"run:{rid}",
            unit=self.chart_unit,
        )
        if _stop(charts):
            return record

        # ========== D09 结论 ==========
        insights = generate_insights(
            metrics_data=metrics.data,
            analysis_data=analysis.data,
            planning_data=planning.data,
        )
        if _stop(insights):
            return record

        # ========== D10 正式/诊断归档 ==========
        _finalize_archive(
            insights_data=insights.data,
            metrics_data=metrics.data,
            analysis_data=analysis.data,
        )
        return record

    def run_to_dict(self, source_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        """便捷方法：直接返回 summary dict，适合快速打印。"""
        return self.run(source_path, **kwargs).summary()
