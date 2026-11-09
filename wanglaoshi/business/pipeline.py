"""业务分析总控：按阶段串联，阻断后不生成正式下游结果。

可以把 BusinessPipeline 理解成一条「装配线」：
  收货(D01) → 规整(D02) → 清洗(D03) → EDA(D12) → 衍生(D04)
  → 计划(D05) → 指标(D06) → 多维分析(D07) → 出图(D08)
  → 结论建议(D09) → 报告归档(D10)

重要约定（初学者务必记住）：
1. 上游 blocked 时，下游记为 skipped，不要假装跑成功了
2. 正式交付（report / delivery_index）在最后；若整次 blocked，则不可交付
3. 字段角色（日期列、金额列、维度列）能显式传入最好；不传时才做有限自动探测
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .contracts import RunRecord, StageResult
from .intake import intake_file
from .structuring import structure_file
from .cleaning import clean_dataframe
from .eda import run_eda_profile
from .derived import derive_metrics
from .planning import build_analysis_plan
from .metrics import calculate_metrics
from .analysis import run_multidimensional_analysis
from .visualization import archive_charts
from .insights import generate_insights
from .archive import archive_delivery


# 上游阻断后，需要显式 skipped 的下游阶段列表
# （写全是为了报告里能看到「哪几步没跑、因为什么」）
_DOWNSTREAM_AFTER = {
    "D01": [
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
    ],
    "D02": [
        "D03_cleaning",
        "D12_eda",
        "D04_derived",
        "D05_planning",
        "D06_metrics",
        "D07_analysis",
        "D08_visualization",
        "D09_insights",
        "D10_report_archive",
    ],
}


def _skip_remaining(record: RunRecord, stage_ids: Sequence[str], reason: str) -> None:
    """把尚未执行的阶段统一记为 skipped，并写明原因。"""
    for stage_id in stage_ids:
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
        redact_fields: Optional[Sequence[str]] = None,
        chart_unit: str = "currency_unit",
    ) -> None:
        """创建流水线。

        常用参数：
        - archive_dir       : 中间产物与交付物目录
        - required_columns  : D01 检查的必需列
        - period            : 业务周期，会写进图表元数据与台账
        - random_seed       : 抽样等随机过程的种子，保证可复现
        - stop_on_block     : True 时上游 blocked 立即跳过下游
        - date_col/value_col/dim_cols/segment_cols : 字段角色（强烈建议显式传入）
        - redact_fields     : 归档时脱敏的字段名列表
        - chart_unit        : 图纵轴单位文案，如 CNY
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
        self.redact_fields = list(redact_fields) if redact_fields else None
        self.chart_unit = chart_unit

    def run(self, source_path: str | Path, run_id: Optional[str] = None) -> RunRecord:
        """对单个输入文件跑完整主链，返回 RunRecord。"""
        # run_id 用于区分同一次会话里的多次运行；不传则自动生成短 uuid
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
            },
            random_seed=self.random_seed,
        )

        # ========== D01 接收 ==========
        intake = intake_file(
            source_path,
            self.archive_dir,
            required_columns=self.required_columns,
            period=self.period,
        )
        record.add_stage(intake)
        if intake.data.get("sha256"):
            record.input_hashes[str(source_path)] = intake.data["sha256"]

        if intake.status == "blocked" and self.stop_on_block:
            _skip_remaining(record, _DOWNSTREAM_AFTER["D01"], "skipped because upstream blocked")
            return record

        # ========== D02 结构化 ==========
        structured_path = self.archive_dir / f"{rid}__structured.csv"
        structuring = structure_file(source_path, structured_path)
        record.add_stage(structuring)
        if structuring.status == "blocked" and self.stop_on_block:
            _skip_remaining(record, _DOWNSTREAM_AFTER["D02"], "skipped because upstream blocked")
            return record

        # 优先用 StageResult 上挂的 DataFrame；没有则读落盘文件
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
        record.add_stage(cleaning)

        cleaned_df = getattr(cleaning, "frame", None)
        if cleaned_df is None and cleaned_path.exists():
            cleaned_df = pd.read_csv(cleaned_path)
        if cleaned_df is None:
            cleaned_df = pd.DataFrame()

        # ---------- 字段角色：显式参数优先，否则做「有限」自动探测 ----------
        # 注意：自动探测只认常见列名，生产环境请显式传参，避免静默猜错
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

        # ========== D12 EDA（放在衍生前，先看清洗后的数据画像）==========
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
        record.add_stage(eda)

        # ========== D04 衍生字段 ==========
        derived = derive_metrics(
            cleaned_df,
            date_col=date_col,
            value_col=value_col,
            share_group_cols=dim_cols[:1] if dim_cols else None,
            add_wow=bool(date_col and value_col),
        )
        record.add_stage(derived)

        derived_df = getattr(derived, "frame", cleaned_df)

        # ========== D05 四方向分析计划 ==========
        planning = build_analysis_plan(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_cols=dim_cols,
            segment_cols=segment_cols,
            goal="sales_overview",
        )
        record.add_stage(planning)

        # ========== D06 指标 ==========
        group_cols = self.metric_group_cols or dim_cols[:1] or None
        metrics = calculate_metrics(
            derived_df,
            group_cols=group_cols,
            share_of="gmv" if value_col else None,
        )
        record.add_stage(metrics)

        # ========== D07 六视角分析 ==========
        analysis = run_multidimensional_analysis(
            derived_df,
            date_col=date_col,
            value_col=value_col,
            dim_col=dim_cols[0] if dim_cols else None,
            segment_col=segment_cols[0] if segment_cols else None,
        )
        record.add_stage(analysis)

        # ========== D08 图表归档 ==========
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
        record.add_stage(charts)

        # ========== D09 结论与建议（必须能引用前面的 evidence）==========
        insights = generate_insights(
            metrics_data=metrics.data,
            analysis_data=analysis.data,
            planning_data=planning.data,
        )
        record.add_stage(insights)

        # ========== D10 报告 + 交付索引 ==========
        # 注意：若前面已 blocked，archive_delivery 会把 deliverable 标为 False
        delivery_dir = self.archive_dir / f"{rid}__delivery"
        archive = archive_delivery(
            record,
            delivery_dir,
            insights_data=insights.data,
            metrics_data=metrics.data,
            analysis_data=analysis.data,
            redact_fields=self.redact_fields,
        )
        record.add_stage(archive)

        return record

    def run_to_dict(self, source_path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        """便捷方法：直接返回 summary dict，适合快速打印。"""
        return self.run(source_path, **kwargs).summary()
