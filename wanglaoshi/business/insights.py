"""D09 insight-conclusion + actionable-recommendations（规则版 MVP）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contracts import StageResult


def generate_insights(
    *,
    metrics_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    planning_data: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """基于指标与六视角结果生成结论与行动建议。

    规则：
    - 有对比/证据 → 现状/亮点/问题
    - 缺对比或缺证据 → 降级为观察
    - 收益目标无依据 → 建议标记 pending
    """
    rules = ["D09.rule_insights", "D09.recommendations"]
    warnings: List[str] = []
    findings: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []
    evidence_ids: List[str] = []

    metrics = (metrics_data or {}).get("metrics") or {}
    share_check = (metrics_data or {}).get("share_check") or {}
    views = (analysis_data or {}).get("views") or {}
    plan_tasks = (planning_data or {}).get("tasks") or []

    # --- 现状：大盘指标 ---
    if "gmv" in metrics and metrics["gmv"] == metrics["gmv"]:  # not NaN
        eid = "metric:gmv"
        evidence_ids.append(eid)
        findings.append(
            {
                "type": "status",
                "title": "大盘 GMV",
                "statement": f"当期 GMV={metrics['gmv']:.4g}",
                "evidence_ids": [eid],
                "support": {"gmv": metrics["gmv"], "orders": metrics.get("orders"), "aov": metrics.get("aov")},
            }
        )
    else:
        findings.append(
            {
                "type": "observation",
                "title": "大盘指标缺失",
                "statement": "未计算到有效 GMV，无法给出现状结论",
                "evidence_ids": [],
            }
        )
        warnings.append("gmv missing; status insight downgraded")

    # --- 亮点：头部贡献（二八）---
    pareto = views.get("pareto") or {}
    if pareto.get("status") == "ok" and pareto.get("head_keys"):
        eid = "view:pareto"
        evidence_ids.append(eid)
        findings.append(
            {
                "type": "highlight",
                "title": "头部贡献集中",
                "statement": (
                    f"约 {pareto.get('head_share'):.0%} 贡献来自前 {pareto.get('n_head')} 个维度键"
                    f"（阈值 {pareto.get('threshold')}）"
                ),
                "evidence_ids": [eid],
                "support": {"head_keys": pareto.get("head_keys"), "note": pareto.get("note")},
            }
        )
        recommendations.append(
            {
                "id": "rec_focus_head",
                "problem_ref": "highlight:头部贡献集中",
                "action": "优先复盘头部渠道/品类的增长动作，并设定对照实验",
                "steps": ["锁定头部键名单", "对比头部 vs 腰尾转化", "设计可验证的运营动作"],
                "verify_metric": "gmv_share_head",
                "owner_role": "growth",
                "window": "14d",
                "expected_uplift": "pending",
                "expected_uplift_reason": "无历史对照基线，收益目标待定",
            }
        )
    else:
        findings.append(
            {
                "type": "observation",
                "title": "头部贡献待观察",
                "statement": "缺少可用的二八分解结果",
                "evidence_ids": [],
            }
        )

    # --- 问题：异动 ---
    anomaly = views.get("anomaly") or {}
    anomalies = anomaly.get("anomalies") or []
    if anomaly.get("status") == "ok" and anomalies:
        eid = "view:anomaly"
        evidence_ids.append(eid)
        top = anomalies[0]
        findings.append(
            {
                "type": "problem",
                "title": "存在显著异动日",
                "statement": f"检测到 {len(anomalies)} 个异动点，例如 {top.get('date')} z={top.get('z'):.2f}",
                "evidence_ids": [eid],
                "support": {"anomalies": anomalies[:5]},
            }
        )
        recommendations.append(
            {
                "id": "rec_investigate_anomaly",
                "problem_ref": "problem:存在显著异动日",
                "action": "排查异动日的活动/故障/数据口径变化",
                "steps": ["核对埋点与订单日志", "排除节假日/大促", "若为口径问题则回滚口径版本"],
                "verify_metric": "anomaly_count",
                "owner_role": "data",
                "window": "7d",
                "expected_uplift": "pending",
                "expected_uplift_reason": "异动修复收益取决于根因，暂不量化",
            }
        )
    elif anomaly.get("status") == "ok":
        findings.append(
            {
                "type": "status",
                "title": "未发现显著异动",
                "statement": f"在 z>={anomaly.get('z_thresh')} 阈值下无异常点",
                "evidence_ids": ["view:anomaly"],
            }
        )
        evidence_ids.append("view:anomaly")

    # --- 结构占比校验 ---
    if share_check:
        eid = "metric:share_check"
        evidence_ids.append(eid)
        if share_check.get("ok"):
            findings.append(
                {
                    "type": "status",
                    "title": "结构占比一致",
                    "statement": f"分组 share 之和={share_check.get('share_sum'):.6f}，通过容差校验",
                    "evidence_ids": [eid],
                }
            )
        else:
            findings.append(
                {
                    "type": "problem",
                    "title": "结构占比不一致",
                    "statement": f"share_sum={share_check.get('share_sum')} 超出容差 {share_check.get('tol')}",
                    "evidence_ids": [eid],
                }
            )
            recommendations.append(
                {
                    "id": "rec_fix_share",
                    "problem_ref": "problem:结构占比不一致",
                    "action": "检查分组键空值、重复归因与过滤条件",
                    "steps": ["打印 workbook", "核对过滤前后总量", "修复后重跑 D06"],
                    "verify_metric": "share_sum",
                    "owner_role": "data",
                    "window": "3d",
                    "expected_uplift": "pending",
                    "expected_uplift_reason": "数据质量修复不直接对应业务收益",
                }
            )

    # --- 计划不适用方向 → 观察 ---
    for t in plan_tasks:
        if t.get("status") == "not_applicable":
            findings.append(
                {
                    "type": "observation",
                    "title": f"方向不可用: {t.get('direction')}",
                    "statement": t.get("reason") or "not applicable",
                    "evidence_ids": ["plan:task"],
                }
            )

    if not findings:
        warnings.append("no findings generated")
        findings.append(
            {
                "type": "observation",
                "title": "暂无结论",
                "statement": "输入指标/分析为空",
                "evidence_ids": [],
            }
        )

    status = "warning" if warnings else "passed"
    return StageResult(
        stage_id="D09_insights",
        status=status,
        data={
            "findings": findings,
            "recommendations": recommendations,
            "counts": {
                "status": sum(1 for f in findings if f["type"] == "status"),
                "highlight": sum(1 for f in findings if f["type"] == "highlight"),
                "problem": sum(1 for f in findings if f["type"] == "problem"),
                "observation": sum(1 for f in findings if f["type"] == "observation"),
                "recommendations": len(recommendations),
            },
        },
        rules=rules,
        warnings=warnings,
        evidence_ids=sorted(set(evidence_ids)),
    )
