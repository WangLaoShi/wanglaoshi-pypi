"""假设检验最小工具箱（P0）。对齐 E14。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def _arr(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


class StatsToolkit:
    """t / χ² / 单因素 ANOVA + 效应量与适用性提示。"""

    @staticmethod
    def ttest_ind(
        a: Sequence[float],
        b: Sequence[float],
        *,
        equal_var: bool = False,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        x, y = _arr(a), _arr(b)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        warnings: List[str] = []
        if len(x) < 3 or len(y) < 3:
            warnings.append("small sample (n<3); interpret cautiously")
        # Shapiro 仅作提示（大样本跳过）
        if 3 <= len(x) <= 50:
            warnings.append(f"shapiro_a_p≈{stats.shapiro(x).pvalue:.4g}")
        if 3 <= len(y) <= 50:
            warnings.append(f"shapiro_b_p≈{stats.shapiro(y).pvalue:.4g}")

        res = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
        pooled = np.sqrt(
            ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / max(len(x) + len(y) - 2, 1)
        )
        d = float((y.mean() - x.mean()) / pooled) if pooled > 0 else float("nan")
        return {
            "statistic": float(res.statistic),
            "pvalue": float(res.pvalue),
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "n_a": int(len(x)),
            "n_b": int(len(y)),
            "equal_var": equal_var,
            "cohens_d": d,
            "warnings": warnings,
        }

    @staticmethod
    def chi2_independence(table: Sequence[Sequence[float]]) -> Dict[str, Any]:
        tab = np.asarray(table, dtype=float)
        if tab.ndim != 2:
            raise ValueError("table must be 2-D contingency")
        chi2, p, dof, expected = stats.chi2_contingency(tab)
        warnings: List[str] = []
        if expected.min() < 5:
            warnings.append(
                f"min expected frequency={expected.min():.3g} < 5; consider merge categories or Fisher"
            )
        return {
            "chi2": float(chi2),
            "pvalue": float(p),
            "dof": int(dof),
            "expected": expected,
            "min_expected": float(expected.min()),
            "warnings": warnings,
        }

    @staticmethod
    def anova_oneway(*groups: Sequence[float]) -> Dict[str, Any]:
        cleaned = []
        for g in groups:
            a = _arr(g)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                raise ValueError("empty group")
            cleaned.append(a)
        if len(cleaned) < 2:
            raise ValueError("need >= 2 groups")
        f_stat, p = stats.f_oneway(*cleaned)
        warnings: List[str] = []
        if any(len(g) < 3 for g in cleaned):
            warnings.append("at least one group has n<3")
        # eta-squared 近似
        grand = np.concatenate(cleaned)
        ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in cleaned)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")
        return {
            "f_statistic": float(f_stat),
            "pvalue": float(p),
            "n_groups": int(len(cleaned)),
            "group_sizes": [int(len(g)) for g in cleaned],
            "eta_squared": eta2,
            "warnings": warnings,
            "note": "pairwise post-hoc not corrected; use FDR/Tukey in production",
        }
