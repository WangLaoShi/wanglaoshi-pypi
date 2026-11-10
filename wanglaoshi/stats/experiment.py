"""试验设计最小工具（P0）。对齐 E13。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats


class ExperimentDesign:
    """两比例检验、样本量、贝叶斯门槛概率（共轭，无 pymc 硬依赖）。"""

    @staticmethod
    def two_proportion_test(
        x1: int,
        n1: int,
        x2: int,
        n2: int,
    ) -> Dict[str, Any]:
        """两比例 z 检验（pooled）+ 未合并 SE 的 95% CI。"""
        if min(n1, n2) <= 0:
            raise ValueError("n1 and n2 must be positive")
        p1, p2 = x1 / n1, x2 / n2
        p = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = 0.0 if se == 0 else (p2 - p1) / se
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        diff = p2 - p1
        ci = (float(diff - 1.96 * se_unpooled), float(diff + 1.96 * se_unpooled))
        # χ² 2x2
        table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=float)
        chi2, p_chi, dof, _ = stats.chi2_contingency(table)
        return {
            "p1": float(p1),
            "p2": float(p2),
            "lift_pp": float(diff * 100),
            "z": float(z),
            "pvalue": pval,
            "ci95": ci,
            "chi2": float(chi2),
            "chi2_pvalue": float(p_chi),
            "dof": int(dof),
            "n1": int(n1),
            "n2": int(n2),
        }

    @staticmethod
    def sample_size_two_proportion(
        p1: float,
        mde_abs: float,
        *,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> Dict[str, Any]:
        """等权两臂、双侧、正态近似下的每臂样本量。"""
        from statsmodels.stats.power import NormalIndPower
        from statsmodels.stats.proportion import proportion_effectsize

        p2 = p1 + mde_abs
        if not (0 < p1 < 1 and 0 < p2 < 1):
            raise ValueError("p1 and p1+mde_abs must be in (0,1)")
        es = float(proportion_effectsize(p1, p2))
        n = NormalIndPower().solve_power(
            effect_size=es, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
        )
        return {
            "n_per_arm": int(np.ceil(n)),
            "effect_size_h": es,
            "p1": p1,
            "p2": p2,
            "mde_abs": mde_abs,
            "alpha": alpha,
            "power": power,
        }

    @staticmethod
    def bayes_prob_beat(
        x_a: int,
        n_a: int,
        x_b: int,
        n_b: int,
        *,
        delta: float = 0.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        n_mc: int = 50_000,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Beta-Binomial 后验：P(p_B > p_A + delta)。"""
        rng = np.random.default_rng(random_state)
        pa = rng.beta(alpha0 + x_a, beta0 + n_a - x_a, size=n_mc)
        pb = rng.beta(alpha0 + x_b, beta0 + n_b - x_b, size=n_mc)
        prob = float(np.mean(pb > pa + delta))
        return {
            "prob_b_beats_a": prob,
            "delta": delta,
            "prior": (alpha0, beta0),
            "posterior_mean_a": float((alpha0 + x_a) / (alpha0 + beta0 + n_a)),
            "posterior_mean_b": float((alpha0 + x_b) / (alpha0 + beta0 + n_b)),
            "n_mc": n_mc,
        }

    @staticmethod
    def prior_sensitivity(
        x_a: int,
        n_a: int,
        x_b: int,
        n_b: int,
        *,
        delta: float = 0.005,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        n_mc: int = 30_000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """多组先验下的门槛概率表。"""
        if priors is None:
            priors = {
                "uniform(1,1)": (1.0, 1.0),
                "weak_10pct(2,18)": (2.0, 18.0),
                "strong_10pct(20,180)": (20.0, 180.0),
            }
        rows = []
        for name, (a0, b0) in priors.items():
            r = ExperimentDesign.bayes_prob_beat(
                x_a, n_a, x_b, n_b, delta=delta, alpha0=a0, beta0=b0, n_mc=n_mc, random_state=random_state
            )
            rows.append({"prior": name, **r})
        return {"delta": delta, "rows": rows}
