"""Tutorial 04 smoke: analytics StatsToolkit + ExperimentDesign."""

from __future__ import annotations

import numpy as np

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import StatsToolkit, ExperimentDesign


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    rng = np.random.default_rng(42)
    control = rng.normal(loc=10.0, scale=2.0, size=80)
    treat = rng.normal(loc=10.8, scale=2.0, size=80)

    tt = StatsToolkit.ttest_ind(control, treat)
    assert "pvalue" in tt

    ss = ExperimentDesign.sample_size_two_proportion(
        p1=0.08, mde_abs=0.04, alpha=0.05, power=0.8
    )
    assert ss["n_per_arm"] > 0

    prop = ExperimentDesign.two_proportion_test(x1=20, n1=200, x2=32, n2=200)
    assert "pvalue" in prop
    print("04_analytics_stats OK", tt["pvalue"], ss["n_per_arm"])


if __name__ == "__main__":
    main()
