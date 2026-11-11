"""Tutorial 08 smoke: TimeSeriesTools on synthetic seasonal series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.analytics import TimeSeriesTools


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    t = np.arange(56)
    y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 7)
    df = pd.DataFrame({"y": y})

    lagged = TimeSeriesTools.safe_lag(df, "y", lags=(1, 2), assume_row_order=True)
    assert "y_lag1" in lagged.columns
    rolled = TimeSeriesTools.safe_rolling_mean(lagged, "y", window=3, assume_row_order=True)
    assert "y_roll3" in rolled.columns
    adf = TimeSeriesTools.adf_report(df["y"])
    assert "pvalue" in adf
    dec = TimeSeriesTools.decompose_summary(df["y"], period=7)
    assert dec["period"] == 7
    print("08_timeseries_tools OK", adf.get("pvalue"), dec.get("resid_std"))


if __name__ == "__main__":
    main()
