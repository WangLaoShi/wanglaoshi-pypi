"""Handbook 01 smoke: DataAnalyzer on tips + Iris."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import seaborn as sns
from sklearn.datasets import load_iris

from wanglaoshi.analyze import DataAnalyzer
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)

    tips = sns.load_dataset("tips")
    az = DataAnalyzer(tips)
    assert az.get_basic_info()
    assert isinstance(az.get_memory_usage(), float)
    assert az.basic_statistics() is not None and len(az.basic_statistics()) > 0
    assert az.missing_value_analysis() is not None
    assert az.duplicate_analysis() is not None
    assert az.outlier_analysis() is not None
    assert az.get_categorical_stats() is not None
    assert az.correlation_analysis() is not None
    assert az.multicollinearity_analysis() is not None
    assert az.normality_test() is not None
    assert isinstance(az.get_time_stats(), dict)
    _ = az.plot_distribution("total_bill")
    _ = az.plot_correlation_heatmap()
    explored = az.explore_dataframe(name="tips", show_plots=False)
    interp = az.interpret_results()
    assert isinstance(explored, dict) and isinstance(interp, dict)

    iris = load_iris(as_frame=True).frame
    az2 = DataAnalyzer(iris)
    pca = az2.pca_analysis()
    assert isinstance(pca, dict) and pca
    print("01_data_analyzer OK", len(tips), len(iris))


if __name__ == "__main__":
    main()
