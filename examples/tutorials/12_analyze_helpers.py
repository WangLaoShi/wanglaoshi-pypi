"""Tutorial 12 smoke: analyze helpers beyond DataAnalyzer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from sklearn.datasets import load_iris

from wanglaoshi.analyze import (
    DataAnalyzer,
    DataProcessor,
    ReportGenerator,
    Visualizer,
    analyze_data,
    analyze_multiple_files,
    analyze_notebook,
    load_data,
)
from wanglaoshi.runtime import ready


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        csv_path = td_path / "iris.csv"
        load_iris(as_frame=True).frame.to_csv(csv_path, index=False)
        (td_path / "iris2.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

        df = load_data(str(csv_path))
        DataProcessor.validate_dataframe(df)
        assert DataProcessor.get_column_types(df)

        summary = analyze_notebook(df.head(40), show_plots=False)
        assert isinstance(summary, dict)

        out_html = td_path / "one.html"
        analyze_data(str(csv_path), output_html=str(out_html))
        assert out_html.exists()

        multi_dir = td_path / "reports"
        analyze_multiple_files(str(td_path), output_dir=str(multi_dir))
        assert multi_dir.exists()

        az = DataAnalyzer(df.head(50))
        viz: Visualizer = az.visualizer
        viz.plot_distribution(df["sepal length (cm)"], "sepal length (cm)")
        viz.plot_correlation_heatmap(az.correlation_analysis())
        viz.clear_cache()
        assert hasattr(ReportGenerator, "generate_report")
    print("12_analyze_helpers OK")


if __name__ == "__main__":
    main()
