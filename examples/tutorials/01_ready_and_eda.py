"""Tutorial 01 smoke: runtime.ready + analyze.DataAnalyzer on Iris / tips."""

from __future__ import annotations

from sklearn.datasets import load_iris
import seaborn as sns

from wanglaoshi.runtime import ready
from wanglaoshi.analyze import DataAnalyzer


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)

    iris = load_iris(as_frame=True)
    az = DataAnalyzer(iris.frame)
    info = az.get_basic_info()
    stats = az.basic_statistics()
    miss = az.missing_value_analysis()
    assert info
    assert stats is not None and len(stats) > 0
    assert miss is not None

    tips = sns.load_dataset("tips")
    az2 = DataAnalyzer(tips)
    assert az2.basic_statistics() is not None
    print("01_ready_and_eda OK", len(iris.frame), len(tips))


if __name__ == "__main__":
    main()
