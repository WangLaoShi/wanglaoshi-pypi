"""Analyzer / FeatureEngineering / CompetitionTools 最小烟测。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wanglaoshi import VERSIONS
from wanglaoshi.Analyzer import DataAnalyzer, load_data
from wanglaoshi.CompetitionTools import LeakageDetector, SubmissionGenerator
from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer


@pytest.mark.quick
def test_versions_show_env_info_callable() -> None:
    assert callable(VERSIONS.show_env_info)
    assert callable(VERSIONS.check_versions)
    libs = VERSIONS.get_libraries("simple")
    assert isinstance(libs, dict) and len(libs) > 0


@pytest.mark.quick
def test_analyzer_basic_stats_and_missing(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "amount": [10.0, 20.0, None, 40.0],
            "channel": ["web", "app", "web", "app"],
            "const": [1, 1, 1, 1],
        }
    )
    analyzer = DataAnalyzer(df)
    assert "amount" in analyzer.numeric_cols
    assert "channel" in analyzer.categorical_cols

    stats = analyzer.basic_statistics()
    assert not stats.empty
    assert "amount" in set(stats["列名"].astype(str))

    missing = analyzer.missing_value_analysis()
    assert not missing.empty

    src = tmp_path / "tiny.csv"
    df.to_csv(src, index=False)
    loaded = load_data(str(src))
    assert len(loaded) == 4


@pytest.mark.quick
def test_feature_engineering_datetime_and_frequency() -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-01-06", "2024-01-01"],
            "channel": ["web", "app", "web"],
            "amount": [1.0, 2.0, 3.0],
        }
    )
    fe = AdvancedFeatureEngineer()
    out = fe.extract_datetime_features(df, "order_date", features=["year", "month", "weekday", "is_weekend"])
    assert "order_date_year" in out.columns
    assert "order_date_is_weekend" in out.columns
    assert out.loc[1, "order_date_is_weekend"] == 1  # 2024-01-06 is Saturday

    enc = fe.frequency_encode(out, "channel")
    assert "channel_frequency" in enc.columns
    assert abs(float(enc.loc[0, "channel_frequency"]) - 2 / 3) < 1e-9


@pytest.mark.quick
def test_leakage_detector_high_correlation() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    X = pd.DataFrame(
        {
            "almost_y": y * 1.0,  # perfect correlation
            "noise": [0.1, -0.2, 0.05, 0.0, 0.2],
        }
    )
    hits = LeakageDetector().detect_target_leakage(X, y, threshold=0.95)
    names = {h["feature"] if isinstance(h, dict) else h for h in hits}
    assert "almost_y" in names
    assert "noise" not in names


@pytest.mark.quick
@pytest.mark.submission
def test_submission_generator_roundtrip_smoke(tmp_path: Path) -> None:
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(sample_path, index=False)
    out = tmp_path / "sub.csv"
    gen = SubmissionGenerator()
    df = gen.create_submission([0, 1, 1], str(sample_path), str(out))
    assert list(df.columns) == ["id", "target"]
    result = gen.validate_submission(str(out), str(sample_path))
    assert result["is_valid"] is True
