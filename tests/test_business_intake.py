"""D01 intake / pipeline 最小回归。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wanglaoshi.business import BusinessPipeline, intake_file, structure_dataframe


def test_intake_archives_and_hashes(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1, 2], "amount": [10.0, 20.0]}).to_csv(src, index=False)
    archive = tmp_path / "archive"

    result = intake_file(src, archive, required_columns=["order_id", "amount"], period="2024-Q1")
    assert result.status == "passed"
    assert result.data["sha256"]
    assert result.data["missing_columns"] == []
    assert any(Path(p).name.startswith("orders__") for p in result.output_paths)
    ledger = archive / "intake_ledger.jsonl"
    assert ledger.exists()
    assert "sha256" in ledger.read_text(encoding="utf-8")


def test_intake_missing_columns_warning(tmp_path: Path) -> None:
    src = tmp_path / "orders.csv"
    pd.DataFrame({"order_id": [1]}).to_csv(src, index=False)
    result = intake_file(src, tmp_path / "archive", required_columns=["order_id", "amount"])
    assert result.status == "warning"
    assert "amount" in result.data["missing_columns"]


def test_intake_missing_file_blocked(tmp_path: Path) -> None:
    result = intake_file(tmp_path / "nope.csv", tmp_path / "archive")
    assert result.status == "blocked"
    assert result.block_reasons


def test_pipeline_stops_on_block(tmp_path: Path) -> None:
    pipe = BusinessPipeline(tmp_path / "archive", stop_on_block=True)
    record = pipe.run(tmp_path / "missing.csv")
    assert record.blocked()
    assert record.stages[0].stage_id == "D01_intake"
    assert record.stages[1].status == "skipped"


def test_pipeline_happy_path_placeholders(tmp_path: Path) -> None:
    src = tmp_path / "sales.csv"
    pd.DataFrame({"user_id": [1], "gmv": [9.9]}).to_csv(src, index=False)
    pipe = BusinessPipeline(tmp_path / "archive", required_columns=["user_id", "gmv"])
    record = pipe.run(src)
    assert not record.blocked()
    assert record.stages[0].status == "passed"
    assert record.stages[1].stage_id == "D02_structuring"
    assert record.stages[1].status == "passed"
    assert any(s.status == "skipped" for s in record.stages[2:])


def test_structure_normalizes_columns() -> None:
    df = pd.DataFrame({"Order ID": [1], "GMV ($)": ["3.5"]})
    result = structure_dataframe(df, dtype_map={"gmv": "float64"})
    assert result.status in {"passed", "warning"}
    assert "order_id" in result.data["columns"]
    assert "gmv" in result.data["columns"]
    assert result.frame["gmv"].dtype == "float64"  # type: ignore[attr-defined]
