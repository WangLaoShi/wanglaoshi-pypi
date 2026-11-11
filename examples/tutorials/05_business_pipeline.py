"""Tutorial 05 smoke: business.BusinessPipeline on synthetic orders."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from wanglaoshi.runtime import ready
from wanglaoshi.business import BusinessPipeline


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "orders.csv"
        pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4, 5, 6],
                "order_date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ],
                "amount": [10.5, 20.0, 7.2, 80.0, 15.0, 12.0],
                "channel": ["web", "app", "web", "app", "web", "app"],
                "user_tier": ["A", "B", "A", "A", "B", "B"],
            }
        ).to_csv(src, index=False)

        pipe = BusinessPipeline(
            root / "archive",
            required_columns=["order_id", "amount", "channel"],
            date_col="order_date",
            value_col="amount",
            dim_cols=["channel"],
            segment_cols=["user_tier"],
            period="2024-Q1",
        )
        record = pipe.run(src)
        ids = [s.stage_id for s in record.stages]
        assert "D01_intake" in ids
        assert not record.blocked()
        print("05_business_pipeline OK", record.summary())


if __name__ == "__main__":
    main()
