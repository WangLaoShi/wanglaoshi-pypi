"""C03：固定划分 / 种子 / 抽样与搜索预算，并记录环境。"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class ExperimentContext:
    random_seed: int = 42
    test_size: float = 0.3
    sample_n: Optional[int] = None
    search_budget: int = 20  # 例如 RandomizedSearch 迭代上限
    notes: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def environment(self) -> Dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }

    def maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_n is None or len(df) <= self.sample_n:
            return df.copy()
        return df.sample(n=self.sample_n, random_state=self.random_seed).reset_index(drop=True)

    def split(
        self,
        X: pd.DataFrame,
        y: Sequence,
        *,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
        strat = y if stratify else None
        try:
            return train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=strat,
            )
        except ValueError:
            # 小样本无法分层时回退
            return train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_seed, stratify=None
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["environment"] = self.environment()
        return d

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
