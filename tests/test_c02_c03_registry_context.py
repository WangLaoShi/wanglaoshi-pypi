"""C02 数据登记 + C03 实验划分/种子/环境记录。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wanglaoshi.data_registry import DataRegistry, file_sha256
from wanglaoshi.experiment_context import ExperimentContext


@pytest.mark.quick
def test_c02_register_and_verify(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    reg = DataRegistry(ledger)
    asset = reg.register(
        "sample",
        data,
        source="synthetic",
        license_or_terms="internal-test",
        version="1.0",
        notes="unit test",
    )
    assert asset.sha256 == file_sha256(data)
    assert reg.verify("sample") is True
    # 篡改后校验失败
    data.write_text("a,b\n9,9\n", encoding="utf-8")
    assert reg.verify("sample") is False
    # 重载台账
    reg2 = DataRegistry(ledger)
    assert reg2.get("sample").version == "1.0"


@pytest.mark.quick
def test_c02_missing_file_raises(tmp_path: Path) -> None:
    reg = DataRegistry(tmp_path / "ledger.json")
    with pytest.raises(FileNotFoundError):
        reg.register(
            "missing",
            tmp_path / "nope.csv",
            source="x",
            license_or_terms="x",
            version="0",
        )


@pytest.mark.quick
def test_c03_split_seed_and_env(tmp_path: Path) -> None:
    ctx = ExperimentContext(random_seed=7, test_size=0.25, sample_n=20, search_budget=5)
    df = pd.DataFrame({"x": range(40), "y": [i % 2 for i in range(40)]})
    sampled = ctx.maybe_sample(df)
    assert len(sampled) == 20
    X = sampled[["x"]]
    y = sampled["y"]
    Xtr1, Xte1, ytr1, yte1 = ctx.split(X, y, stratify=True)
    Xtr2, Xte2, ytr2, yte2 = ctx.split(X, y, stratify=True)
    # 同种子可复现
    assert list(Xtr1["x"]) == list(Xtr2["x"])
    assert list(yte1) == list(yte2)
    out = tmp_path / "ctx.json"
    ctx.to_json(str(out))
    payload = out.read_text(encoding="utf-8")
    assert "random_seed" in payload
    assert "environment" in payload
    assert "python" in payload
    env = ctx.environment()
    assert "numpy" in env and "pandas" in env
