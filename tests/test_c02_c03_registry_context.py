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
def test_c02_register_without_require_exists(tmp_path: Path) -> None:
    """下载与测试分离：可先登记占位路径，sha 为空、cached=False。"""
    reg = DataRegistry(tmp_path / "ledger.json")
    placeholder = tmp_path / "pending" / "external.csv"
    asset = reg.register(
        "pending",
        placeholder,
        source="kaggle",
        license_or_terms="CC0",
        version="0.1",
        require_exists=False,
    )
    assert asset.sha256 == ""
    assert asset.cached is False
    assert reg.verify("pending") is False
    assert len(reg.list()) == 1


@pytest.mark.quick
def test_c02_verify_false_when_file_deleted(tmp_path: Path) -> None:
    data = tmp_path / "gone.csv"
    data.write_text("x\n1\n", encoding="utf-8")
    reg = DataRegistry(tmp_path / "ledger.json")
    reg.register("gone", data, source="local", license_or_terms="internal", version="1")
    data.unlink()
    assert reg.verify("gone") is False


@pytest.mark.quick
def test_c02_overwrite_same_name(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("v1\n", encoding="utf-8")
    b.write_text("v2-longer\n", encoding="utf-8")
    reg = DataRegistry(tmp_path / "ledger.json")
    reg.register("ds", a, source="s", license_or_terms="t", version="1")
    h1 = reg.get("ds").sha256
    reg.register("ds", b, source="s", license_or_terms="t", version="2")
    assert reg.get("ds").version == "2"
    assert reg.get("ds").sha256 != h1
    assert len(reg.list()) == 1


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


@pytest.mark.quick
def test_c03_maybe_sample_noop_when_small() -> None:
    ctx = ExperimentContext(sample_n=100, random_seed=1)
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = ctx.maybe_sample(df)
    assert len(out) == 3
    assert list(out["x"]) == [1, 2, 3]


@pytest.mark.quick
def test_c03_stratify_fallback_on_tiny_single_class() -> None:
    """稀有类只有 1 条时分层失败：应 warning 后回退，并标记 stratify_applied=False。"""
    ctx = ExperimentContext(random_seed=0, test_size=0.5)
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = [0, 0, 0, 1]  # 类别 1 仅 1 条，分层会失败
    with pytest.warns(UserWarning, match="无法分层"):
        Xtr, Xte, ytr, yte = ctx.split(X, y, stratify=True)
    assert ctx.stratify_applied is False
    assert len(Xtr) + len(Xte) == 4
    assert len(ytr) + len(yte) == 4
    with pytest.raises(ValueError):
        ctx.split(X, y, stratify=True, strict_stratify=True)


@pytest.mark.quick
def test_c03_different_seeds_change_split() -> None:
    df = pd.DataFrame({"x": range(30), "y": [i % 2 for i in range(30)]})
    a = ExperimentContext(random_seed=1, test_size=0.3)
    b = ExperimentContext(random_seed=99, test_size=0.3)
    Xtr_a, _, _, _ = a.split(df[["x"]], df["y"])
    Xtr_b, _, _, _ = b.split(df[["x"]], df["y"])
    assert list(Xtr_a["x"]) != list(Xtr_b["x"])
