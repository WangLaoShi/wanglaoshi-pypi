"""文档与示例可用性烟测（不依赖外部竞赛数据）。

覆盖：
- 门面导入与关键 API
- analyzer_demo 核心路径（不写完整 HTML 也可）
- BusinessPipeline 最小 CSV 主链
- examples/tutorials/* 门面教程脚本
- advanced/01_iris 进程内执行（代表进阶案例）

退出码：0 全过；1 有失败。
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(name: str) -> None:
    print(f"[PASS] {name}")


def _fail(name: str, err: BaseException) -> None:
    print(f"[FAIL] {name}: {type(err).__name__}: {err}")
    print(traceback.format_exc()[-800:])


def check_imports() -> None:
    from wanglaoshi.runtime import VERSIONS as V, ready
    from wanglaoshi.analyze import DataAnalyzer
    from wanglaoshi.compete import SubmissionGenerator, QuickBaseline
    from wanglaoshi.features import AdvancedFeatureEngineer
    from wanglaoshi.business import (
        BusinessPipeline,
        intake_file,
        generate_insights,
        archive_delivery,
        archive_charts,
        run_eda_profile,
    )

    assert hasattr(V, "show_env_info")
    assert hasattr(V, "check_all_versions")
    assert callable(ready)
    assert callable(DataAnalyzer)
    assert callable(SubmissionGenerator)
    assert callable(QuickBaseline)
    assert callable(AdvancedFeatureEngineer)
    assert callable(BusinessPipeline)
    assert callable(intake_file)
    assert callable(generate_insights)
    assert callable(archive_delivery)
    assert callable(archive_charts)
    assert callable(run_eda_profile)
    _ok("facade imports")


def check_competition_min() -> None:
    import numpy as np
    import pandas as pd
    from wanglaoshi import compete as CT
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    baseline = CT.QuickBaseline()
    results = baseline.run_all_models(Xtr, ytr, Xte, yte, task_type="classification")
    cmp = baseline.compare_models(results, top_n=3)
    assert len(cmp) >= 1
    pred = results[cmp.iloc[0]["Model"]]["predictions"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sample = pd.DataFrame({"Id": np.arange(len(pred)), "target": yte.to_numpy()})
        sample_path = td / "sample.csv"
        sample.to_csv(sample_path, index=False)
        out = td / "sub.csv"
        gen = CT.SubmissionGenerator()
        gen.create_submission(pred, str(sample_path), str(out), prediction_type="label")
        val = gen.validate_submission(str(out), str(sample_path))
        assert val["is_valid"] is True
    _ok("competition baseline + submission")


def check_analyzer_demo_core() -> None:
    # 复用 examples/analyzer_demo.py 的数据构造与基础调用
    sys.path.insert(0, str(ROOT / "examples"))
    import analyzer_demo as demo  # type: ignore

    df = demo.create_sample_data()
    from wanglaoshi.analyze import DataAnalyzer

    analyzer = DataAnalyzer(df)
    info = analyzer.get_basic_info()
    assert info
    stats = analyzer.basic_statistics()
    assert stats is not None and len(stats) > 0
    _ok("analyzer_demo core")


def check_tutorials() -> None:
    import importlib.util

    tutorial_dir = ROOT / "examples" / "tutorials"
    scripts = sorted(tutorial_dir.glob("[0-9][0-9]_*.py"))
    assert scripts, "missing examples/tutorials/NN_*.py"
    for path in scripts:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        _ok(f"tutorial {path.name}")


def check_business_pipeline() -> None:
    import pandas as pd
    from wanglaoshi.business import BusinessPipeline

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "orders.csv"
        pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4],
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "amount": [10.0, 20.0, 15.0, 30.0],
                "channel": ["web", "app", "web", "app"],
                "user_tier": ["A", "B", "A", "B"],
            }
        ).to_csv(src, index=False)
        pipe = BusinessPipeline(
            td / "archive",
            required_columns=["order_id", "amount"],
            date_col="order_date",
            value_col="amount",
            dim_cols=["channel"],
            segment_cols=["user_tier"],
            period="2024-Q1",
        )
        record = pipe.run(src)
        ids = [s.stage_id for s in record.stages]
        assert "D01_intake" in ids and "D10_report_archive" in ids
        assert not record.blocked()
        d10 = record.stages[-1]
        assert d10.data.get("delivery_index", {}).get("deliverable") is True
        report = next(Path(p) for p in d10.output_paths if p.endswith("report.md"))
        assert report.exists() and report.stat().st_size > 0
    _ok("business pipeline")


def check_iris_notebook() -> None:
    # 轻量：调用证据脚本只跑 iris
    import subprocess

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_advanced_evidence.py"),
        "--merge",
        "--only",
        "01_iris",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-500:] + "\n" + proc.stderr[-500:])
    evidence = json.loads((ROOT / "docs" / "evidence" / "advanced_evidence.json").read_text(encoding="utf-8"))
    iris = next(r for r in evidence["results"] if r["slug"] == "01_iris")
    assert iris["status"] == "passed", iris
    _ok("advanced 01_iris notebook")


def main() -> int:
    checks = [
        check_imports,
        check_competition_min,
        check_analyzer_demo_core,
        check_business_pipeline,
        check_tutorials,
        check_iris_notebook,
    ]
    failed = 0
    for fn in checks:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            _fail(fn.__name__, e)
            failed += 1
    print(f"summary: {len(checks) - failed}/{len(checks)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
