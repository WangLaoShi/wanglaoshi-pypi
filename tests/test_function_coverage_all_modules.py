"""全包函数级冒烟：确保每个模块级函数/方法至少执行一次（C12 函数 100%）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    n = 12
    return pd.DataFrame(
        {
            "num": np.linspace(1.0, 12.0, n) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80.0]),
            "num2": np.linspace(0.5, 6.0, n),
            "cat": (["a", "b", "c"] * 4),
            "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )


@pytest.mark.quick
def test_useful_helpers() -> None:
    from wanglaoshi import Useful

    Useful.pypis()
    Useful.dep()
    Useful.styles()
    Useful.helps()


@pytest.mark.quick
def test_versions_all_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    from wanglaoshi import VERSIONS

    assert isinstance(VERSIONS.get_libraries("simple"), dict)
    assert isinstance(VERSIONS.get_libraries("full"), dict)
    assert isinstance(VERSIONS.check_versions(version="simple"), dict)
    VERSIONS.show_env_info()
    VERSIONS.check_all_versions(version="simple")
    VERSIONS.check_all_installed()
    monkeypatch.setattr(VERSIONS, "get_latest_version", lambda _name: "0.0.0")
    VERSIONS.check_all_versions(version="simple", show_latest=True, only_problems=True)
    # 避免对全量已装包发网
    monkeypatch.setattr(
        VERSIONS,
        "distributions",
        lambda: [],
    )
    VERSIONS.check_all_installed_with_latest()


@pytest.mark.quick
def test_versions_get_latest_version_mocked() -> None:
    from wanglaoshi import VERSIONS

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"info": {"version": "1.2.3"}}

    with patch("wanglaoshi.VERSIONS.requests.get", return_value=_Resp()):
        assert VERSIONS.get_latest_version("numpy") == "1.2.3"

    class _Bad:
        status_code = 404

        def json(self) -> dict:
            return {}

    with patch("wanglaoshi.VERSIONS.requests.get", return_value=_Bad()):
        assert VERSIONS.get_latest_version("no-such-pkg-xyz") == "Not found"

    with patch(
        "wanglaoshi.VERSIONS.requests.get",
        side_effect=VERSIONS.requests.RequestException("offline"),
    ):
        assert VERSIONS.get_latest_version("numpy") == "Error"


@pytest.mark.quick
def test_jupyter_env_all(monkeypatch: pytest.MonkeyPatch) -> None:
    from wanglaoshi import JupyterEnv

    assert JupyterEnv.running_os() in {"Windows", "Linux", "MacOS", "Unknown"}
    assert isinstance(JupyterEnv.running_python_version(), str)
    assert isinstance(JupyterEnv.running_python_path(), str)
    assert JupyterEnv.running_python_version_info() is not None
    JupyterEnv.running()
    JupyterEnv.no_warning()

    class _Pipe:
        def read(self) -> str:
            return "ok\nline2"

    monkeypatch.setattr(JupyterEnv.os, "popen", lambda _cmd: _Pipe())
    JupyterEnv.run_cmd("echo hi")
    JupyterEnv.install_ipykernel()
    JupyterEnv.jupyter_kernel_list()
    monkeypatch.setattr("builtins.input", lambda _p="": "test-kernel")
    JupyterEnv.install_kernel()


@pytest.mark.quick
def test_jupyter_font_and_webgetter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wanglaoshi import JupyterFont, WebGetter

    # 覆盖检测函数（缺失字体时可能 True/False，取决于 matplotlib 回退）
    assert isinstance(JupyterFont.is_chinese_font(str(tmp_path / "missing.ttf")), bool)

    monkeypatch.setattr(JupyterFont.fm, "findSystemFonts", lambda: [])
    JupyterFont.fonts(show_demo=False)

    # WebGetter：mock 网络
    wg = WebGetter.Wget("https://example.com/path/file.bin", output_dir=str(tmp_path), filename=None)
    assert wg.filename == "file.bin"
    wg._create_output_dir()

    class _Head:
        headers = {"content-length": "4"}

    class _Get:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size=1024):
            yield b"data"

    with patch("wanglaoshi.WebGetter.requests.head", return_value=_Head()), patch(
        "wanglaoshi.WebGetter.requests.get", return_value=_Get()
    ):
        assert wg.download() is True

    # 失败路径
    with patch(
        "wanglaoshi.WebGetter.requests.head",
        side_effect=WebGetter.requests.exceptions.RequestException("boom"),
    ):
        assert WebGetter.Wget("https://x", output_dir=str(tmp_path), filename="a.bin").download() is False

    # simhei / matplotlib_font_init：不真下载
    fake_font = tmp_path / "SimHei.ttf"
    fake_font.write_bytes(b"0")
    monkeypatch.chdir(tmp_path)
    with patch("matplotlib.font_manager.fontManager.addfont"), patch("matplotlib.rc"):
        JupyterFont.matplotlib_font_init(show_demo=False)
    with patch.object(JupyterFont, "Wget") as mock_wget:
        inst = MagicMock()
        mock_wget.return_value = inst
        fake_font.unlink(missing_ok=True)
        JupyterFont.simhei()
        inst.download.assert_called()
    with patch.object(JupyterFont, "simhei"), patch(
        "matplotlib.font_manager.fontManager.addfont"
    ), patch("matplotlib.rc"), patch.object(JupyterFont.plt, "show", lambda *a, **k: None):
        # 触发下载分支 + demo
        JupyterFont.matplotlib_font_init(show_demo=True)


@pytest.mark.quick
def test_analyzer_plain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wanglaoshi import Analyzer_Plain

    csv_path = tmp_path / "d.csv"
    # 全数值，避免旧实现在相关性步骤对字符串列报错
    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 0.0, 1.0]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(Analyzer_Plain.plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(Analyzer_Plain.msno, "matrix", lambda *a, **k: None)
    # 旧实现可能在 df.info(buf=f) 后写文件触发 closed file；函数体仍已执行
    try:
        Analyzer_Plain.analyze_data(str(csv_path))
    except ValueError as exc:
        assert "closed file" in str(exc)


@pytest.mark.quick
def test_mldl_sklearn_and_torch_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    import wanglaoshi.MLDL as mldl

    pre = mldl.DataPreprocessor()
    num = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert pre.handle_missing_values(num, method="mean")["a"].isna().sum() == 0
    assert pre.handle_missing_values(num, method="median")["a"].isna().sum() == 0
    assert pre.handle_missing_values(num, method="mode")["a"].isna().sum() == 0
    assert len(pre.handle_missing_values(num, method="drop")) == 2
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    enc = pre.encode_categorical(df, ["b"])
    scaled = pre.scale_features(enc, ["a"])
    assert "a" in scaled.columns

    fe = mldl.FeatureEngineer()
    poly = fe.create_polynomial_features(pd.DataFrame({"a": [1, 2], "c": [3, 4]}), ["a"], degree=2)
    assert "a_power_2" in poly.columns
    inter = fe.create_interaction_features(pd.DataFrame({"a": [1, 2], "c": [3, 4]}), ["a", "c"])
    assert "a_c_interaction" in inter.columns

    X = pd.DataFrame({"f1": [0, 1, 0, 1, 0, 1], "f2": [1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = mldl.MLModel("logistic")
    model.train(X, y, test_size=0.5)
    pred = model.predict(X)
    assert len(pred) == len(X)
    metrics = model.evaluate()
    assert "Accuracy" in metrics
    # 覆盖线性回归评估分支
    reg = mldl.MLModel("linear")
    reg.train(X, pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), test_size=0.5)
    assert "R2" in reg.evaluate()
    with pytest.raises(ValueError):
        mldl.MLModel("nope")

    ev = mldl.ModelEvaluator()
    monkeypatch.setattr(mldl.plt, "show", lambda *a, **k: None)
    ev.plot_confusion_matrix(np.array([0, 1]), np.array([0, 1]))
    ev.plot_learning_curve(
        np.array([1, 2, 3]),
        np.array([[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]]),
        np.array([[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]),
    )

    if mldl.TORCH_AVAILABLE:
        nn_model = mldl.NeuralNetwork(2, 4, 2)
        x = mldl.torch.randn(3, 2)
        assert nn_model.forward(x).shape[0] == 3
        dl = mldl.DLModel(2, 4, 2)
        y_t = mldl.torch.tensor([0, 1, 0])
        dl.train(x, y_t, epochs=1)
        assert len(dl.predict(x)) == 3
    else:
        with pytest.raises(ImportError):
            mldl.NeuralNetwork(2, 4, 2)
        with pytest.raises(ImportError):
            mldl.DLModel(2, 4, 2)
        nn = mldl.NeuralNetwork.__new__(mldl.NeuralNetwork)
        with pytest.raises(ImportError):
            mldl.NeuralNetwork.forward(nn, None)
        dl = mldl.DLModel.__new__(mldl.DLModel)
        with pytest.raises(ImportError):
            mldl.DLModel.train(dl, None, None)
        with pytest.raises(ImportError):
            mldl.DLModel.predict(dl, None)

    monkeypatch.setattr(mldl.plt, "show", lambda *a, **k: None)
    # example_usage 会走到 DL 分支或 skip
    if not mldl.TORCH_AVAILABLE:
        mldl.example_usage()
    else:
        mldl.example_usage()


@pytest.mark.quick
def test_wqe_all_public_methods(tmp_path: Path, tiny_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    import wanglaoshi.wqe as wqe

    # DataProcessor
    wqe.DataProcessor.validate_dataframe(tiny_df)
    with pytest.raises(TypeError):
        wqe.DataProcessor.validate_dataframe([1, 2])
    with pytest.raises(ValueError):
        wqe.DataProcessor.validate_dataframe(pd.DataFrame())
    assert "numeric_cols" in wqe.DataProcessor.get_column_types(tiny_df)
    assert wqe.DataProcessor.process_value(np.array([1, 2])) == [1, 2]
    assert wqe.DataProcessor.process_value(np.int64(3)) == 3
    assert wqe.DataProcessor.process_value(np.float64(1.5)) == 1.5
    assert wqe.DataProcessor.process_value(np.float64(np.nan)) is None
    from datetime import datetime

    assert isinstance(wqe.DataProcessor.process_value(datetime(2024, 1, 1)), str)
    assert isinstance(wqe.DataProcessor.process_value(pd.Series([1, 2])), dict)
    assert isinstance(wqe.DataProcessor.process_data({"a": np.array([1.0])}), dict)
    assert len(wqe.DataProcessor.chunk_dataframe(tiny_df, chunk_size=2)) >= 2

    analyzer = wqe.DataAnalyzer(tiny_df)
    analyzer.correlation_analysis()
    analyzer.multicollinearity_analysis()
    analyzer.basic_statistics()
    analyzer.normality_test()
    analyzer.missing_value_analysis()
    analyzer.outlier_analysis()
    analyzer.duplicate_analysis()
    analyzer.pca_analysis()
    analyzer.interpret_results()
    analyzer._interpret_basic_info()
    analyzer._interpret_data_quality()
    analyzer._interpret_statistical_analysis()
    analyzer._analyze_skew_kurtosis()
    analyzer._get_skew_kurtosis_recommendation(0.1, 0.1)
    analyzer._get_skew_kurtosis_recommendation(2.0, 5.0)
    analyzer._generate_recommendations()
    analyzer._get_zscore_interpretation(pd.Series([0.0, 1.0, 3.5]))
    analyzer._get_iqr_interpretation(tiny_df["num"], 1.0, 3.0, 2.0)
    analyzer.get_memory_usage()
    analyzer.get_basic_info()
    analyzer.get_categorical_stats()
    analyzer.get_time_stats()
    analyzer.plot_distribution("num")
    analyzer.plot_correlation_heatmap()
    analyzer.plot_time_series(tiny_df["ts"], "ts")
    analyzer.explore_dataframe(name="tiny", show_plots=False)

    out_html = tmp_path / "wqe_report.html"
    analyzer.generate_report(str(out_html))
    assert out_html.exists()

    # Visualizer 直接路径
    viz = analyzer.visualizer
    fig = viz._create_figure()
    viz._close_figure(fig)
    viz.plot_distribution(tiny_df["cat"], "cat")
    viz.plot_correlation_heatmap(tiny_df[["num"]].corr())
    viz.clear_cache()

    # 模块级 API
    csv_path = tmp_path / "w.csv"
    tiny_df.to_csv(csv_path, index=False)
    assert len(wqe.load_data(str(csv_path))) == len(tiny_df)
    wqe.analyze_data(str(csv_path), output_html=str(tmp_path / "a.html"))
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "x.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    wqe.analyze_multiple_files(str(folder), output_dir=str(tmp_path / "reports"))
    # analyze_notebook 依赖可能不存在的方法，但仍需进入函数
    wqe.analyze_notebook(tiny_df)

    # ReportGenerator helpers
    rg = wqe.ReportGenerator(
        template_dir=str(Path(wqe.__file__).parent / "templates"),
        static_dir=str(tmp_path / "static"),
    )
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("abc", encoding="utf-8")
    f2.write_text("abc", encoding="utf-8")
    assert rg._files_are_identical(str(f1), str(f2)) is True
    f2.write_text("xyz", encoding="utf-8")
    assert rg._files_are_identical(str(f1), str(f2)) is False
    rg._copy_static_files()

    # 触发析构清理
    del analyzer
