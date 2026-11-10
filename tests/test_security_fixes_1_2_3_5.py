"""回归：修复 pytest 入口依赖 / kernel 注入 / 分层告警 / WebGetter 路径与超时。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.mark.quick
def test_init_does_not_eager_import_testing_utils() -> None:
    """包入口不再顶层 import testing_utils（避免强依赖 pytest）。"""
    import wanglaoshi as wl

    init_src = Path(wl.__file__).read_text(encoding="utf-8")
    assert "from .testing_utils import" not in init_src
    assert "def __getattr__" in init_src
    # 符号仍可通过懒加载拿到（当前环境有 pytest）
    assert callable(wl.require_data_path)
    assert callable(wl.optional_data_available)


@pytest.mark.quick
def test_install_kernel_rejects_injection() -> None:
    from wanglaoshi import JupyterEnv

    with patch.object(JupyterEnv, "jupyter_kernel_list"), patch.object(
        JupyterEnv, "install_ipykernel"
    ), patch.object(JupyterEnv, "running"), patch.object(
        JupyterEnv, "run_cmd"
    ) as mock_run:
        with pytest.raises(ValueError, match="仅允许"):
            JupyterEnv.install_kernel(kernel_name="x; rm -rf /", kernel_display_name="ok")
        mock_run.assert_not_called()

        JupyterEnv.install_kernel(kernel_name="my-kernel", kernel_display_name="My Kernel")
        args = mock_run.call_args[0][0]
        assert isinstance(args, list)
        assert args[1:4] == ["-m", "ipykernel", "install"]
        assert "--name=my-kernel" in args


@pytest.mark.quick
def test_experiment_context_records_stratify_flag() -> None:
    from wanglaoshi.experiment_context import ExperimentContext

    ctx = ExperimentContext(random_seed=1, test_size=0.3)
    df = pd.DataFrame({"x": range(40), "y": [i % 2 for i in range(40)]})
    ctx.split(df[["x"]], df["y"], stratify=True)
    assert ctx.stratify_applied is True


@pytest.mark.quick
def test_wget_timeout_and_path_guard(tmp_path: Path) -> None:
    from wanglaoshi.WebGetter import Wget

    w = Wget("https://example.com/a.bin", output_dir=str(tmp_path), filename="a.bin")
    assert w.timeout == (10, 60)

    class _Head:
        headers = {"content-length": str(10**12)}

    with patch("wanglaoshi.WebGetter.requests.head", return_value=_Head()):
        assert w.download() is False  # 超过 max_bytes

    traversed = Wget(
        "https://example.com/a.bin",
        output_dir=str(tmp_path),
        filename="../outside.bin",
    )._safe_output_path()
    assert traversed.parent == tmp_path.resolve()
    assert traversed.name == "outside.bin"
