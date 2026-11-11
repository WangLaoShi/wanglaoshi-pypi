"""导入隔离：轻量入口应可不经根包兼容 shim；子包可独立导入。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.quick
def test_subprocess_compete_without_root_shim_load() -> None:
    """直接从 compete 导入时，不应主动加载根上 CompetitionTools shim。"""
    proc = _run(
        "import sys\n"
        "from wanglaoshi.compete import SubmissionGenerator\n"
        "assert SubmissionGenerator is not None\n"
        "assert 'wanglaoshi.CompetitionTools' not in sys.modules\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.quick
def test_subprocess_runtime_and_testing_light() -> None:
    proc = _run(
        "from wanglaoshi.runtime import ready\n"
        "from wanglaoshi._testing import assert_close\n"
        "ready(seed=0, quiet=True, show_env=False)\n"
        "assert_close(1.0, 1.0)\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.quick
def test_root_compat_shim_is_lazy() -> None:
    """import wanglaoshi 后，兼容 shim 应尚未进入 sys.modules，直至显式访问。"""
    proc = _run(
        "import sys\n"
        "import wanglaoshi\n"
        "assert 'wanglaoshi.CompetitionTools' not in sys.modules\n"
        "assert 'wanglaoshi.FeatureEngineering' not in sys.modules\n"
        "assert 'wanglaoshi.MLDL' not in sys.modules\n"
        "_ = wanglaoshi.CompetitionTools\n"
        "assert 'wanglaoshi.CompetitionTools' in sys.modules\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
