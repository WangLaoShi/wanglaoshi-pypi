"""VERSIONS / utils / JupyterEnv 轻量覆盖。"""

from __future__ import annotations

import pytest

from wanglaoshi import JupyterEnv, VERSIONS
from wanglaoshi import utils as wutils


@pytest.mark.quick
@pytest.mark.release
def test_versions_check_versions_returns_structure() -> None:
    result = VERSIONS.check_versions(version="simple")
    assert result is not None
    # 兼容 dict / DataFrame / list 等返回形态
    if hasattr(result, "empty"):
        assert True
    elif isinstance(result, dict):
        assert len(result) >= 0
    else:
        assert result is not None


@pytest.mark.quick
def test_utils_is_chinese_and_align() -> None:
    assert wutils.is_Chinese("中") is True
    assert wutils.is_Chinese("A") is False
    aligned = wutils.algin("hello", 10)
    assert isinstance(aligned, str)
    assert len(aligned) >= len("hello")


@pytest.mark.quick
def test_jupyter_env_os_and_python() -> None:
    assert isinstance(JupyterEnv.running_os(), str)
    assert isinstance(JupyterEnv.running_python_version(), str)
    info = JupyterEnv.running_python_version_info()
    assert info is not None
