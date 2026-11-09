"""F01/F02 发布烟测：导入与核心 API 可用性。"""

from __future__ import annotations

import pytest


@pytest.mark.quick
@pytest.mark.release
def test_import_wanglaoshi_core():
    import wanglaoshi
    from wanglaoshi import (
        BusinessPipeline,
        CompetitionTools,
        DataAnalyzer,
        VERSIONS,
        archive_delivery,
        generate_insights,
    )

    assert wanglaoshi.__name__ == "wanglaoshi"
    assert BusinessPipeline is not None
    assert CompetitionTools is not None
    assert DataAnalyzer is not None
    assert callable(generate_insights)
    assert callable(archive_delivery)
    assert hasattr(VERSIONS, "show_env_info")


@pytest.mark.quick
@pytest.mark.release
def test_mldl_optional_does_not_break_import():
    import wanglaoshi

    # torch 未安装时 MLDL 可为 None；安装后应为模块
    assert "MLDL" in dir(wanglaoshi)
