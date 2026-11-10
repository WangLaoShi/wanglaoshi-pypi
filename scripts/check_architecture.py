#!/usr/bin/env python3
"""架构兼容性验收（docs/ARCHITECTURE.md §4）。

检查：
1. 门面 / 根 shim / 顶层导入均可加载，且关键符号 identity 一致
2. 可选：检测 site-packages 里是否残留旧版 wanglaoshi 目录（易抢先）

用法::

    python scripts/check_architecture.py
    ./scripts/run_tests.sh arch
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"OK  {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL {msg}", file=sys.stderr)


def check_imports_and_identity() -> list[str]:
    errors: list[str] = []
    try:
        from wanglaoshi import CompetitionTools, DataAnalyzer, ready
        from wanglaoshi.Analyzer import DataAnalyzer as DA_shim
        from wanglaoshi.CompetitionTools import SubmissionGenerator as SG_ct
        from wanglaoshi.FeatureEngineering import AdvancedFeatureEngineer as FE_shim
        from wanglaoshi._testing import assert_close
        from wanglaoshi.analytics import StatsToolkit
        from wanglaoshi.analyze import DataAnalyzer as DA_facade
        from wanglaoshi.asserting import assert_close as ac_shim
        from wanglaoshi.compete import SubmissionGenerator
        from wanglaoshi.features import AdvancedFeatureEngineer
        from wanglaoshi.runtime import JupyterEnv, VERSIONS, ready as ready_rt
        from wanglaoshi.stats import StatsToolkit as ST_shim

        assert DA_facade is DataAnalyzer is DA_shim
        assert SubmissionGenerator is SG_ct is CompetitionTools.SubmissionGenerator
        assert AdvancedFeatureEngineer is FE_shim
        assert assert_close is ac_shim
        assert StatsToolkit is ST_shim
        assert ready is ready_rt
        assert callable(ready) and StatsToolkit is not None
        assert hasattr(VERSIONS, "show_env_info")
        assert hasattr(JupyterEnv, "no_warning")
        assert StatsToolkit.__module__.startswith("wanglaoshi.analytics.stats.")
        _ok("facade / shim / top-level imports + identity")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import/identity: {exc}")
        _fail(errors[-1])
    return errors


def check_stale_site_packages() -> list[str]:
    """若 site-packages 内存在实体 wanglaoshi/ 目录（非 egg-link），可能抢先于开发树。"""
    warnings: list[str] = []
    try:
        import site

        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:  # noqa: BLE001
        return warnings

    for root in roots:
        if not root:
            continue
        pkg = Path(root) / "wanglaoshi"
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            # egg-link / editable 通常不在 site-packages 下建完整包目录
            # 若同时有 .dist-info 旧版本，提示清理
            msg = (
                f"检测到 site-packages 实体包目录: {pkg} "
                f"（可能与 editable 开发树抢先；建议卸载旧版后 pip install -e .）"
            )
            warnings.append(msg)
            print(f"WARN {msg}")
    if not warnings:
        _ok("no stale site-packages wanglaoshi/ directory")
    return warnings


def main() -> int:
    # 保证优先用仓库内包
    root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    while root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)

    # 清掉可能已加载的旧模块，强制按当前 path 再导
    for name in list(sys.modules):
        if name == "wanglaoshi" or name.startswith("wanglaoshi."):
            del sys.modules[name]

    errors = check_imports_and_identity()
    # 旧 site-packages 只警告，不令检查失败（环境各异）
    check_stale_site_packages()

    if not errors:
        # 再确认模块文件落在仓库内（导入已成功时再查路径）
        analyze_core = importlib.import_module("wanglaoshi.analyze.core")
        core_file = Path(getattr(analyze_core, "__file__", "") or "")
        if root.resolve() not in core_file.resolve().parents:
            errors.append(f"analyze.core 不在仓库内: {core_file}")
            _fail(errors[-1])
        else:
            _ok(f"analyze.core → {core_file.relative_to(root)}")

    if errors:
        print(f"architecture check FAILED ({len(errors)} error(s))", file=sys.stderr)
        print(
            "提示: 请用与开发一致的解释器，例如 "
            "`PYTHON=/opt/anaconda3/bin/python ./scripts/run_tests.sh arch`",
            file=sys.stderr,
        )
        return 1
    print("architecture check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
