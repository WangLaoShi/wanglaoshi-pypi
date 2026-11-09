"""Notebook / 课堂一键就绪入口。

历史上很多示例写成::

    import warnings
    warnings.filterwarnings('ignore')
    from wanglaoshi import VERSIONS as V
    from wanglaoshi import JupyterEnv as JE
    JE.no_warning()
    V.show_env_info()

其中 ``warnings.filterwarnings`` 与 ``JE.no_warning()`` **完全重复**。
请改用本模块的 :func:`ready`。
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def ready(
    *,
    quiet: bool = True,
    show_env: bool = True,
    seed: Optional[int] = 42,
    check_core: bool = False,
) -> Dict[str, Any]:
    """一键准备课堂 / Notebook 环境。

    参数:
        quiet: 是否忽略 Python warnings（等价于旧的 JE.no_warning）
        show_env: 是否打印 OS / Python / 虚拟环境信息
        seed: 若给出，则设置 ``random`` 与 ``numpy`` 种子；``None`` 表示不设
        check_core: 若 True，额外检查 numpy/pandas/sklearn 是否可导入

    返回:
        本次生效的配置字典，便于写进日志或断言。

    示例::

        from wanglaoshi import ready
        ready()
    """
    if quiet:
        import warnings

        warnings.filterwarnings("ignore")

    if seed is not None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:  # noqa: BLE001
            pass

    if show_env:
        from . import VERSIONS

        VERSIONS.show_env_info()

    core: Dict[str, Any] = {}
    if check_core:
        for name in ("numpy", "pandas", "sklearn"):
            try:
                __import__(name if name != "sklearn" else "sklearn")
                core[name] = "ok"
            except Exception as e:  # noqa: BLE001
                core[name] = f"missing: {e}"

    return {
        "quiet": quiet,
        "show_env": show_env,
        "seed": seed,
        "core": core,
    }
