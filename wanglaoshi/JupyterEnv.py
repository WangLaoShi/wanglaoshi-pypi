"""
Jupyter环境管理模块 (JupyterEnv.py)

这个模块提供了Jupyter Notebook环境相关的工具函数，包括：
1. 系统命令执行
2. 运行环境检测（操作系统、Python版本等）
3. Jupyter内核管理（安装、列出等）

使用场景：
- 在Jupyter Notebook中检查运行环境
- 安装和配置Jupyter内核
- 获取Python环境信息
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import List, Sequence, Union

# 内核名/显示名：禁止 shell 元字符，避免经 popen/拼接注入
_KERNEL_NAME_RE = re.compile(r"^[\w.-]+$")
_KERNEL_DISPLAY_RE = re.compile(r"^[\w.\- ]+$")


def run_cmd(cmd: Union[str, Sequence[str]]) -> None:
    """
    在系统中执行命令并打印输出。

    优先传入参数列表（无 shell）；若传入字符串则仍走 shell（兼容旧用法，勿拼接用户输入）。
    """
    print("Running command:", cmd)
    if isinstance(cmd, (list, tuple)):
        completed = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (completed.stdout or "") + (completed.stderr or "")
    else:
        # 兼容旧字符串调用；调用方不得把未校验用户输入拼进字符串
        completed = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (completed.stdout or "") + (completed.stderr or "")
    for line in out.split("\n"):
        print(line)


def running_os() -> str:
    """检测当前运行的操作系统。"""
    env = sys.platform
    if env == "win32":
        return "Windows"
    if env == "linux":
        return "Linux"
    if env == "darwin":
        return "MacOS"
    return "Unknown"


def running_python_version() -> str:
    """获取当前运行的Python版本字符串。"""
    return sys.version


def running_python_path() -> str:
    """获取当前运行的Python解释器路径。"""
    return sys.executable


def running_python_version_info():
    """获取当前运行的Python版本详细信息。"""
    return sys.version_info


def running():
    """显示完整的运行环境信息并返回元组。"""
    print("*" * 40)
    print("Running environment:", running_os())
    print("Running Python version:", running_python_version())
    print("Running Python path:", running_python_path())
    print("Running Python version info:", running_python_version_info())
    print("*" * 40)
    return running_os(), running_python_version(), running_python_path(), running_python_version_info()


def install_ipykernel() -> None:
    """安装 ipykernel（参数列表，不经 shell）。"""
    run_cmd([sys.executable, "-m", "pip", "install", "ipykernel"])


def jupyter_kernel_list() -> None:
    """列出已安装的 Jupyter 内核。"""
    run_cmd([sys.executable, "-m", "jupyter", "kernelspec", "list"])


def _validate_kernel_name(name: str) -> str:
    name = (name or "").strip()
    if not _KERNEL_NAME_RE.match(name):
        raise ValueError(
            "kernel name 仅允许字母/数字/下划线/点/连字符，"
            f"收到: {name!r}"
        )
    return name


def _validate_kernel_display_name(name: str) -> str:
    name = (name or "").strip()
    if not _KERNEL_DISPLAY_RE.match(name):
        raise ValueError(
            "display name 仅允许字母/数字/下划线/点/连字符/空格，"
            f"收到: {name!r}"
        )
    return name


def install_kernel(
    kernel_name: str | None = None,
    kernel_display_name: str | None = None,
) -> None:
    """
    安装新的 Jupyter 内核。

    可用参数直接传入名称（测试/脚本）；不传则交互 input()。
    名称经白名单校验后以 subprocess 参数列表执行，避免命令注入。
    """
    jupyter_kernel_list()
    install_ipykernel()
    running()
    python_path = running_python_path()
    print("Please input the new kernel name and display name.")
    if kernel_name is None:
        kernel_name = input("Please input the kernel name:")
    if kernel_display_name is None:
        kernel_display_name = input("Please input the kernel display name:")
    kernel_name = _validate_kernel_name(kernel_name)
    kernel_display_name = _validate_kernel_display_name(kernel_display_name)
    run_cmd(
        [
            python_path,
            "-m",
            "ipykernel",
            "install",
            "--user",
            f"--name={kernel_name}",
            f"--display-name={kernel_display_name}",
        ]
    )


def no_warning() -> None:
    """禁用 Python 警告信息（仅做 filterwarnings('ignore')）。"""
    import warnings

    warnings.filterwarnings("ignore")
