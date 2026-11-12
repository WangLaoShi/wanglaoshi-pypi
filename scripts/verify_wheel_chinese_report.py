#!/usr/bin/env python3
"""发版验收：从当前源码构建 wheel，在源码树外的临时 venv 安装，再跑中文完整报告。

用法：
  python scripts/verify_wheel_chinese_report.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, **kwargs)


def main() -> int:
    # 使用仓库外的系统临时目录，避免从源码 cwd 误导入本地 wanglaoshi
    with tempfile.TemporaryDirectory(prefix="wl-wheel-zh-") as tmp:
        tmp_path = Path(tmp)
        venv_dir = tmp_path / "venv"
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        for name in ("verify_wheel_assets.py", "verify_chinese_report.py"):
            shutil.copy2(ROOT / "scripts" / name, scripts_dir / name)

        print(f"[1] build wheel from {ROOT}", flush=True)
        _run([sys.executable, "setup.py", "bdist_wheel", "-q"], cwd=str(ROOT))
        wheels = sorted((ROOT / "dist").glob("wanglaoshi-*.whl"))
        if not wheels:
            print("FAIL: no wheel in dist/")
            return 1
        wheel = wheels[-1]
        print(f"[2] wheel={wheel.name}", flush=True)

        print(f"[3] create venv at {venv_dir}", flush=True)
        venv.create(venv_dir, with_pip=True)
        pip = venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
        py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run([str(pip), "install", "-q", "--upgrade", "pip"])
        _run([str(pip), "install", "-q", str(wheel)])

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
        (tmp_path / "mpl").mkdir(exist_ok=True)

        print("[4] verify_wheel_assets in clean venv", flush=True)
        _run(
            [str(py), str(scripts_dir / "verify_wheel_assets.py")],
            cwd=str(tmp_path),
            env=env,
        )
        out_html = tmp_path / "chinese_report_smoke.html"
        print("[5] verify_chinese_report in clean venv", flush=True)
        _run(
            [
                str(py),
                str(scripts_dir / "verify_chinese_report.py"),
                "--out",
                str(out_html),
            ],
            cwd=str(tmp_path),
            env=env,
        )
        if not out_html.exists():
            print(f"FAIL: report missing at {out_html}")
            return 1
        _run(
            [
                str(py),
                "-c",
                "import wanglaoshi, wanglaoshi.analyze.core as c; "
                "from pathlib import Path; "
                "r=Path(wanglaoshi.__file__).resolve().parent; "
                "assert 'site-packages' in str(r) or 'dist-packages' in str(r), r; "
                "assert Path(r)==Path(c.font_path).resolve().parent, (r, c.font_path); "
                "print('installed_at', r); "
                "print('font', c.font_path, c.font_prop.get_name())",
            ],
            cwd=str(tmp_path),
            env=env,
        )
        print(f"OK wheel chinese report {out_html}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
