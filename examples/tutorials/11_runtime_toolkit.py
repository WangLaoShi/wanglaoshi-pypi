"""Tutorial 11 smoke: runtime VERSIONS / Jupyter* / Useful / WebGetter."""

from __future__ import annotations

from wanglaoshi.runtime import (
    JupyterEnv,
    JupyterFont,
    Useful,
    VERSIONS,
    WebGetter,
    ready,
)


def main() -> None:
    ready(seed=42, quiet=True, show_env=False)
    versions = VERSIONS.check_versions(version="simple")
    assert isinstance(versions, dict) and versions
    VERSIONS.show_env_info()
    JupyterEnv.no_warning()
    assert JupyterEnv.running_python_version()
    assert callable(JupyterFont.matplotlib_font_init)
    assert callable(JupyterFont.fonts)
    assert callable(Useful.dep) and callable(Useful.helps)
    wget = WebGetter.Wget(url="https://example.com/demo.csv", output_dir=".")
    assert hasattr(wget, "download")
    print("11_runtime_toolkit OK", len(versions))


if __name__ == "__main__":
    main()
