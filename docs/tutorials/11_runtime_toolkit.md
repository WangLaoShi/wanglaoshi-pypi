# 11 · runtime 工具箱（VERSIONS / Jupyter / 下载）

## 目标

10 分钟摸清 `runtime` 其余公开入口（不只是 `ready`）。

## 用到的门面 API

`VERSIONS` · `JupyterEnv` · `JupyterFont` · `Useful` · `WebGetter`（及 `Wget`）

## 数据集

无需外部数据。

## 完整示例

```python
from wanglaoshi.runtime import ready, VERSIONS, JupyterEnv, JupyterFont, Useful, WebGetter

ready(seed=42, quiet=True, show_env=False)

# 版本表（simple 更快）
print(VERSIONS.check_versions(version="simple"))
VERSIONS.show_env_info()

# 环境辅助：关警告 / 打印运行信息
JupyterEnv.no_warning()
print(JupyterEnv.running_python_version())

# 字体模块：确认入口存在（完整下载见 JupyterFont.simhei）
assert callable(JupyterFont.matplotlib_font_init)
assert callable(JupyterFont.fonts)

# 依赖/样式帮助入口
assert callable(Useful.dep)
assert callable(Useful.helps)

# 下载器：构造即可；真正下载请换可信 URL
wget = WebGetter.Wget(url="https://example.com/demo.csv", output_dir="outputs/wget_demo")
assert hasattr(wget, "download")
```

配套脚本：[`examples/tutorials/11_runtime_toolkit.py`](../../examples/tutorials/11_runtime_toolkit.py)

## 常见坑

- `WebGetter.Wget.download()` 会访问网络，CI/课堂请限速与校验目标
- `JupyterFont.simhei()` 可能下载字体，首次较慢

## 下一步

- analyze 函数式入口 → [12_analyze_helpers.md](12_analyze_helpers.md)
