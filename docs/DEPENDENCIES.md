# F02 依赖说明：核心 vs 可选

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 F02。

## 核心（`setup.py` install_requires）

| 包 | 用途 |
| --- | --- |
| pandas / numpy | 数据处理 |
| matplotlib / seaborn / missingno | 可视化与缺失图 |
| scikit-learn / scipy / statsmodels | 建模与统计 |
| jinja2 | HTML 报告模板 |
| rich / tqdm / requests | 体验与下载 |

`python_requires>=3.8`。

## 可选

| 包 | 模块 | 未安装时行为 |
| --- | --- | --- |
| `torch` | `wanglaoshi.MLDL` | 顶层 `import wanglaoshi` 不失败；`MLDL` 为 `None` |
| `openpyxl` / `xlrd` | Excel 读写（若用户读 xlsx） | 读 Excel 时由 pandas 报错提示安装 |
| `jupyter` / `seaborn`（演示） | `examples/advanced` | 演示环境按 README 另行安装 |

## 验证

```bash
python -c "import wanglaoshi; from wanglaoshi import BusinessPipeline, DataAnalyzer, CompetitionTools; print('ok')"
bash scripts/run_tests.sh release
```
