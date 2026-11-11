# 12 · analyze 函数与辅助类

## 目标

覆盖 `DataAnalyzer` 之外的公开入口：`load_data` / `analyze_*` / `DataProcessor` / `Visualizer` / `ReportGenerator`。

## 用到的门面 API

`load_data` · `analyze_data` · `analyze_multiple_files` · `analyze_notebook` · `DataProcessor` · `Visualizer` · `ReportGenerator`

## 数据集

临时 CSV（Iris 写出）。

## 完整示例

```python
from pathlib import Path
import tempfile
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use("Agg")

from wanglaoshi.runtime import ready
from wanglaoshi.analyze import (
    load_data,
    analyze_data,
    analyze_multiple_files,
    analyze_notebook,
    DataProcessor,
    Visualizer,
    ReportGenerator,
)

ready(seed=42, quiet=True, show_env=False)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    csv_path = td / "iris.csv"
    load_iris(as_frame=True).frame.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))
    DataProcessor.validate_dataframe(df)
    print(DataProcessor.get_column_types(df))

    # Notebook 风格摘要（不弹图）
    summary = analyze_notebook(df.head(40), show_plots=False)
    assert isinstance(summary, dict)

    out_html = td / "one.html"
    analyze_data(str(csv_path), output_html=str(out_html))
    assert out_html.exists()

    # 目录内多 CSV → 多份报告
    (td / "iris2.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    multi_dir = td / "reports"
    analyze_multiple_files(str(td), output_dir=str(multi_dir))

    from wanglaoshi.analyze import DataAnalyzer

    az = DataAnalyzer(df.head(50))
    viz: Visualizer = az.visualizer
    viz.plot_distribution(df["sepal length (cm)"], "sepal length (cm)")
    viz.plot_correlation_heatmap(az.correlation_analysis())
    viz.clear_cache()

    # ReportGenerator 通常由 DataAnalyzer.generate_report 封装；确认类可导入
    assert hasattr(ReportGenerator, "generate_report")
```

配套脚本：[`examples/tutorials/12_analyze_helpers.py`](../../examples/tutorials/12_analyze_helpers.py)

## 常见坑

- `analyze_data` / `analyze_multiple_files` 会写 HTML，注意输出目录
- `Visualizer` 在无界面环境请设 `MPLBACKEND=Agg`

## 下一步

- compete 进阶 → [13_compete_advanced.md](13_compete_advanced.md)
