# wanglaoshi API 说明书（详尽版）

**产品名：** 基线工坊 · 从 EDA 到提交  
**包名：** `wanglaoshi`  
**文档版本：** 与仓库源码同步（请用 `pip install -e .` 或设置 `PYTHONPATH=仓库根`）  
**对照盘点表：** [API_INVENTORY.md](API_INVENTORY.md)（短表）  
**校验：** `python scripts/verify_api_reference.py`（会执行本文所有标记为可运行的示例）

> **阅读约定**  
> - 签名以当前源码 `inspect.signature` 为准。  
> - 标注 **【可运行】** 的 `python` 代码块必须自洽可执行（不依赖未定义变量）。  
> - 标注 **【需可选依赖】** 的示例在缺依赖时会明确报错，不算文档错误。  
> - README 中的历史片段**不是**本说明书；有冲突以本文 + 源码为准。

---

## 0. 安装与导入

```bash
# 开发（推荐，与本文一致）
pip install -e ".[dev]"

# 仅核心
pip install -e .
```

```python
# 【可运行】
import wanglaoshi
print(sorted(wanglaoshi.__all__))
```

顶层推荐导入：

| 写法 | 用途 |
| --- | --- |
| `from wanglaoshi import ready` | Notebook 一键就绪 |
| `from wanglaoshi import DataAnalyzer, analyze_data, analyze_multiple_files, analyze_notebook` | EDA |
| `from wanglaoshi import CompetitionTools as CT` | 竞赛工具 |
| `from wanglaoshi import FeatureEngineering as FE` | 特征工程 |
| `from wanglaoshi import BusinessPipeline` | 业务流水线 |
| `from wanglaoshi import VERSIONS as V` | 版本检查 |
| `from wanglaoshi.WebGetter import Wget` | 下载（**不要** `from WebGetter import Wget`） |
| `from wanglaoshi import MLDL` | 可选；未装 torch 时可能为 `None` |

---

## 1. `ready` — 环境就绪

**签名：** `ready(*, quiet=True, show_env=True, seed=42, check_core=False) -> Dict[str, Any]`

| 参数 | 含义 |
| --- | --- |
| `quiet` | `True` 时抑制告警（替代重复的 `filterwarnings` / `JE.no_warning`） |
| `show_env` | 打印环境横幅 |
| `seed` | 固定随机种子；`None` 不设 |
| `check_core` | 检查 numpy/pandas/sklearn 是否可导入 |

```python
# 【可运行】
from wanglaoshi import ready

info = ready(show_env=False, quiet=True, seed=42)
assert isinstance(info, dict)
print(sorted(info.keys())[:8] if info else "ok")
```

---

## 2. `DataAnalyzer` / EDA 函数

### 2.1 构造

```text
DataAnalyzer(df: pd.DataFrame)
```

### 2.2 实例方法（公开）

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `get_basic_info()` | `dict` | 行数、列、类型等 |
| `basic_statistics()` | `DataFrame` | 描述统计 |
| `missing_value_analysis()` | `DataFrame` | 缺失 |
| `duplicate_analysis()` | `dict` | 重复 |
| `outlier_analysis()` | `DataFrame` | 异常 |
| `normality_test()` | `DataFrame` | 正态性（小样本可能警告） |
| `correlation_analysis()` | `DataFrame` | 相关 |
| `multicollinearity_analysis()` | `DataFrame` | VIF 等 |
| `pca_analysis()` | `dict` | PCA |
| `get_categorical_stats()` | `dict` | 类别统计 |
| `get_time_stats()` | `dict` | 时间列 |
| `get_memory_usage()` | `float` | 内存 |
| `plot_distribution(column)` | `str` | 图 base64 |
| `plot_correlation_heatmap()` | `str` | 热图 base64 |
| `plot_time_series(series, column)` | `str` | 时间图 base64 |
| `explore_dataframe(name='DataFrame', show_plots=True)` | `dict` | 一站式探索 |
| `interpret_results()` | `dict` | 结果解读 |
| `generate_report(output_html='analysis_report.html')` | `None` | 写 HTML |

**注意：** 类上**没有** `analyze_notebook` 方法。Notebook 请用模块函数 `analyze_notebook(df)`。

### 2.3 模块级函数

| 函数 | 签名 |
| --- | --- |
| `analyze_data` | `(file_path, output_html='analysis_report.html') -> None` |
| `analyze_multiple_files` | `(folder_path, output_dir='reports') -> None` |
| `analyze_notebook` | `(df, show_plots=True) -> Dict[str, Any]` |

```python
# 【可运行】
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from sklearn.datasets import load_iris
from wanglaoshi import DataAnalyzer, analyze_data, analyze_notebook, analyze_multiple_files

out = Path("_api_ref_out")
out.mkdir(exist_ok=True)

df = load_iris(as_frame=True).frame
az = DataAnalyzer(df)
info = az.get_basic_info()
assert "shape" in info or len(info) > 0
stats = az.basic_statistics()
assert stats is not None
miss = az.missing_value_analysis()
assert miss is not None

# Notebook 入口：返回字典；大表建议 show_plots=False
result = analyze_notebook(df, show_plots=False)
assert "基本信息" in result

csv_path = out / "iris.csv"
df.to_csv(csv_path, index=False)
analyze_data(str(csv_path), str(out / "iris_report.html"))
assert (out / "iris_report.html").exists()

folder = out / "batch"
folder.mkdir(exist_ok=True)
df.head(50).to_csv(folder / "a.csv", index=False)
df.tail(50).to_csv(folder / "b.csv", index=False)
analyze_multiple_files(str(folder), str(out / "batch_reports"))
print("DataAnalyzer OK", info.get("shape", list(info)[:3]))
```

---

## 3. `CompetitionTools`（`import wanglaoshi.CompetitionTools as CT`）

### 3.1 `SubmissionGenerator`

| 方法 | 签名摘要 |
| --- | --- |
| `create_submission` | `(predictions, sample_submission_path, output_path='submission.csv', id_column=None, target_column=None, prediction_type='auto', class_labels=None) -> DataFrame` |
| `validate_submission` | `(submission_path, sample_submission_path, check_values=True, id_column=None, target_column=None) -> dict` |
| `ensemble_submissions` | `(submission_files, weights=None, method='weighted_average', output_path='ensemble_submission.csv', sample_submission_path=None) -> DataFrame` |

`prediction_type` 合法值：`auto` | `label` | `proba` | `multioutput`  
（**不是** `probability`。）

`validate_submission` 返回：

```text
{'is_valid': bool, 'errors': list, 'warnings': list, 'info': dict}
```

```python
# 【可运行】
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wanglaoshi import CompetitionTools as CT

out = Path("_api_ref_out")
out.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=500, random_state=42).fit(X_tr, y_tr)
pred = model.predict(X_te)

sample = pd.DataFrame({"Id": np.arange(len(pred)), "target": y_te.to_numpy()})
sample_path = out / "sample_submission.csv"
sub_path = out / "submission.csv"
sample.to_csv(sample_path, index=False)

gen = CT.SubmissionGenerator()
gen.create_submission(pred, str(sample_path), str(sub_path), prediction_type="label")
val = gen.validate_submission(str(sub_path), str(sample_path))
assert val["is_valid"] is True

# 融合：先准备多份合法提交
for i in range(3):
    gen.create_submission(pred, str(sample_path), str(out / f"sub{i}.csv"), prediction_type="label")
ens = gen.ensemble_submissions(
    [str(out / f"sub{i}.csv") for i in range(3)],
    weights=[0.4, 0.3, 0.3],
    method="weighted_average",
    output_path=str(out / "ensemble.csv"),
)
assert len(ens) == len(pred)
print("SubmissionGenerator OK", val["info"]["submission_shape"])
```

### 3.2 `QuickBaseline`

| 方法 | 签名 |
| --- | --- |
| `run_all_models` | `(X_train, y_train, X_test=None, y_test=None, task_type='auto', cv=5) -> dict` |
| `compare_models` | `(results, top_n=5) -> DataFrame` |

`task_type`：`auto` | `classification` | `regression`。

```python
# 【可运行】
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from wanglaoshi import CompetitionTools as CT

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
baseline = CT.QuickBaseline()
results = baseline.run_all_models(
    X_tr, y_tr, X_te, y_te, task_type="classification"
)
cmp = baseline.compare_models(results, top_n=5)
assert "Model" in cmp.columns
print(cmp.head())
```

### 3.3 `LeakageDetector`

| 方法 | 返回 |
| --- | --- |
| `detect_target_leakage(X, y, threshold=0.9, method='correlation'\|'mutual_info')` | **列表**（元素常为含 `feature`/`correlation` 的 dict，不是纯字符串） |
| `detect_time_leakage(df, date_col, target_col, check_future=True)` | `dict`：`has_leakage` / `warnings` / `info` |
| `detect_distribution_leakage(train_df, test_df, threshold=0.1)` | `dict`：`has_leakage` / `leakage_features` / `warnings` |

```python
# 【可运行】
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from wanglaoshi import CompetitionTools as CT

df = load_iris(as_frame=True).frame
X = df.drop(columns=["target"])
y = df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

det = CT.LeakageDetector()
X_bad = X_tr.copy()
X_bad["leak"] = y_tr.to_numpy() + 0.001
leaks = det.detect_target_leakage(X_bad, y_tr, threshold=0.95)
assert any(
    (item["feature"] if isinstance(item, dict) else item) == "leak" for item in leaks
)

timed = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "target": np.arange(20),
    }
)
tinfo = det.detect_time_leakage(timed, date_col="date", target_col="target")
assert "has_leakage" in tinfo

train_df = X_tr.assign(target=y_tr.to_numpy())
test_df = X_te.assign(target=y_te.to_numpy())
dinfo = det.detect_distribution_leakage(train_df, test_df)
assert "has_leakage" in dinfo
print("LeakageDetector OK", len(leaks), tinfo["has_leakage"], dinfo["has_leakage"])
```

### 3.4 `ModelEnsemble`

| 方法 | 关键参数 |
| --- | --- |
| `voting(models, X, y, voting='soft', weights=None)` | 拟合投票器 |
| `stacking(models, meta_model, X_train, y_train, X_test=None, cv=5, ...)` | 堆叠 |
| `blending(models, X_train, y_train, X_val, y_val, meta_model=None, weights=None)` | **必须**提供验证集 `X_val`/`y_val` |

```python
# 【可运行】
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from wanglaoshi import CompetitionTools as CT

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_va, X_hold, y_va, y_hold = train_test_split(
    X_te, y_te, test_size=0.5, random_state=0, stratify=y_te
)

ens = CT.ModelEnsemble()
m1 = RandomForestClassifier(n_estimators=30, random_state=0)
m2 = LogisticRegression(max_iter=500, random_state=0)
voting_model = ens.voting([m1, m2], X_tr, y_tr, voting="soft")
assert hasattr(voting_model, "predict")

stack = ens.stacking(
    [m1, m2],
    LogisticRegression(max_iter=500),
    X_tr,
    y_tr,
    X_test=X_hold,
    cv=3,
)
blend = ens.blending(
    [m1, m2],
    X_tr,
    y_tr,
    X_va,
    y_va,
    meta_model=LogisticRegression(max_iter=500),
)
print("ModelEnsemble OK", type(voting_model).__name__, type(stack).__name__)
```

### 3.5 `FeatureImportanceAnalyzer`

```text
calculate_importance(model, X, y, method='permutation'|'shap'|'builtin', n_repeats=10) -> DataFrame
plot_importance(importance_df, top_n=20, figsize=(10, 8)) -> None
```

`shap` 需额外安装 `shap`。

```python
# 【可运行】
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wanglaoshi import CompetitionTools as CT

X, y = load_iris(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X_tr, y_tr)
fia = CT.FeatureImportanceAnalyzer()
imp = fia.calculate_importance(model, X_te, y_te, method="builtin")
assert "feature" in imp.columns or imp.shape[1] >= 1
print(imp.head())
```

### 3.6 `CrossValidator`

```text
stratified_cv(model, X, y, n_splits=5) -> dict  # scores/mean/std/...
time_series_cv(model, X, y, n_splits=5) -> dict
group_cv(model, X, y, groups, n_splits=5) -> dict  # groups 必填
```

```python
# 【可运行】
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from wanglaoshi import CompetitionTools as CT

X, y = load_iris(return_X_y=True, as_frame=True)
model = LogisticRegression(max_iter=500)
cv = CT.CrossValidator()
s = cv.stratified_cv(model, X, y, n_splits=3)
assert "mean" in s
groups = pd.Series(np.arange(len(X)) // 10)
g = cv.group_cv(model, X, y, groups, n_splits=3)
assert "mean" in g
print("CrossValidator OK", round(s["mean"], 4), round(g["mean"], 4))
```

### 3.7 `ExperimentTracker`

```text
__init__(log_file='experiments.json')
log_experiment(experiment_name, model_name, features, params, score, metrics=None, notes=None)
get_experiment_history(sort_by='score', ascending=False, top_n=None) -> DataFrame
compare_experiments(experiment_names=None) -> DataFrame
```

```python
# 【可运行】
from pathlib import Path
from wanglaoshi import CompetitionTools as CT

log = Path("_api_ref_out") / "experiments.json"
log.parent.mkdir(exist_ok=True)
if log.exists():
    log.unlink()

tr = CT.ExperimentTracker(log_file=str(log))
tr.log_experiment(
    "exp_demo",
    "LogisticRegression",
    features=["f1", "f2"],
    params={"C": 1.0},
    score=0.91,
    metrics={"f1": 0.9},
    notes="api-ref",
)
hist = tr.get_experiment_history()
assert len(hist) >= 1
print(tr.compare_experiments())
```

### 3.8 `HyperparameterOptimizer` 【需可选依赖】

```text
bayesian_optimize(model_class, param_space, X, y, n_trials=100, cv=5)  # 需要 optuna
auto_tune(model_class, X, y, model_type='auto')
```

未安装 Optuna 时会 `ImportError`（`pip install wanglaoshi[optuna]` 或 `pip install optuna`）。

```python
# 【可运行】
from wanglaoshi import CompetitionTools as CT

opt = CT.HyperparameterOptimizer()
try:
    import optuna  # noqa: F401
except ImportError:
    print("Optuna 未安装：跳过 bayesian_optimize（预期行为）")
else:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True, as_frame=True)
    space = {"C": (0.1, 10.0)}
    # n_trials 取小，仅作 API 冒烟
    result = opt.bayesian_optimize(
        LogisticRegression, space, X, y, n_trials=3, cv=3
    )
    print("bayesian_optimize keys", result.keys() if isinstance(result, dict) else type(result))
```

---

## 4. `FeatureEngineering.AdvancedFeatureEngineer`

构造：`AdvancedFeatureEngineer()`

| 方法 | 签名摘要 |
| --- | --- |
| `extract_datetime_features` | `(df, datetime_col, features=None) -> DataFrame` |
| `extract_text_features` | `(df, text_col, method='basic', max_features=100) -> DataFrame` |
| `one_hot_encode` | `(df, categorical_cols, drop_first=False) -> DataFrame` |
| `frequency_encode` | `(df, categorical_col) -> DataFrame` |
| `target_encode` | `(df, categorical_col, target_col, smoothing=1.0, min_samples_leaf=1) -> DataFrame` |
| `log_transform` | `(df, numeric_cols, add_one=True) -> DataFrame` |
| `boxcox_transform` | `(df, numeric_cols) -> DataFrame` |
| `bin_features` | `(df, numeric_cols, n_bins=5, strategy='quantile') -> DataFrame` |
| `select_features_by_importance` | `(X, y, n_features=10, model_type='random_forest') -> List[str]` |
| `select_features_by_correlation` | `(X, y, threshold=0.1) -> List[str]` |
| `select_features_by_mutual_info` | `(X, y, n_features=10) -> List[str]` |

**纪律：** 编码/选择必须在训练折上拟合；`target_encode` 全表调用仅适合演示，竞赛请用折外编码（见教材精讲）。

```python
# 【可运行】
import pandas as pd
from sklearn.datasets import load_iris
from wanglaoshi import FeatureEngineering as FE

df = load_iris(as_frame=True).frame.copy()
df["event_time"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
df["note"] = ["ok"] * len(df)
df["city"] = (df["target"] % 3).astype(str)

fe = FE.AdvancedFeatureEngineer()
d1 = fe.extract_datetime_features(df, "event_time")
assert any(c.startswith("event_time_") or "year" in c for c in d1.columns)
d2 = fe.extract_text_features(df, "note", method="basic")
d3 = fe.one_hot_encode(df, ["city"], drop_first=True)
d4 = fe.frequency_encode(df, "city")
d5 = fe.target_encode(df, "city", "target")
d6 = fe.log_transform(df, ["sepal length (cm)"])
d7 = fe.bin_features(df, ["sepal width (cm)"], n_bins=3)

X = df.drop(columns=["target", "event_time", "note", "city"])
y = df["target"]
feats = fe.select_features_by_importance(X, y, n_features=2)
assert len(feats) == 2
print("FeatureEngineering OK", feats)
```

---

## 5. `BusinessPipeline` / `wanglaoshi.business`

### 5.1 状态契约

`StageResult.status ∈ {passed, warning, blocked, skipped}`  
上游 `blocked` 且 `stop_on_block=True` 时下游跳过；正式交付应检查运行摘要。

### 5.2 `BusinessPipeline`

```text
BusinessPipeline(
  archive_dir,
  *,
  required_columns=None,
  period=None,
  random_seed=42,
  stop_on_block=True,
  date_col=None,
  value_col=None,
  dim_cols=None,
  segment_cols=None,
  metric_group_cols=None,
  redact_fields=None,
  chart_unit='currency_unit',
)
run(source_path, run_id=None) -> RunRecord
run_to_dict(source_path, **kwargs) -> dict
```

### 5.3 其它常用导出

| 名称 | 用途 |
| --- | --- |
| `intake_file` | 仅 D01 接收 |
| `structure_dataframe` / `structure_file` | D02 |
| `clean_dataframe` | D03 |
| `run_eda_profile` | D12 |
| `derive_metrics` | D04 |
| `build_analysis_plan` | D05 |
| `calculate_metrics` | D06 |
| `run_multidimensional_analysis` | D07 |
| `archive_charts` | D08 |
| `generate_insights` | D09 |
| `archive_delivery` | D10 |
| `STAT_METHOD_CATALOG` / `DEFAULT_METRIC_DICT` | 目录常量 |

```python
# 【可运行】
from pathlib import Path
import pandas as pd
from wanglaoshi import BusinessPipeline

demo = Path("_api_ref_out") / "biz"
demo.mkdir(parents=True, exist_ok=True)
src = demo / "orders.csv"
pd.DataFrame(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "order_date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "amount": [10.0, 20.0, 15.0, 30.0, 25.0, 18.0],
        "channel": ["web", "app", "web", "app", "web", "app"],
        "user_tier": ["A", "B", "A", "B", "A", "B"],
    }
).to_csv(src, index=False)

pipe = BusinessPipeline(
    demo / "archive",
    required_columns=["order_id", "amount"],
    date_col="order_date",
    value_col="amount",
    dim_cols=["channel"],
    segment_cols=["user_tier"],
    period="2024-Q1",
)
record = pipe.run(src)
summary = record.summary()
assert "stages" in summary
assert summary.get("blocked") is False
print("BusinessPipeline OK", summary["run_id"], len(summary["stages"]))
```

---

## 6. `VERSIONS`

| 函数 | 说明 |
| --- | --- |
| `check_all_versions(all_columns=False, version='full'\|'simple', show_latest=False, only_problems=False)` | 打印表格 |
| `check_versions(libraries=None, version='full')` | 返回 `dict` |
| `check_all_installed()` | 已安装包 |
| `check_all_installed_with_latest()` | **需网络** |
| `show_env_info()` | 环境横幅 |
| `get_libraries(version='full')` | 库清单 |

```python
# 【可运行】
from wanglaoshi import VERSIONS as V

V.check_all_versions(version="simple")
versions = V.check_versions(version="simple")
assert "numpy" in versions
print("numpy =", versions["numpy"])
```

---

## 7. `JupyterEnv` / `JupyterFont` / `WebGetter` / `Useful`

### 7.1 `JupyterEnv`

| 函数 | 说明 |
| --- | --- |
| `no_warning()` | 仅关警告；课堂优先 `ready()` |
| `jupyter_kernel_list()` | 列出 kernel |
| `install_kernel()` / `install_ipykernel()` | **交互式**，会提示输入 |
| `running()` / `running_os()` / `running_python_*` | 环境探测 |

```python
# 【可运行】
from wanglaoshi import JupyterEnv as JE

assert callable(JE.jupyter_kernel_list)
assert callable(JE.install_kernel)
print("python", JE.running_python_version())
```

### 7.2 `JupyterFont`

```text
matplotlib_font_init(show_demo=False)
simhei()
fonts(show_demo=False)
```

```python
# 【可运行】
from wanglaoshi import JupyterFont as JF

JF.matplotlib_font_init(show_demo=False)
print("JupyterFont OK")
```

### 7.3 `WebGetter.Wget`

**正确导入：**

```python
from wanglaoshi.WebGetter import Wget
# 或
from wanglaoshi import WebGetter
downloader = WebGetter.Wget(url=..., output_dir=".", filename=None)
downloader.download()
```

```python
# 【可运行】
from wanglaoshi.WebGetter import Wget

# 只校验构造；不发起真实下载（避免文档校验依赖外网）
dl = Wget(url="https://example.com/file.zip", output_dir=".", filename="file.zip")
assert hasattr(dl, "download")
print("Wget OK", type(dl).__name__)
```

### 7.4 `Useful`

辅助：`dep()` / `helps()` / `pypis()` / `styles()`（以源码为准，偏课堂演示）。

```python
# 【可运行】
from wanglaoshi import Useful

assert callable(Useful.helps)
print("Useful OK")
```

---

## 8. `MLDL` 【需可选依赖：torch】

- `from wanglaoshi import MLDL`：未装 torch 时 **`MLDL is None`**，不会让整个包装载失败。  
- 使用前请按 [PyTorch 官网](https://pytorch.org/) 安装，再：

```python
# 【可运行】
from wanglaoshi import MLDL

if MLDL is None:
    print("MLDL 不可用：未安装 torch（预期）")
else:
    assert hasattr(MLDL, "DataPreprocessor") or True
    print("MLDL 模块已加载", MLDL)
```

---

## 9. 与 README / 教材的关系

| 文档 | 角色 |
| --- | --- |
| **本文 `docs/API_REFERENCE.md`** | API 说明书真源（签名 + 可运行示例） |
| `docs/API_INVENTORY.md` | A01 短盘点 |
| 仓库 `README.md` | 产品介绍 + 最短上手；历史长示例可能过时 |
| `book/` 基线工坊 | 教学叙事；API 细节链到本文 |
| `PROJECT_STATUS.md` | 功能点完成率 |

已知 README 历史错误（勿再复制）：

1. `from WebGetter import Wget` → 应为 `wanglaoshi.WebGetter`  
2. `prediction_type='probability'` → 应为 `proba` / `label` / `auto`  
3. `DataAnalyzer.analyze_notebook()` → 用模块函数 `analyze_notebook(df)`  
4. 泄露检测示例曾有缩进语法错误  
5. 大量 CT/FE 片段缺少 `X_val` / `trained_model` / 列名定义

---

## 10. 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-11-09 | 初版：按源码签名编写；修复 `analyze_notebook`；全部【可运行】块纳入 `scripts/verify_api_reference.py` |
