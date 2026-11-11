# 10 · 测试与复现治理（Iris）

## 目标

10 分钟内：用 `_testing` 做数据台账、实验上下文、数值断言与分数下限。

## 用到的门面 API

| 符号 | 门面 | 作用 |
| --- | --- | --- |
| `DataRegistry` / `file_sha256` | `_testing` | 数据集来源与校验 |
| `ExperimentContext` | `_testing` | 种子、划分、环境快照 |
| `assert_close` | `_testing` | 浮点近似断言 |
| `assert_above_floor` / `SCORE_FLOORS` | `_testing` | 回归分数门槛 |
| `ready` | `runtime` | 环境就绪 |

## 数据集

sklearn Iris（可离线）+ 临时 CSV 台账。

## 完整示例

```python
from pathlib import Path
import tempfile

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from wanglaoshi.runtime import ready
from wanglaoshi._testing import (
    DataRegistry,
    ExperimentContext,
    assert_close,
    assert_above_floor,
    SCORE_FLOORS,
)

ready(seed=42)
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

ctx = ExperimentContext(random_seed=42, test_size=0.3, notes="tutorial-10")
Xtr, Xte, ytr, yte = ctx.split(X, y, stratify=True)
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(Xtr, ytr)
acc = float(model.score(Xte, yte))
assert_above_floor("iris_accuracy", acc)
assert_close(acc, acc, atol=0, rtol=0)
print("floor", SCORE_FLOORS["iris_accuracy"], "acc", acc)
print(ctx.to_dict()["environment"])

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "iris.csv"
    iris.frame.to_csv(path, index=False)
    reg = DataRegistry(Path(td) / "ledger.json")
    asset = reg.register(
        "iris",
        path,
        source="sklearn",
        license_or_terms="BSD",
        version="1.0",
    )
    assert reg.verify("iris") is True
    print(asset.sha256[:12], "…")
```

配套脚本：[`examples/tutorials/10_testing_repro.py`](../../examples/tutorials/10_testing_repro.py)

## 常见坑

- `_testing` 面向开发/CI，生产业务代码勿强依赖
- 改文件后 `verify` 应变 `False`（防静默换数）
- 分数下限刻意保守；竞赛专用集另走证据脚本策略

## 下一步

- runtime 工具箱 → [11_runtime_toolkit.md](11_runtime_toolkit.md)
- `_testing` 其余 → [17_testing_extra.md](17_testing_extra.md)
- 进阶案例 → `examples/advanced/`
