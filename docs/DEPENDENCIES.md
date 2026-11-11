# 依赖说明（核心 / 可选 / 文件怎么用）

对应 Issue [#11](https://github.com/WangLaoShi/wanglaoshi-pypi/issues/11) / 路线图 F02。

本文回答三件事：

1. **默认装什么**（`pip install wanglaoshi`）
2. **什么时候再加装**（Excel / Jupyter / Optuna / torch）
3. **仓库里的 `requirements*` 文件各自干什么**

设计原则很简单：

- **默认安装 = 课堂能用**：数据分析、sklearn 建模、EDA 报告、业务流水线都能跑
- **可选依赖按场景加**：不要把 torch / Jupyter / 调参库塞进默认，避免安装慢、环境冲突
- **不拆多个发行包**：不用 `wanglaoshi-lite` / `wanglaoshi-full`；用 extras 即可
- **安装真相在 `setup.py`**：`requirements/` 只是方便 CI / 本地 `pip -r`，内容与 setup 对齐

---

## 1. 一分钟怎么选

| 你的目标 | 命令 |
| --- | --- |
| 平时上课、写作业、跑 Analyzer / business / CompetitionTools | `pip install wanglaoshi` |
| 还要读 `.xlsx` | `pip install "wanglaoshi[excel]"` |
| 还要跑仓库里的 Notebook 示例 | `pip install "wanglaoshi[jupyter]"` |
| 还要用 Optuna 贝叶斯调参 | `pip install "wanglaoshi[optuna]"` 或 `[compete]` |
| 业务流水线要读 Excel | `pip install "wanglaoshi[business]"` 或 `[excel]` |
| 上面非深度学习的可选一次装齐 | `pip install "wanglaoshi[all]"` |
| 本地开发 / 跑 pytest | `pip install -e ".[dev]"` |
| 构建基线工坊教材站（Jupyter Book） | `pip install -e ".[book]"` 后执行 `python scripts/build_book.py` |
| 深度学习 `wanglaoshi.ml` | **不要**指望 extras；请按 [PyTorch 官网](https://pytorch.org/) 安装匹配的 `torch` |

组合写法：

```bash
pip install "wanglaoshi[excel,jupyter]"
pip install -e ".[dev,excel]"
```

---

## 2. 默认核心（`install_requires`）

`python_requires>=3.9`（与 CI 矩阵及本仓库验收证据对齐；不再声明未经验证的 3.8）。

| 包 | 用途 |
| --- | --- |
| `pandas` / `numpy` | 表数据与数值计算 |
| `matplotlib` / `seaborn` / `missingno` | 画图与缺失值可视化 |
| `scikit-learn` / `scipy` / `statsmodels` | 建模、统计、科学计算 |
| `jinja2` | Analyzer HTML 报告模板 |
| `rich` / `tqdm` / `requests` | 终端展示、进度条、下载 |

这些会在下列场景用到：

- `DataAnalyzer` / 报告生成
- `CompetitionTools`、`FeatureEngineering`
- `wanglaoshi.business` 业务流水线
- README 里的 Iris 等最小示例

**刻意不放进默认的：**

| 不默认安装 | 原因 |
| --- | --- |
| `torch` | 体积大、CPU/CUDA 源不同；未装时 `import wanglaoshi` / `wanglaoshi.ml` 仍成功，`TORCH_AVAILABLE=False` |
| `jupyter` | 只有跑 Notebook 才需要 |
| `openpyxl` | 只有读 Excel 才需要 |
| `optuna` | 只有贝叶斯调参才需要 |

---

## 3. 可选 extras（`setup.py` → `extras_require`）

| extras 名 | 安装的包 | 什么时候用 |
| --- | --- | --- |
| `excel` | `openpyxl` | `pandas.read_excel` / 业务读 xlsx |
| `jupyter` | `jupyter`, `ipykernel`, `nbformat`, `nbclient` | `examples/`、进阶 Notebook / 证据脚本 |
| `optuna` | `optuna` | `HyperparameterOptimizer.bayesian_optimize` |
| `compete` | `optuna` | 竞赛场景；与 `optuna` 等价，语义别名 |
| `business` | `openpyxl` | 业务流水线读 xlsx；与 `excel` 等价，语义别名 |
| `dev` | `pytest`, `pytest-cov`, `flake8` | 开发与 CI |
| `book` | `jupyter-book`, `jupyter`, `ipykernel`, `nbformat`, `nbclient` | 构建基线工坊 / Pages；`release_check full` 前置 |
| `all` | `excel` + `jupyter` + `optuna`（含 nbformat/nbclient） | 一次性装齐（**不含** torch / dev / book） |

### 未安装时的行为

| 能力 | 未安装时 |
| --- | --- |
| `ml` / torch | 顶层导入不失败；`TORCH_AVAILABLE` 为 False；DL 类调用会提示安装 torch |
| Excel | 读 xlsx 时由 pandas 提示缺少引擎 |
| Jupyter | 不影响 `import wanglaoshi`；只是打不开 `.ipynb` 交互环境 |
| Optuna | 调参 API 会抛出明确的 `ImportError`（测试已覆盖） |

### 关于 torch（单独说明）

```bash
# 示例：仅 CPU（以官网当前命令为准）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

装好后：

```python
from wanglaoshi.ml import TORCH_AVAILABLE, DataPreprocessor
assert isinstance(TORCH_AVAILABLE, bool)
_ = DataPreprocessor()
```

---

## 4. 仓库里的 requirements 文件

目录：

```text
requirements.txt              # 兼容旧习惯 → 转调 core
requirements/
  core.txt                    # = 默认核心
  examples.txt                # core + jupyter
  dev.txt                     # core + pytest/flake8
```

### 怎么用

```bash
# 只装核心（等价于 pip install wanglaoshi 的依赖部分）
pip install -r requirements/core.txt

# 跑示例 Notebook
pip install -r requirements/examples.txt

# 开发测试
pip install -r requirements/dev.txt
# 或（推荐，同时以可编辑模式装本包）
pip install -e ".[dev]"
```

### 和 setup.py 的关系

| 文件 | 关系 |
| --- | --- |
| `requirements/core.txt` | 与 `install_requires` **内容对齐** |
| `requirements/examples.txt` | ≈ `core` + `extras jupyter` |
| `requirements/dev.txt` | ≈ `core` + `extras dev` |
| `requirements.txt` | 仅转发到 `core.txt`，方便旧脚本 / CI |

改依赖时请同时改：

1. `setup.py` 的 `install_requires` / `extras_require`
2. 对应的 `requirements/*.txt`
3. 本文档表格（如有新增分组）

---

## 5. 和 `VERSIONS` 的区别（容易混）

| 概念 | 作用 |
| --- | --- |
| `install_requires` / `requirements/core.txt` | **安装**哪些包 |
| `VERSIONS.get_libraries("simple"|"full")` | **检查**环境里装了什么（清单可以比安装更全，例如列出 tensorflow） |

因此：`VERSIONS` 的 full 清单里出现 pytorch / tensorflow，**不等于**本包装上了它们。

---

## 6. 推荐环境组合

### A. 学生上课（最常见）

```bash
pip install wanglaoshi
```

### B. 跟仓库跑进阶案例

```bash
pip install -e ".[jupyter]"
# 或
pip install -r requirements/examples.txt
pip install -e .
```

### C. 贡献代码 / 跑测试

```bash
pip install -e ".[dev]"
bash scripts/run_tests.sh quick
```

### D. 需要 Excel + Notebook

```bash
pip install -e ".[excel,jupyter,dev]"
```

---

## 7. 验证是否装对

```bash
# 核心导入
python -c "import wanglaoshi; from wanglaoshi import BusinessPipeline, DataAnalyzer, CompetitionTools; print('ok')"

# 可选：torch
python -c "from wanglaoshi.ml import TORCH_AVAILABLE; print('torch', TORCH_AVAILABLE)"

# 测试
bash scripts/run_tests.sh quick
# 发布档（含文档示例烟测）
bash scripts/run_tests.sh release
```

CI 使用：`pip install -e ".[dev]"`，再跑 `quick or submission or release` 标记的测试。
