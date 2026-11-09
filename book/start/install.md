# 安装环境

推荐 Python `>= 3.8`。完整依赖分层见 {doc}`../reference/dependencies`。

## 学生上课（最常见）

```bash
pip install wanglaoshi
```

## 跟本手册跑 Notebook

```bash
git clone https://github.com/WangLaoShi/wanglaoshi-pypi.git
cd wanglaoshi-pypi
python -m pip install -e ".[jupyter]"
```

或：

```bash
python -m pip install -r requirements/examples.txt
python -m pip install -e .
```

## 可选能力

| 需求 | 命令 |
| --- | --- |
| 读 Excel | `pip install "wanglaoshi[excel]"` |
| Optuna 调参 | `pip install "wanglaoshi[optuna]"` |
| 开发测试 | `pip install -e ".[dev]"` |
| 非 DL 一次齐 | `pip install "wanglaoshi[all]"` |
| 深度学习 `MLDL` | 按 [PyTorch 官网](https://pytorch.org/) 装 `torch`（不进 extras） |

## 验证

```bash
python -c "import wanglaoshi; from wanglaoshi import DataAnalyzer, BusinessPipeline, CompetitionTools; print('ok')"
python scripts/smoke_docs_examples.py
```

## 构建基线工坊站点（维护者）

```bash
python -m pip install -e ".[book]"
python scripts/build_book.py
# 产物：book/_build/html/index.html
```
