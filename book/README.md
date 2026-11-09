# 基线工坊 · Jupyter Book 源码

本目录是 **基线工坊 · 从 EDA 到提交** 的站点内容根。  
Python 包名仍为 `wanglaoshi`（`pip install wanglaoshi` / `import wanglaoshi`）。

- 配置：`_config.yml` / `_toc.yml`
- 手写教材：`intro.md`、`start/`、`tracks/`、`practice/`、`cases/overview.md`
- 构建时同步：`cases/*.ipynb`、部分 `reference/*.md`

```bash
python scripts/build_book.py
```

详见仓库根 README「基线工坊」一节。
