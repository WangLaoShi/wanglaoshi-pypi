# Jupyter Book 源码目录

本目录是 **公开学习手册** 的内容根。

- 配置：`_config.yml` / `_toc.yml`
- 手写教材：`intro.md`、`start/`、`tracks/`、`cases/overview.md`、`reference/contribute.md`
- 构建时同步进来的内容（勿手改）：`cases/*.ipynb`、`reference/api.md` 等

构建：

```bash
python scripts/build_book.py
```

详见仓库根 README「学习手册」一节。
