# 如何贡献 / 如何改手册

## 改学习内容

| 改什么 | 改哪里 |
| --- | --- |
| 学习路径、安装说明、主线讲解 | `book/**/*.md` |
| 12 个进阶案例 | `examples/advanced/*.ipynb`（再跑构建同步进书） |
| API / Skill 映射原文 | `docs/API_INVENTORY.md`、`docs/SKILL_API_MAPPING.md` |
| 依赖分层 | `docs/DEPENDENCIES.md`、`setup.py` extras |

## 构建手册

```bash
python -m pip install -e ".[book]"
python scripts/build_book.py
```

`scripts/prepare_book.py` 会把 Notebook 与参考文档同步到 `book/cases/`、`book/reference/`（生成物可本地忽略，CI 每次重建）。

## 发布 GitHub Pages

推送到 `master` 后，工作流 `.github/workflows/jupyter-book.yml` 会构建并发布到 `gh-pages`。  
仓库 Settings → Pages → Source 选 `gh-pages` 分支。
