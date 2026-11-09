# 文档导航

本目录存放开发与交付相关文档。上手用法请先看仓库根目录 [README.md](../README.md)。

| 文档 | 内容 |
| --- | --- |
| [API_INVENTORY.md](API_INVENTORY.md) | 公开 API / 可选依赖盘点（A01） |
| [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md) | Skill → API / 配置 / 测试映射（A02–A03） |
| [DEPENDENCIES.md](DEPENDENCIES.md) | 依赖分层：默认核 / extras / requirements 文件 / torch（F02） |
| [evidence/ADVANCED_EVIDENCE.md](evidence/ADVANCED_EVIDENCE.md) | 12 个进阶 Notebook 跑通证据（E17） |
| [../book/](../book/) | Jupyter Book 公开学习手册（教材站） |
| [../DEVELOPMENT_ROADMAP.md](../DEVELOPMENT_ROADMAP.md) | 增量开发总计划与勾选状态 |
| [../examples/README.md](../examples/README.md) | 示例入口 |

## 常用命令

```bash
# 安装（开发）
pip install -e ".[dev]"

# 快速测试
bash scripts/run_tests.sh quick

# 文档/示例烟测
python scripts/smoke_docs_examples.py

# 12 进阶 Notebook 证据
python scripts/run_advanced_evidence.py

# 构建公开学习手册（Jupyter Book）
python -m pip install -e ".[book]"
python scripts/build_book.py
```
