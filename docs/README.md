# 文档导航

本目录存放开发与交付相关文档。上手用法请先看仓库根目录 [README.md](../README.md)。

| 文档 | 内容 |
| --- | --- |
| **[RELEASE.md](RELEASE.md)** | **发布只记三句命令**（`release_check.sh`） |
| **[API_REFERENCE.md](API_REFERENCE.md)** | **详尽 API 说明书**（签名 + 已校验【可运行】示例） |
| [API_INVENTORY.md](API_INVENTORY.md) | 公开 API / 可选依赖短盘点（A01） |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 包分层门面与零破坏迁移（Phase 1–4） |
| [SKILL_API_MAPPING.md](SKILL_API_MAPPING.md) | Skill → API / 配置 / 测试映射（A02–A03） |
| [DEPENDENCIES.md](DEPENDENCIES.md) | 依赖分层：默认核 / extras / requirements 文件 / torch（F02） |
| [evidence/ADVANCED_EVIDENCE.md](evidence/ADVANCED_EVIDENCE.md) | 进阶 Notebook 跑通证据 |
| [ANALYSIS_COVERAGE.md](ANALYSIS_COVERAGE.md) | 常见分析方法覆盖矩阵与 wanglaoshi 优化缺口 |
| [../PROJECT_STATUS.md](../PROJECT_STATUS.md) | 功能点数量与完成率仪表盘 |
| [../book/](../book/) | 基线工坊 · 从 EDA 到提交（Jupyter Book 教材站） |
| [../DEVELOPMENT_ROADMAP.md](../DEVELOPMENT_ROADMAP.md) | 增量开发总计划与勾选状态 |
| [../examples/README.md](../examples/README.md) | 示例入口 |

## 常用命令（只记总入口）

详见 **[RELEASE.md](RELEASE.md)**。

```bash
pip install -e ".[dev]"

./scripts/release_check.sh quick   # 日常
./scripts/release_check.sh         # 发版前
./scripts/release_check.sh full    # 大版本（含案例证据 + 教材构建）
```

单个脚本（`run_tests` / `verify_api_reference` / `build_book` 等）仍可单独跑，日常不必背。
