# 发布怎么做（只记这一页）

## 你要记住的只有三句

```bash
./scripts/release_check.sh quick   # 日常改完代码
./scripts/release_check.sh         # 发版前（= release）
./scripts/release_check.sh full    # 大版本 / 教材也要重验
```

不用背 `verify_api_reference`、`smoke_docs_examples`、`run_advanced_evidence`、`build_book`……它们都包在上面这条命令里。

## 各档跑什么

| 档 | 测试 | API 说明书 | 文档烟测 | 导入 | 12 案例证据 | 构建 Book |
| --- | --- | --- | --- | --- | --- | --- |
| `quick` | quick | ✅ | ✅ | — | — | — |
| `release`（默认） | 全量 | ✅ | ✅ | ✅ | — | — |
| `full` | 全量 | ✅ | ✅ | ✅ | ✅ | ✅ |

## 脚本通过之后，你手动做

1. 改版本号（`setup.py`、README 版本摘要）
2. 需要时更新 `PROJECT_STATUS.md`
3. `git commit` / `tag` / `push`
4. 发 PyPI（若需要）：`python setup.py sdist bdist_wheel && twine upload dist/*`
5. 教材站：push `master` → GitHub Pages 工作流自动发

## 单个脚本仍在，但不必日常点名

| 脚本 | 何时单独跑 |
| --- | --- |
| `scripts/run_tests.sh` | 只想跑测试 |
| `scripts/verify_api_reference.py` | 只改了 API 文档 |
| `scripts/smoke_docs_examples.py` | 只改了 README 示例 |
| `scripts/run_advanced_evidence.py` | 只改了进阶 Notebook |
| `scripts/build_book.py` | 只改了教材文稿 |

默认入口永远是 **`./scripts/release_check.sh`**。
