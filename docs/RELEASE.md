# 发布怎么做（只记这一页）

## 你要记住的验收命令

```bash
./scripts/release_check.sh quick          # 日常改完代码
./scripts/release_check.sh                # 发版前验收（= release）
./scripts/release_check.sh full           # 大版本 / 教材也要重验
./scripts/verify_clean_env.sh             # 干净 venv 全量 pytest（F01 证据）
```

`release_check` **只做检查，不会**清理 dist、上传 PyPI、打 tag 或 push。失败即停。

## 各档跑什么

| 档 | 测试 | API 说明书 | 文档烟测 | 导入 | 包内字体/模板 | Optuna 方向 | 案例证据 | 构建 Book |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `quick` | quick | ✅ | ✅ | — | — | — | — | — |
| `release`（默认） | 全量 | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| `full` | 全量 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅（先装 `[book]`） | ✅ |

`full` 会在跑 Notebook 证据**之前**安装 `.[book]`（含 `nbformat`/`nbclient`）。案例 `passed` ≠ 真实数据验收，见 `docs/evidence/DATA_ACCEPTANCE.md`。

发版产物验收：构建 wheel 后在干净环境安装，再跑 `python scripts/verify_wheel_assets.py`（确认 `SimHei.ttf` 等资源进包）。支持的最低 Python：**3.9**。

## 发版请分步（不要一条龙）

每步成功后再做下一步；任一步失败立即停止。

1. **验收**：`./scripts/release_check.sh`（建议再跑 `./scripts/verify_clean_env.sh`）
2. **改版本号**：`setup.py`、README 版本摘要；需要时更新 `PROJECT_STATUS.md`
3. **提交代码**：`git add … && git commit && git push`（仅代码，不含 tag）
4. **构建产物**：`rm -rf build dist *.egg-info && python setup.py sdist bdist_wheel`
5. **检查包**：`twine check dist/*`
6. **上传 PyPI**（若需要）：`twine upload dist/*`
7. **打 tag 并推送**：`git tag vX.Y.Z && git push origin vX.Y.Z`
8. **教材站**：`master` push 后由 GitHub Pages 工作流发布

> `python setup.py upload`（旧 UploadCommand）已停用，防止清理/上传/commit/push 混在一起。

## 单个脚本仍在，但不必日常点名

| 脚本 | 何时单独跑 |
| --- | --- |
| `scripts/run_tests.sh` | 只想跑测试 |
| `scripts/verify_clean_env.sh` | 独立干净环境基线 |
| `scripts/verify_api_reference.py` | 只改了 API 文档 |
| `scripts/smoke_docs_examples.py` | 只改了 README / 教程示例 |
| `scripts/run_advanced_evidence.py` | 只改了进阶 Notebook |
| `scripts/build_book.py` | 只改了教材文稿 |
| `scripts/verify_wheel_assets.py` | wheel / 可编辑安装后核对中文字体与报告模板 |
| `scripts/verify_chinese_plot.py` | 用包内 SimHei 渲染中文标题图（无缺字警告） |
| `scripts/verify_chinese_report.py` | 中文列名完整报告（含 PCA）并捕获缺字/DejaVu 回退 |
| `scripts/fetch_release_case_data.py` | 欺诈/共享单车真实数据指纹（见 RELEASE_CASE_DATA.md） |
| `scripts/run_advanced_evidence.py --release-data` | 必填真实数据案例不得 DEMO（退出码 2） |

默认验收入口：**`./scripts/release_check.sh`**。干净环境证据入口：**`./scripts/verify_clean_env.sh`**。
