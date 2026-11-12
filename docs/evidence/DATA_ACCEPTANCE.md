# 案例数据验收口径

## 两种证据，不要混用

| 证据 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| `ADVANCED_EVIDENCE.md`（inprocess） | Notebook 能跑完、主路径无未捕获异常 | 是否使用真实外部数据；指标是否手算一致 |
| 业务手算 / 契约测试 | 口径与阻断行为正确 | 22 课全部真实数据 |

## 发布验证（真实数据）最低字段

每个声称「真实数据通过」的案例应记录：

1. **data_mode**：`real` | `demo` | `synthetic`
2. **source**：URL / 本地路径 / 包内资源
3. **sha256**（或等价指纹）与 **n_rows**
4. **业务断言**：至少一条与案例目标相关的数值/形状检查
5. **命令与日期**

若运行中发生 DEMO/合成回退：`data_mode` 必须记为 `demo`/`synthetic`，且**不得**计入「真实数据验收通过」。

## 发布优先案例（`REQUIRE_REAL_DATA`）

下列案例在 `python scripts/run_advanced_evidence.py --release-data` 下须同时满足：

1. 执行 `status=passed`
2. `data_mode` 以 `real` 开头（不得 `demo` / `unspecified`）
3. 具备 **`data_source`、`n_rows`、`data_sha256`（完整 64 位十六进制）**

缺任一字段 → `release_data_verdict=data_fail`（退出码 2）。仅声称 `real` 而无证据不算通过。

| 案例 | 真实数据入口 | 允许 DEMO（教学） |
| --- | --- | --- |
| `05_credit_card_fraud` | `data/creditcard.csv`（Kaggle，需人工放置） | 是 |
| `08_bike_sharing` | `data/bike_sharing_hour.csv` 或 OpenML v2 | 是 |
| `09_mall_customers` | 本地/约定数据源；当前常见合成回退 | 是 |

其余案例默认允许合成 / DEMO。指纹台账见 [`RELEASE_CASE_DATA.md`](RELEASE_CASE_DATA.md)。

## 日常演示

允许合成与 DEMO；教学烟测：`python scripts/run_advanced_evidence.py`（退出码只看执行失败）。

## 脚本如何记录

`run_advanced_evidence.py` 在每个案例结果里写入：

- `data_mode`：若 Notebook 命名空间有 `DEMO=True` → `demo`；`DEMO=False` → `real_or_local`（或 Notebook 自设的 `data_mode`）；否则 `unspecified`
- 透传：`data_source` / `n_rows` / `data_sha256`（发布必验案例必须赋值完整哈希）

`release_check.sh full` 会以 `--release-data` 跑证据。汇总字段 `demo_or_synthetic` / `real_or_local_claimed` 仅作分类，**不得**把 `demo` 或「缺证据的 real」记进发布通过。
