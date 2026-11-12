# 发布案例真实数据验收

- 生成时间(UTC): `2026-11-12T00:25:31.594655+00:00`
- 口径见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)

| Case | status | data_mode | n_rows | source |
| --- | --- | --- | --- | --- |
| `05_credit_card_fraud` | **blocked** | `missing` | — | expected:data/creditcard.csv (Kaggle creditcardfraud) |
| | note | 需人工放置授权 CSV；禁止无授权自动下载 | | |
| `08_bike_sharing` | **ready** | `real` | 17379 | data/bike_sharing_hour.csv |
| | sha256 | `375d870c6a8adb4c…` | | |
| `09_mall_customers` | **ready** | `real` | 200 | data/Mall_Customers.csv |
| | sha256 | `416a4f62a8f33841…` | | |

复现：
```bash
python scripts/fetch_release_case_data.py
# 欺诈检测：将 Kaggle creditcard.csv 放到 data/creditcard.csv 后再跑
```
