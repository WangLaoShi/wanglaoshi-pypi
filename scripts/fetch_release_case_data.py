#!/usr/bin/env python3
"""为发布案例拉取/校验真实数据（欺诈检测 / 共享单车 / Mall Customers）。

- 共享单车：优先本地 data/bike_sharing_hour.csv；否则尝试 OpenML Bike_Sharing_Demand
- Mall：优先本地 data/Mall_Customers.csv；否则尝试公开镜像下载
- 信用卡欺诈：仅校验本地 data/creditcard.csv（Kaggle 授权数据，不自动下载）

写出 docs/evidence/RELEASE_CASE_DATA.md 与 JSON 指纹。
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_MD = ROOT / "docs" / "evidence" / "RELEASE_CASE_DATA.md"
OUT_JSON = ROOT / "docs" / "evidence" / "release_case_data.json"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _record(slug: str, **kwargs) -> dict:
    row = {"slug": slug, **kwargs}
    return row


def fraud_case() -> dict:
    path = DATA / "creditcard.csv"
    if not path.exists():
        return _record(
            "05_credit_card_fraud",
            data_mode="missing",
            source="expected:data/creditcard.csv (Kaggle creditcardfraud)",
            status="blocked",
            note="需人工放置授权 CSV；禁止无授权自动下载",
        )
    preview = pd.read_csv(path, nrows=5)
    full = pd.read_csv(path, usecols=["Class"])
    return _record(
        "05_credit_card_fraud",
        data_mode="real",
        source=str(path.relative_to(ROOT)),
        n_rows=int(len(full)),
        sha256=_sha256_file(path),
        status="ready",
        columns_sample=list(preview.columns)[:8],
        class_rate=float(full["Class"].mean()) if "Class" in full.columns else None,
    )


def bike_case() -> dict:
    path = DATA / "bike_sharing_hour.csv"
    if path.exists():
        df = pd.read_csv(path)
        return _record(
            "08_bike_sharing",
            data_mode="real",
            source=str(path.relative_to(ROOT)),
            n_rows=int(len(df)),
            sha256=_sha256_file(path),
            status="ready",
        )

    try:
        from sklearn.datasets import fetch_openml

        # OpenML 上该数据集当前活跃版本为 2/3（v1 已不可用）
        raw = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
        df = raw.frame.copy()
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return _record(
            "08_bike_sharing",
            data_mode="real",
            source="OpenML:Bike_Sharing_Demand/v2 → data/bike_sharing_hour.csv",
            n_rows=int(len(df)),
            sha256=_sha256_file(path),
            status="ready",
        )
    except Exception as e:  # noqa: BLE001
        return _record(
            "08_bike_sharing",
            data_mode="missing",
            source="OpenML Bike_Sharing_Demand",
            status="blocked",
            note=f"{type(e).__name__}: {e}",
        )


def mall_case() -> dict:
    path = DATA / "Mall_Customers.csv"
    if path.exists():
        df = pd.read_csv(path)
        return _record(
            "09_mall_customers",
            data_mode="real",
            source=str(path.relative_to(ROOT)),
            n_rows=int(len(df)),
            sha256=_sha256_file(path),
            status="ready",
        )

    # 教学常用公开镜像（非 Kaggle 授权墙）；失败则 blocked
    mirrors = [
        "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
        "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearning-/master/Mall_Customers.csv",
    ]
    last_err = ""
    for url in mirrors:
        try:
            DATA.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as resp:
                path.write_bytes(resp.read())
            df = pd.read_csv(path)
            if len(df) < 50:
                raise ValueError(f"too few rows: {len(df)}")
            return _record(
                "09_mall_customers",
                data_mode="real",
                source=f"{url} → data/Mall_Customers.csv",
                n_rows=int(len(df)),
                sha256=_sha256_file(path),
                status="ready",
            )
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if path.exists():
                path.unlink(missing_ok=True)
    return _record(
        "09_mall_customers",
        data_mode="missing",
        source="expected:data/Mall_Customers.csv",
        status="blocked",
        note=last_err or "download failed",
    )


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [fraud_case(), bike_case(), mall_case()]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 发布案例真实数据验收",
        "",
        f"- 生成时间(UTC): `{payload['generated_at_utc']}`",
        "- 口径见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)",
        "",
        "| Case | status | data_mode | n_rows | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['slug']}` | **{r.get('status')}** | `{r.get('data_mode')}` | "
            f"{r.get('n_rows', '—')} | {r.get('source', '')} |"
        )
        if r.get("sha256"):
            lines.append(f"| | sha256 | `{r['sha256'][:16]}…` | | |")
        if r.get("note"):
            lines.append(f"| | note | {r['note']} | | |")
    lines += [
        "",
        "复现：",
        "```bash",
        "python scripts/fetch_release_case_data.py",
        "# 欺诈检测：将 Kaggle creditcard.csv 放到 data/creditcard.csv 后再跑",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    blocked = [r for r in rows if r.get("status") != "ready"]
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
