#!/usr/bin/env python3
"""为发布案例拉取/校验真实数据（欺诈检测 / 共享单车 / Mall Customers）。

- 共享单车：优先本地 data/bike_sharing_hour.csv；否则尝试 OpenML Bike_Sharing_Demand
- Mall：优先本地；否则仅当哈希匹配钉扎镜像时才写入
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

# 下载硬限制，防止镜像返回超大文件撑爆磁盘
_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

# Mall：钉扎内容哈希（与仓库已验收的 data/Mall_Customers.csv 一致）
_MALL_PINNED = {
    "sha256": "416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77",
    "n_rows": 200,
    "urls": [
        "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
    ],
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_limited(url: str, max_bytes: int = _MAX_DOWNLOAD_BYTES) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "wanglaoshi-release-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        chunks: list[bytes] = []
        total = 0
        while True:
            block = resp.read(64 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError(f"download exceeds {max_bytes} bytes from {url}")
            chunks.append(block)
    return b"".join(chunks)


def _record(slug: str, **kwargs) -> dict:
    return {"slug": slug, **kwargs}


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
    pinned = _MALL_PINNED["sha256"]
    if path.exists():
        digest = _sha256_file(path)
        df = pd.read_csv(path)
        if digest != pinned:
            return _record(
                "09_mall_customers",
                data_mode="real",
                source=str(path.relative_to(ROOT)),
                n_rows=int(len(df)),
                sha256=digest,
                status="blocked",
                note=f"local file sha256 mismatch pinned={pinned[:16]}…",
            )
        return _record(
            "09_mall_customers",
            data_mode="real",
            source=str(path.relative_to(ROOT)),
            n_rows=int(len(df)),
            sha256=digest,
            status="ready",
        )

    last_err = ""
    for url in _MALL_PINNED["urls"]:
        try:
            DATA.mkdir(parents=True, exist_ok=True)
            raw = _download_limited(url)
            digest = _sha256_bytes(raw)
            if digest != pinned:
                raise ValueError(
                    f"mirror sha mismatch got={digest} expected={pinned} url={url}"
                )
            path.write_bytes(raw)
            df = pd.read_csv(path)
            if int(len(df)) != int(_MALL_PINNED["n_rows"]):
                raise ValueError(f"unexpected n_rows={len(df)}")
            return _record(
                "09_mall_customers",
                data_mode="real",
                source=f"{url} → data/Mall_Customers.csv (pinned)",
                n_rows=int(len(df)),
                sha256=digest,
                status="ready",
            )
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if path.exists():
                path.unlink(missing_ok=True)
    return _record(
        "09_mall_customers",
        data_mode="missing",
        source="expected:data/Mall_Customers.csv (pinned sha)",
        status="blocked",
        note=last_err or "download failed",
    )


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [fraud_case(), bike_case(), mall_case()]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
        "mall_pinned_sha256": _MALL_PINNED["sha256"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 发布案例真实数据验收",
        "",
        f"- 生成时间(UTC): `{payload['generated_at_utc']}`",
        "- 口径见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)",
        f"- Mall 钉扎 sha256: `{_MALL_PINNED['sha256']}`",
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
