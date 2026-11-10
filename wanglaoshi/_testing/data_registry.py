"""C02：数据登记（来源 / 使用条件 / 版本 / 哈希 / 缓存路径）。

下载与测试分离：本模块只登记与校验本地文件，不负责下载。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def file_sha256(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class DataAsset:
    name: str
    path: str
    source: str
    license_or_terms: str
    version: str
    sha256: str
    cached: bool = False
    notes: str = ""
    registered_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataRegistry:
    """本地数据台账。"""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._assets: Dict[str, DataAsset] = {}
        if self.ledger_path.exists():
            self.load()

    def register(
        self,
        name: str,
        path: str | Path,
        *,
        source: str,
        license_or_terms: str,
        version: str,
        require_exists: bool = True,
        notes: str = "",
    ) -> DataAsset:
        p = Path(path)
        if require_exists and not p.exists():
            raise FileNotFoundError(f"data file not found (download separately): {p}")
        digest = file_sha256(p) if p.exists() else ""
        asset = DataAsset(
            name=name,
            path=str(p.resolve()) if p.exists() else str(p),
            source=source,
            license_or_terms=license_or_terms,
            version=version,
            sha256=digest,
            cached=p.exists(),
            notes=notes,
        )
        self._assets[name] = asset
        self.save()
        return asset

    def verify(self, name: str) -> bool:
        asset = self._assets[name]
        p = Path(asset.path)
        if not p.exists():
            return False
        return file_sha256(p) == asset.sha256

    def get(self, name: str) -> DataAsset:
        return self._assets[name]

    def list(self) -> List[DataAsset]:
        return list(self._assets.values())

    def save(self) -> None:
        payload = {k: v.to_dict() for k, v in self._assets.items()}
        self.ledger_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        self._assets = {k: DataAsset(**v) for k, v in raw.items()}
