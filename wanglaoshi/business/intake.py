"""D01 data-intake：原始文件只读备份、完整性校验、元信息登记。"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import StageResult


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def intake_file(
    source_path: str | Path,
    archive_dir: str | Path,
    *,
    required_columns: Optional[Sequence[str]] = None,
    period: Optional[str] = None,
    source_name: Optional[str] = None,
    copy: bool = True,
) -> StageResult:
    """接收单个数据文件并归档。

    - 源文件只读打开校验哈希
    - 复制到 archive_dir（默认不覆盖已有同名+哈希文件）
    - 若提供 required_columns，尝试用 pandas 读表头做缺列检查（失败则 warning，不阻断哈希归档）

    Returns:
        StageResult(stage_id='D01_intake')
    """
    src = Path(source_path)
    out_dir = Path(archive_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    block_reasons: List[str] = []
    rules = ["D01.require_file_exists", "D01.sha256", "D01.readonly_archive"]
    evidence: List[str] = []
    outputs: List[str] = []

    if not src.exists() or not src.is_file():
        return StageResult(
            stage_id="D01_intake",
            status="blocked",
            block_reasons=[f"source file not found: {src}"],
            rules=rules,
            input_paths=[str(src)],
        )

    file_hash = _sha256_file(src)
    evidence.append(f"sha256:{file_hash}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_name = f"{src.stem}__{stamp}__{file_hash[:12]}{src.suffix}"
    archived_path = out_dir / archived_name

    if copy:
        shutil.copy2(src, archived_path)
        outputs.append(str(archived_path))

    meta: Dict[str, Any] = {
        "source_path": str(src.resolve()),
        "source_name": source_name or src.name,
        "archived_path": str(archived_path) if copy else None,
        "sha256": file_hash,
        "size_bytes": src.stat().st_size,
        "period": period,
        "received_at_utc": stamp,
        "required_columns": list(required_columns) if required_columns else [],
        "missing_columns": [],
    }

    if required_columns:
        rules.append("D01.required_columns")
        try:
            import pandas as pd

            # 只读表头，避免大文件内存压力
            header = pd.read_csv(src, nrows=0)
            missing = [c for c in required_columns if c not in header.columns]
            meta["missing_columns"] = missing
            meta["columns"] = list(header.columns)
            if missing:
                warnings.append(f"missing required columns: {missing}")
        except Exception as e:  # noqa: BLE001 - 读表失败不应吞掉归档
            warnings.append(f"column check skipped: {e}")

    ledger_path = out_dir / "intake_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    outputs.append(str(ledger_path))

    status = "warning" if warnings else "passed"
    if block_reasons:
        status = "blocked"

    return StageResult(
        stage_id="D01_intake",
        status=status,
        data=meta,
        rules=rules,
        evidence_ids=evidence,
        warnings=warnings,
        block_reasons=block_reasons,
        input_paths=[str(src)],
        output_paths=outputs,
    )
