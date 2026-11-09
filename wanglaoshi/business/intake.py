"""D01 data-intake：原始文件只读备份、完整性校验、元信息登记。

初学者可以把它理解成「收货入库」：
1. 确认文件在不在
2. 算指纹（sha256），以后能证明「分析用的就是这份原始文件」
3. 复制一份到归档目录（尽量不要改动原件）
4. 可选：检查表头有没有必需列；缺列给 warning，但不阻碍先把文件收进来
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import StageResult


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 sha256，避免大文件一次性读进内存。"""
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

    参数说明：
    - source_path       : 原始 CSV 等文件路径
    - archive_dir       : 归档目录（不存在会自动创建）
    - required_columns  : 期望表头列名；缺列 → warning（仍会归档文件本身）
    - period            : 业务周期标记，如 2024-Q1（写进台账，方便以后对账）
    - source_name       : 来源名称；默认用文件名
    - copy              : True 时复制到归档目录；False 只登记不复制

    返回：
        StageResult(stage_id='D01_intake')
        - 文件不存在 → status=blocked
        - 缺必需列   → status=warning
        - 其他正常   → status=passed
    """
    src = Path(source_path)
    out_dir = Path(archive_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    block_reasons: List[str] = []
    # rules 用来追溯「按哪几条规范检查过」
    rules = ["D01.require_file_exists", "D01.sha256", "D01.readonly_archive"]
    evidence: List[str] = []
    outputs: List[str] = []

    # ---- 1) 文件必须存在，否则直接阻断 ----
    if not src.exists() or not src.is_file():
        return StageResult(
            stage_id="D01_intake",
            status="blocked",
            block_reasons=[f"source file not found: {src}"],
            rules=rules,
            input_paths=[str(src)],
        )

    # ---- 2) 计算哈希（证据）----
    file_hash = _sha256_file(src)
    evidence.append(f"sha256:{file_hash}")

    # 归档文件名带时间戳 + 哈希前缀，避免同名互相覆盖
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived_name = f"{src.stem}__{stamp}__{file_hash[:12]}{src.suffix}"
    archived_path = out_dir / archived_name

    if copy:
        # copy2 尽量保留修改时间等元数据；原文件保持不动
        shutil.copy2(src, archived_path)
        outputs.append(str(archived_path))

    # ---- 3) 台账元信息 ----
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

    # ---- 4) 可选：只读表头检查必需列 ----
    if required_columns:
        rules.append("D01.required_columns")
        try:
            import pandas as pd

            # nrows=0：只读表头，适合大文件
            header = pd.read_csv(src, nrows=0)
            missing = [c for c in required_columns if c not in header.columns]
            meta["missing_columns"] = missing
            meta["columns"] = list(header.columns)
            if missing:
                warnings.append(f"missing required columns: {missing}")
        except Exception as e:  # noqa: BLE001 - 读表失败不应吞掉已经完成的哈希归档
            warnings.append(f"column check skipped: {e}")

    # 追加写入 JSONL 台账，便于多次接收后审计
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
