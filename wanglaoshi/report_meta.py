"""C06：报告章节完整性与图表元信息（标题/单位/周期/来源）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REQUIRED_REPORT_CHAPTERS = (
    "摘要",
    "数据说明",
    "核心指标",
    "结论",
)


@dataclass
class ChartMeta:
    title: str
    unit: str
    period: str
    source: str
    chart_type: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[str]:
        errs: List[str] = []
        for k in ("title", "unit", "period", "source"):
            if not str(getattr(self, k) or "").strip():
                errs.append(f"missing {k}")
        return errs


def write_chart_sidecar(image_path: str | Path, meta: ChartMeta) -> Path:
    p = Path(image_path)
    side = p.with_suffix(p.suffix + ".meta.json") if p.suffix else Path(str(p) + ".meta.json")
    # 更常见：foo.png -> foo.meta.json
    side = p.with_name(p.stem + ".meta.json")
    side.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return side


def load_chart_sidecar(image_path: str | Path) -> ChartMeta:
    p = Path(image_path)
    side = p.with_name(p.stem + ".meta.json")
    data = json.loads(side.read_text(encoding="utf-8"))
    return ChartMeta(**{k: data.get(k, "") for k in ("title", "unit", "period", "source", "chart_type", "notes")})


def check_report_chapters(text: str, required: Sequence[str] = REQUIRED_REPORT_CHAPTERS) -> List[str]:
    missing = [c for c in required if c not in text]
    return missing


def assert_report_ok(text: str, required: Optional[Sequence[str]] = None) -> None:
    req = list(required) if required is not None else list(REQUIRED_REPORT_CHAPTERS)
    missing = check_report_chapters(text, req)
    if missing:
        raise AssertionError(f"report missing chapters: {missing}")
