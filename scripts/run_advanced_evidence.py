#!/usr/bin/env python3
"""执行 examples/advanced 下 Notebook（E01–E22），产出跑通证据。

默认在当前解释器进程内执行 code cell（避免 Jupyter kernel 环境不一致）。
可选 --kernel 走 nbclient。

用法：
  python scripts/run_advanced_evidence.py
  python scripts/run_advanced_evidence.py --only 01_iris,04_breast_cancer
  python scripts/run_advanced_evidence.py --release-data   # 发布：必填真实数据案例不得 DEMO
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "examples" / "advanced"
OUT_DIR = NB_DIR / "outputs" / "evidence"
DOCS_EVIDENCE = ROOT / "docs" / "evidence"

DEFAULT_ORDER = [
    "01_iris",
    "02_titanic",
    "04_breast_cancer",
    "06_california_housing",
    "12_digit_recognizer",
    "03_wine_quality",
    "07_abalone",
    "09_mall_customers",
    "08_bike_sharing",
    "10_adult_census",
    "05_credit_card_fraud",
    "11_house_prices",
    "13_ab_testing",
    "14_hypothesis_testing",
    "15_timeseries_forecast",
    "16_survival_churn",
    "17_text_eda",
    "18_anomaly_detection",
    "19_association_rules",
    "20_causal_did",
    "21_order_refund_reconciliation",
    "22_movielens_ndcg",
]

# 发布验收：这些案例不允许用合成 DEMO 充当「真实数据通过」
# 教学烟测（默认）仍可 DEMO；加 --release-data 时单独判定
REQUIRE_REAL_DATA = {
    "05_credit_card_fraud",
    "08_bike_sharing",
    "09_mall_customers",
}


def _ensure_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")


def _data_mode_from_namespace(g: dict) -> dict:
    meta: dict = {}
    demo_flag = g.get("DEMO", None)
    if demo_flag is True:
        meta["data_mode"] = "demo"
    elif demo_flag is False:
        meta["data_mode"] = str(g.get("data_mode") or "real_or_local")
    else:
        meta["data_mode"] = str(g.get("data_mode") or "unspecified")
    for key in ("data_source", "n_rows", "data_sha256"):
        if key in g:
            meta[key] = g.get(key)
    return meta


def _collect_notebook_text(nb) -> str:
    chunks: list[str] = []
    for cell in nb.cells:
        for out in cell.get("outputs") or []:
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                chunks.append(text if isinstance(text, str) else "".join(text))
            elif out.get("output_type") in {"execute_result", "display_data"}:
                data = out.get("data") or {}
                plain = data.get("text/plain", "")
                chunks.append(plain if isinstance(plain, str) else "".join(plain))
    return "\n".join(chunks)


def _data_mode_from_output_text(blob: str) -> dict:
    """从 kernel 执行输出中解析 DEMO / data_mode（无进程内命名空间时用）。"""
    meta: dict = {"data_mode": "unspecified"}
    if re.search(r"DEMO\s*(mode\s*)?=\s*True", blob):
        meta["data_mode"] = "demo"
    elif re.search(r"DEMO\s*(mode\s*)?=\s*False", blob):
        meta["data_mode"] = "real_or_local"

    m = re.search(r"['\"]data_mode['\"]\s*:\s*['\"]([^'\"]+)['\"]", blob)
    if m:
        meta["data_mode"] = m.group(1)
    m = re.search(r"\bdata_mode\s*=\s*['\"]?([a-zA-Z0-9_]+)['\"]?", blob)
    if m and meta["data_mode"] == "unspecified":
        meta["data_mode"] = m.group(1)

    m = re.search(r"['\"]n_rows['\"]\s*:\s*(\d+)", blob)
    if m:
        meta["n_rows"] = int(m.group(1))
    m = re.search(r"['\"]data_source['\"]\s*:\s*['\"]([^'\"]+)['\"]", blob)
    if m:
        meta["data_source"] = m.group(1)
    m = re.search(r"['\"]data_sha256['\"]\s*:\s*['\"]([0-9a-fA-F]+)['\"]", blob)
    if m:
        meta["data_sha256"] = m.group(1)
    return meta


def _release_evidence_gaps(result: dict) -> list[str]:
    """按 DATA_ACCEPTANCE：真实数据通过须有 source / n_rows / sha256。"""
    gaps: list[str] = []
    source = result.get("data_source")
    if source is None or not str(source).strip() or str(source).strip() in {
        "unknown",
        "None",
        "null",
    }:
        gaps.append("data_source")
    try:
        if int(result.get("n_rows")) <= 0:  # type: ignore[arg-type]
            gaps.append("n_rows")
    except (TypeError, ValueError):
        gaps.append("n_rows")
    sha = str(result.get("data_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        gaps.append("data_sha256")
    return gaps


def _release_data_verdict(slug: str, result: dict) -> str:
    """返回 ok | n/a | data_fail。

    对 REQUIRE_REAL_DATA：仅 data_mode 以 real 开头不够，还必须具备
    data_source / n_rows / data_sha256（完整 64 位十六进制）。
    """
    if slug not in REQUIRE_REAL_DATA:
        return "n/a"
    if result.get("status") != "passed":
        return "data_fail"
    mode = str(result.get("data_mode") or "unspecified")
    if mode in {"demo", "synthetic", "unspecified", "missing"}:
        return "data_fail"
    if not mode.startswith("real"):
        return "data_fail"
    if _release_evidence_gaps(result):
        return "data_fail"
    return "ok"


def _read_notebook(path: Path) -> dict:
    """优先 nbformat；无依赖时退回 json（inprocess 足够）。"""
    try:
        import nbformat

        return nbformat.read(path, as_version=4)
    except ImportError:
        return json.loads(path.read_text(encoding="utf-8"))


def run_one_inprocess(slug: str) -> dict:
    path = NB_DIR / f"{slug}.ipynb"
    result = {
        "slug": slug,
        "path": str(path.relative_to(ROOT)),
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "cells_executed": 0,
        "mode": "inprocess",
    }
    if not path.exists():
        result["status"] = "missing"
        result["error"] = "notebook not found"
        return result

    log_path = OUT_DIR / f"{slug}.log"
    _ensure_path()
    old_cwd = os.getcwd()
    try:
        import matplotlib

        matplotlib.use("Agg")
        os.chdir(NB_DIR)
        nb = _read_notebook(path)
        g: dict = {"__name__": "__main__"}
        executed = 0
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source") or ""
            if isinstance(src, list):
                src = "".join(src)
            if not str(src).strip():
                continue
            try:
                if "display" not in g:

                    def display(x=None, *args, **kwargs):  # noqa: ANN001
                        return x

                    g["display"] = display
                exec(compile(src, f"{slug}.ipynb:cell{i}", "exec"), g, g)
                executed += 1
            except Exception as e:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = (
                    f"cell[{i}]: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
                )
                result["cells_executed"] = executed
                log_path.write_text(f"FAILED {slug}\n{result['error']}\n", encoding="utf-8")
                result["finished_at"] = datetime.now(timezone.utc).isoformat()
                result["log"] = str(log_path.relative_to(ROOT))
                return result

        result["status"] = "passed"
        result["cells_executed"] = executed
        result.update(_data_mode_from_namespace(g))
        log_path.write_text(
            f"PASSED {slug}\ncells={executed}\ndata_mode={result.get('data_mode')}\n",
            encoding="utf-8",
        )
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
        log_path.write_text(f"ERROR {slug}\n{result['error']}\n", encoding="utf-8")
    finally:
        os.chdir(old_cwd)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["log"] = str(log_path.relative_to(ROOT))
    return result


def run_one_kernel(slug: str, timeout: int) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = NB_DIR / f"{slug}.ipynb"
    result = {
        "slug": slug,
        "path": str(path.relative_to(ROOT)),
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "cells_executed": 0,
        "mode": "kernel",
    }
    if not path.exists():
        result["status"] = "missing"
        result["error"] = "notebook not found"
        return result

    log_path = OUT_DIR / f"{slug}.log"
    _ensure_path()
    try:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(NB_DIR)}},
        )
        client.execute(cwd=str(NB_DIR))
        result["status"] = "passed"
        result["cells_executed"] = sum(1 for c in nb.cells if c.cell_type == "code")
        result.update(_data_mode_from_output_text(_collect_notebook_text(nb)))
        log_path.write_text(
            f"PASSED {slug}\ncells={result['cells_executed']}\n"
            f"data_mode={result.get('data_mode')}\n",
            encoding="utf-8",
        )
    except CellExecutionError as e:
        result["status"] = "failed"
        result["error"] = str(e)[:2000]
        log_path.write_text(f"FAILED {slug}\n{e}\n", encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        log_path.write_text(f"ERROR {slug}\n{result['error']}\n", encoding="utf-8")
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["log"] = str(log_path.relative_to(ROOT))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="comma-separated slugs")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--kernel", action="store_true", help="use jupyter kernel via nbclient")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--merge", action="store_true", help="merge with existing docs evidence by slug")
    parser.add_argument(
        "--release-data",
        action="store_true",
        help=(
            "发布数据验收：REQUIRE_REAL_DATA 须 data_mode=real* 且具备 "
            "data_source / n_rows / data_sha256(64位)"
        ),
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.only.strip():
        slugs = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        slugs = DEFAULT_ORDER

    evidence = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_dir": str(NB_DIR.relative_to(ROOT)),
        "python": sys.executable,
        "timeout_sec": args.timeout,
        "release_data": bool(args.release_data),
        "require_real_data": sorted(REQUIRE_REAL_DATA),
        "results": [],
    }
    by_slug: dict = {}
    docs_json_existing = DOCS_EVIDENCE / "advanced_evidence.json"
    if args.merge and docs_json_existing.exists():
        try:
            prev = json.loads(docs_json_existing.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                by_slug[r["slug"]] = r
        except Exception:  # noqa: BLE001
            pass

    for slug in slugs:
        print(f"[run] {slug} ...", flush=True)
        item = run_one_kernel(slug, args.timeout) if args.kernel else run_one_inprocess(slug)
        item["release_data_verdict"] = _release_data_verdict(slug, item)
        if slug in REQUIRE_REAL_DATA and item["release_data_verdict"] == "data_fail":
            mode = str(item.get("data_mode") or "unspecified")
            if mode.startswith("real"):
                item["release_data_gaps"] = _release_evidence_gaps(item)
            else:
                item["release_data_gaps"] = [f"data_mode={mode}"]
        by_slug[slug] = item
        gaps = item.get("release_data_gaps")
        gap_note = f" gaps={gaps}" if gaps else ""
        print(
            f"  -> {item['status']} data_mode={item.get('data_mode', '—')} "
            f"release={item['release_data_verdict']}{gap_note}",
            flush=True,
        )
        if args.stop_on_fail and item["status"] not in {"passed"}:
            break
        if args.release_data and args.stop_on_fail and item["release_data_verdict"] == "data_fail":
            break

    ordered = []
    seen = set()
    for s in DEFAULT_ORDER:
        if s in by_slug:
            ordered.append(by_slug[s])
            seen.add(s)
    for s, r in by_slug.items():
        if s not in seen:
            ordered.append(r)
    evidence["results"] = ordered

    passed = sum(1 for r in evidence["results"] if r["status"] == "passed")
    failed = sum(1 for r in evidence["results"] if r["status"] in {"failed", "error", "missing"})
    demo_n = sum(1 for r in evidence["results"] if r.get("data_mode") == "demo")
    real_n = sum(
        1
        for r in evidence["results"]
        if r["status"] == "passed" and str(r.get("data_mode", "")).startswith("real")
    )
    data_fail_n = sum(1 for r in evidence["results"] if r.get("release_data_verdict") == "data_fail")
    evidence["summary"] = {
        "passed": passed,
        "failed": failed,
        "total": len(evidence["results"]),
        "demo_or_synthetic": demo_n,
        "real_or_local_claimed": real_n,
        "release_data_fail": data_fail_n,
        "note": "passed≠真实数据验收；--release-data 时 REQUIRE_REAL_DATA 单独判定",
    }

    out = OUT_DIR / "evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    DOCS_EVIDENCE.mkdir(parents=True, exist_ok=True)
    docs_json = DOCS_EVIDENCE / "advanced_evidence.json"
    docs_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Advanced Notebook 跑通证据",
        "",
        "> **验收边界：** `passed` 只证明执行未抛错，**不**等于真实数据验收通过。",
        "> `data_mode=demo` 表示 Notebook 内 `DEMO=True`（合成/演示回退）。",
        "> 发布必填真实数据案例：`" + "`, `".join(sorted(REQUIRE_REAL_DATA)) + "`。",
        "> 发布级字段见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)。",
        "",
        f"- 生成时间(UTC): `{evidence['generated_at_utc']}`",
        f"- Python: `{evidence['python']}`",
        f"- release_data 模式: `{args.release_data}`",
        f"- 汇总: passed={passed}, failed={failed}, total={evidence['summary']['total']}, "
        f"demo={demo_n}, real_or_local_claimed={real_n}, release_data_fail={data_fail_n}",
        "",
        "| Notebook | Status | Cells | data_mode | release_data | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in evidence["results"]:
        err = (r.get("error") or "").splitlines()[0][:100] if r.get("error") else ""
        note = err if r["status"] != "passed" else r.get("mode", "")
        mode = r.get("data_mode", "unspecified")
        verdict = r.get("release_data_verdict", "n/a")
        md_lines.append(
            f"| `{r['slug']}` | **{r['status']}** | {r.get('cells_executed', 0)} | "
            f"`{mode}` | `{verdict}` | {note} |"
        )
    md_lines += [
        "",
        "原始 JSON：`docs/evidence/advanced_evidence.json`",
        "",
        "复现：",
        "```bash",
        "python scripts/run_advanced_evidence.py",
        "python scripts/run_advanced_evidence.py --release-data",
        "```",
        "",
    ]
    (DOCS_EVIDENCE / "ADVANCED_EVIDENCE.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[done] {docs_json} summary={evidence['summary']}", flush=True)
    if failed:
        return 1
    if args.release_data and data_fail_n:
        print(
            f"[release-data] {data_fail_n} required-real case(s) failed data acceptance",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
