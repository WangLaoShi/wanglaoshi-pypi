"""执行 examples/advanced 下 Notebook（E01–E22），产出跑通证据。

默认在当前解释器进程内执行 code cell（避免 Jupyter kernel 环境不一致）。
可选 --kernel 走 nbclient。

用法：
  python scripts/run_advanced_evidence.py
  python scripts/run_advanced_evidence.py --only 01_iris,04_breast_cancer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import nbformat


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


def _ensure_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")


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
        nb = nbformat.read(path, as_version=4)
        # shared namespace across cells
        g: dict = {"__name__": "__main__"}
        executed = 0
        buf = StringIO()
        for i, cell in enumerate(nb.cells):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source") or ""
            if isinstance(src, list):
                src = "".join(src)
            if not str(src).strip():
                continue
            try:
                # display() shim for notebooks
                if "display" not in g:
                    def display(x=None, *args, **kwargs):  # noqa: ANN001
                        return x

                    g["display"] = display
                exec(compile(src, f"{slug}.ipynb:cell{i}", "exec"), g, g)
                executed += 1
            except Exception as e:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = f"cell[{i}]: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
                result["cells_executed"] = executed
                log_path.write_text(f"FAILED {slug}\n{result['error']}\n", encoding="utf-8")
                result["finished_at"] = datetime.now(timezone.utc).isoformat()
                result["log"] = str(log_path.relative_to(ROOT))
                return result

        result["status"] = "passed"
        result["cells_executed"] = executed
        # Notebook 约定：DEMO=True 表示合成/演示回退；False 表示走了非 DEMO 路径
        demo_flag = g.get("DEMO", None)
        if demo_flag is True:
            result["data_mode"] = "demo"
        elif demo_flag is False:
            result["data_mode"] = str(g.get("data_mode") or "real_or_local")
        else:
            result["data_mode"] = str(g.get("data_mode") or "unspecified")
        for key in ("data_source", "n_rows", "data_sha256"):
            if key in g:
                result[key] = g.get(key)
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
        log_path.write_text(f"PASSED {slug}\ncells={result['cells_executed']}\n", encoding="utf-8")
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
        by_slug[slug] = item
        print(f"  -> {item['status']}", flush=True)
        if args.stop_on_fail and item["status"] not in {"passed"}:
            break

    # keep DEFAULT_ORDER then any extras
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
    evidence["summary"] = {
        "passed": passed,
        "failed": failed,
        "total": len(evidence["results"]),
        "demo_or_synthetic": demo_n,
        "real_or_local_claimed": real_n,
        "note": "passed≠真实数据验收；见 DATA_ACCEPTANCE.md",
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
        "> 发布级真实数据字段见 [`DATA_ACCEPTANCE.md`](DATA_ACCEPTANCE.md)。",
        "",
        f"- 生成时间(UTC): `{evidence['generated_at_utc']}`",
        f"- Python: `{evidence['python']}`",
        f"- 汇总: passed={passed}, failed={failed}, total={evidence['summary']['total']}, "
        f"demo={demo_n}, real_or_local_claimed={real_n}",
        "",
        "| Notebook | Status | Cells | data_mode | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in evidence["results"]:
        err = (r.get("error") or "").splitlines()[0][:100] if r.get("error") else ""
        note = err if r["status"] != "passed" else r.get("mode", "")
        mode = r.get("data_mode", "unspecified")
        md_lines.append(
            f"| `{r['slug']}` | **{r['status']}** | {r.get('cells_executed', 0)} | `{mode}` | {note} |"
        )
    md_lines += [
        "",
        "原始 JSON：`docs/evidence/advanced_evidence.json`",
        "",
        "复现：",
        "```bash",
        "python scripts/run_advanced_evidence.py",
        "```",
        "",
    ]
    (DOCS_EVIDENCE / "ADVANCED_EVIDENCE.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[done] {docs_json} summary={evidence['summary']}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
