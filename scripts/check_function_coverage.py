#!/usr/bin/env python3
"""检查 wanglaoshi 包「函数级」覆盖率是否达到 100%。

口径：模块级函数 + 类方法（含 if/try 块内定义；不含嵌套 def）。
某函数体内任一语句被执行即算覆盖。

用法：
  python -m pytest -m "quick or submission" tests/ --cov=wanglaoshi --cov-report=json:artifacts/coverage.json
  python scripts/check_function_coverage.py artifacts/coverage.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple


def _collect_functions(body: list, class_name: str = "") -> List[Tuple[str, int, int, Optional[str]]]:
    """返回 (qualname, start, end, gate) ；gate 为依赖开关名（如 TORCH_AVAILABLE）。"""
    out: List[Tuple[str, int, int, Optional[str]]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            q = f"{class_name}.{node.name}" if class_name else node.name
            out.append((q, node.lineno, end, None))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(item, "end_lineno", None) or item.lineno
                    out.append((f"{node.name}.{item.name}", item.lineno, end, None))
                elif isinstance(item, (ast.If, ast.Try)):
                    out.extend(_collect_functions([item], class_name=node.name))
        elif isinstance(node, ast.If):
            gate = None
            test = node.test
            if isinstance(test, ast.Name):
                gate = test.id
            elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(test.operand, ast.Name):
                gate = f"!{test.operand.id}"
            for sub in _collect_functions(node.body, class_name=class_name):
                out.append((sub[0], sub[1], sub[2], gate if gate is not None else sub[3]))
            else_gate = None
            if gate and not gate.startswith("!"):
                else_gate = f"!{gate}"
            elif gate and gate.startswith("!"):
                else_gate = gate[1:]
            for sub in _collect_functions(node.orelse, class_name=class_name):
                out.append((sub[0], sub[1], sub[2], else_gate if else_gate is not None else sub[3]))
        elif isinstance(node, ast.Try):
            out.extend(_collect_functions(node.body, class_name=class_name))
            for handler in node.handlers:
                out.extend(_collect_functions(handler.body, class_name=class_name))
            out.extend(_collect_functions(node.orelse, class_name=class_name))
            out.extend(_collect_functions(node.finalbody, class_name=class_name))
    return out


def _stmt_lines(path: Path, start: int, end: int) -> Set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: Set[int] = set()
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        if start <= lineno <= end:
            lines.add(lineno)
    return lines or {start}


def _gate_active(gate: Optional[str]) -> bool:
    if gate is None:
        return True
    try:
        import wanglaoshi.MLDL as mldl  # local import

        torch_on = bool(getattr(mldl, "TORCH_AVAILABLE", False))
    except Exception:
        torch_on = False
    if gate == "TORCH_AVAILABLE":
        return torch_on
    if gate == "!TORCH_AVAILABLE":
        return not torch_on
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("coverage_json", type=Path)
    ap.add_argument("--root", type=Path, default=Path("wanglaoshi"))
    args = ap.parse_args()

    data = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    executed = {}
    missing = {}
    for f, v in data.get("files", {}).items():
        key = f.replace("\\", "/")
        if "wanglaoshi/" in key:
            key = "wanglaoshi/" + key.split("wanglaoshi/")[-1]
        executed[key] = set(v.get("executed_lines") or [])
        missing[key] = set(v.get("missing_lines") or [])

    uncovered: List[str] = []
    covered_n = 0
    total_n = 0
    skipped_inactive = 0
    for path in sorted(args.root.rglob("*.py")):
        rel = str(path).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, start, end, gate in _collect_functions(tree.body):
            if not _gate_active(gate):
                skipped_inactive += 1
                continue
            total_n += 1
            stmts = _stmt_lines(path, start, end)
            ex = executed.get(rel, set())
            miss = missing.get(rel, set())
            if ex:
                hit = any(L in ex for L in stmts)
            elif rel in missing:
                hit = any(L not in miss for L in stmts)
            else:
                hit = False
            if hit:
                covered_n += 1
            else:
                uncovered.append(f"{rel}:{start}  {name}")

    pct = 100.0 * covered_n / total_n if total_n else 100.0
    print(f"function coverage: {covered_n}/{total_n} = {pct:.1f}% (skipped inactive gates: {skipped_inactive})")
    if uncovered:
        print("uncovered functions:")
        for u in uncovered:
            print(f"  {u}")
        return 1
    print("OK: 100% function coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
