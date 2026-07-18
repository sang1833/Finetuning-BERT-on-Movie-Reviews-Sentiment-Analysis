#!/usr/bin/env python
"""C1: Build Paper 2 Table II–VI CSV/Markdown from results.json files.

Does **not** retrain. Reads outputs/<run>/results.json produced by train /
run_all_ours.

Example:
  python -m reproduce_paper2.export_tables --results-root outputs --out-dir outputs/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduce_paper2.config import (
    PAPER_TABLE_II_BASELINES,
    PAPER_TABLE_III_BASELINES,
    PAPER_TABLE_IV_BASELINES,
    PAPER_TABLE_V,
)


def _load_results(results_root: Path) -> Dict[str, dict]:
    """Map run_key → results dict.

    run_key examples: imdb2, sst5_nlpaug, sst5_smote
    Prefer exact folder names under results_root.
    """
    found: Dict[str, dict] = {}
    if not results_root.is_dir():
        return found
    for path in results_root.rglob("results.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # key: relative parent name
        key = path.parent.name
        found[key] = data
        # also index by dataset if bare
        ds = data.get("dataset")
        aug = data.get("augment") or "none"
        if ds and aug == "none" and ds not in found:
            found[ds] = data
        if ds and aug != "none":
            found[f"{ds}_{aug}"] = data
    return found


def _acc(results: Dict[str, dict], key: str) -> Optional[float]:
    r = results.get(key)
    if not r:
        return None
    return r.get("test_accuracy_pct")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def export_table_ii(results: Dict[str, dict], out_dir: Path) -> str:
    cols = ["imdb2", "mr", "sst2", "amazon2"]
    fieldnames = ["model"] + cols
    rows = []
    for model, scores in PAPER_TABLE_II_BASELINES.items():
        row = {"model": model}
        for c in cols:
            v = scores.get(c)
            row[c] = "" if v is None else v
        rows.append(row)
    # Ours (repro)
    ours = {"model": "Ours (repro)"}
    for c in cols:
        v = _acc(results, c)
        ours[c] = "" if v is None else round(v, 2)
    rows.append(ours)
    _write_csv(out_dir / "table_ii.csv", rows, fieldnames)

    md_rows = []
    for r in rows:
        md_rows.append([str(r["model"])] + [str(r[c]) if r[c] != "" else "-" for c in cols])
    md = "## Table II — Binary accuracy (%)\n\n" + _md_table(
        ["Model"] + [c.upper() for c in cols], md_rows
    )
    return md


def export_table_iii(results: Dict[str, dict], out_dir: Path) -> str:
    cols = ["imdb3", "imdb4"]
    fieldnames = ["model"] + cols
    rows = []
    for model, scores in PAPER_TABLE_III_BASELINES.items():
        row = {"model": model}
        for c in cols:
            v = scores.get(c)
            row[c] = "" if v is None else v
        rows.append(row)
    ours = {"model": "Ours (repro)"}
    for c in cols:
        v = _acc(results, c)
        ours[c] = "" if v is None else round(v, 2)
    rows.append(ours)
    _write_csv(out_dir / "table_iii.csv", rows, fieldnames)
    md_rows = [
        [str(r["model"])] + [str(r[c]) if r[c] != "" else "-" for c in cols] for r in rows
    ]
    return "## Table III — IMDb-3 / IMDb-4 accuracy (%)\n\n" + _md_table(
        ["Model", "IMDb-3", "IMDb-4"], md_rows
    )


def export_table_iv(results: Dict[str, dict], out_dir: Path) -> str:
    cols = ["sst5", "amazon5"]
    # Prefer nlpaug run for sst5 if present (paper best)
    fieldnames = ["model"] + cols
    rows = []
    for model, scores in PAPER_TABLE_IV_BASELINES.items():
        row = {"model": model}
        for c in cols:
            v = scores.get(c)
            row[c] = "" if v is None else v
        rows.append(row)
    ours = {"model": "Ours (repro)"}
    for c in cols:
        if c == "sst5":
            v = _acc(results, "sst5_nlpaug") or _acc(results, "sst5")
        else:
            v = _acc(results, c)
        ours[c] = "" if v is None else round(v, 2)
    rows.append(ours)
    _write_csv(out_dir / "table_iv.csv", rows, fieldnames)
    md_rows = [
        [str(r["model"])] + [str(r[c]) if r[c] != "" else "-" for c in cols] for r in rows
    ]
    return "## Table IV — Five-class accuracy (%)\n\n" + _md_table(
        ["Model", "SST-5", "Amazon-5"], md_rows
    )


def export_table_v(results: Dict[str, dict], out_dir: Path) -> str:
    fieldnames = ["variant", "paper_acc", "ours_acc"]
    mapping = [
        ("BERT+BiLSTM", "sst5", PAPER_TABLE_V["BERT+BiLSTM"]),
        ("BERT+BiLSTM+SMOTE", "sst5_smote", PAPER_TABLE_V["BERT+BiLSTM+SMOTE"]),
        ("BERT+BiLSTM+NLPAUG", "sst5_nlpaug", PAPER_TABLE_V["BERT+BiLSTM+NLPAUG"]),
    ]
    rows = []
    for name, key, paper in mapping:
        v = _acc(results, key)
        rows.append(
            {
                "variant": name,
                "paper_acc": paper,
                "ours_acc": "" if v is None else round(v, 2),
            }
        )
    _write_csv(out_dir / "table_v.csv", rows, fieldnames)
    md_rows = [
        [r["variant"], str(r["paper_acc"]), str(r["ours_acc"]) if r["ours_acc"] != "" else "-"]
        for r in rows
    ]
    return "## Table V — SST-5 accuracy improvement (%)\n\n" + _md_table(
        ["Variant", "Paper", "Ours (repro)"], md_rows
    )


def export_table_vi(results: Dict[str, dict], out_dir: Path) -> str:
    fieldnames = [
        "dataset",
        "paper_op",
        "original_op",
        "computed_op",
        "match",
        "test_accuracy_pct",
    ]
    # Prefer canonical keys
    keys = [
        "imdb2",
        "imdb3",
        "imdb4",
        "mr",
        "sst2",
        "sst5",
        "sst5_nlpaug",
        "amazon2",
        "amazon5",
    ]
    rows = []
    seen = set()
    for key in keys:
        r = results.get(key)
        if not r:
            continue
        ds = r.get("dataset", key)
        if ds in seen and key != "sst5_nlpaug":
            continue
        seen.add(ds if key != "sst5_nlpaug" else key)
        op = r.get("overall_polarity") or {}
        rows.append(
            {
                "dataset": key,
                "paper_op": r.get("paper_op_expected") or "",
                "original_op": op.get("original_op", ""),
                "computed_op": op.get("computed_op", ""),
                "match": op.get("match", ""),
                "test_accuracy_pct": round(r.get("test_accuracy_pct", 0), 2)
                if r.get("test_accuracy_pct") is not None
                else "",
            }
        )
    _write_csv(out_dir / "table_vi.csv", rows, fieldnames)
    md_rows = [
        [
            r["dataset"],
            str(r["paper_op"]),
            str(r["original_op"]),
            str(r["computed_op"]),
            str(r["match"]),
            str(r["test_accuracy_pct"]),
        ]
        for r in rows
    ]
    return "## Table VI — Overall polarity\n\n" + _md_table(
        ["Dataset", "Paper OP", "Original OP", "Computed OP", "Match", "Acc%"],
        md_rows,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export Paper 2 tables from results.json")
    p.add_argument("--results-root", type=str, default="outputs")
    p.add_argument("--out-dir", type=str, default="outputs/tables")
    args = p.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_root)
    print(f"Found {len(results)} result files under {results_root}")
    for k in sorted(results):
        acc = results[k].get("test_accuracy_pct")
        print(f"  {k}: acc={acc}")

    parts = [
        "# Paper 2 reproduction tables (Ours column filled from local runs)\n",
        export_table_ii(results, out_dir),
        export_table_iii(results, out_dir),
        export_table_iv(results, out_dir),
        export_table_v(results, out_dir),
        export_table_vi(results, out_dir),
        "\n*Baseline rows are copied from the paper (not re-trained).*\n",
    ]
    md_path = out_dir / "tables.md"
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"CSV files in {out_dir}")


if __name__ == "__main__":
    main()
