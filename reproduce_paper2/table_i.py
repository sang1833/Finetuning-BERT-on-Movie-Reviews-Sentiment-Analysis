#!/usr/bin/env python
"""C2: Print / export Table I-style dataset statistics vs Paper 2.

Example:
  python -m reproduce_paper2.table_i
  python -m reproduce_paper2.table_i --datasets imdb2,sst2,mr --out outputs/tables/table_i.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduce_paper2.config import DATASET_SPECS, PAPER_TABLE_I
from reproduce_paper2.data import dataset_label_stats, load_dataset_by_key


def _fmt_counts(counts: Dict[str, int], keys: List[str]) -> str:
    parts = [f"{k}={counts.get(k, 0)}" for k in keys if counts.get(k, 0) or k in counts]
    return " ".join(parts) if parts else str(counts)


def compare_dataset(key: str, limit_train=None, limit_test=None, seed: int = 42) -> dict:
    spec = DATASET_SPECS[key]
    paper = PAPER_TABLE_I.get(key, {})
    ds = load_dataset_by_key(key, limit_train=limit_train, limit_test=limit_test, seed=seed)
    actual = dataset_label_stats(ds)

    rows = []
    for split in ("train", "test"):
        act = actual[split]
        pap = paper.get(split, {})
        all_labels = list(dict.fromkeys(list(pap.keys()) + list(act.keys())))
        for lab in all_labels:
            a = int(act.get(lab, 0))
            p = pap.get(lab)
            rows.append(
                {
                    "dataset": key,
                    "split": split,
                    "label": lab,
                    "ours_count": a,
                    "paper_count": p if p is not None else "",
                    "delta": (a - p) if p is not None else "",
                }
            )
    return {
        "dataset": key,
        "label_names": spec.label_names,
        "actual": actual,
        "paper": paper,
        "rows": rows,
        "num_train": len(ds.train.texts),
        "num_test": len(ds.test.texts),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Table I stats vs Paper 2")
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_SPECS.keys()),
        help="Comma-separated dataset keys",
    )
    p.add_argument("--out", type=str, default="outputs/tables/table_i.csv")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    print("=== Paper 2 Table I comparison (ours vs paper counts) ===\n")
    for key in keys:
        if key not in DATASET_SPECS:
            print(f"[skip] unknown dataset {key}")
            continue
        try:
            info = compare_dataset(
                key,
                limit_train=args.limit_train,
                limit_test=args.limit_test,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{key}] ERROR: {exc}")
            continue

        print(f"## {key}")
        print(f"  ours  train={info['num_train']} test={info['num_test']}")
        print(f"  train: {_fmt_counts(info['actual']['train'], info['label_names'])}")
        print(f"  test:  {_fmt_counts(info['actual']['test'], info['label_names'])}")
        if info["paper"]:
            print(f"  paper train: {info['paper'].get('train', {})}")
            print(f"  paper test:  {info['paper'].get('test', {})}")
        print()
        all_rows.extend(info["rows"])
        summaries.append(
            {
                "dataset": key,
                "num_train": info["num_train"],
                "num_test": info["num_test"],
                "actual": info["actual"],
                "paper": info["paper"],
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "split", "label", "ours_count", "paper_count", "delta"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
