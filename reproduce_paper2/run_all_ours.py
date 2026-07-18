#!/usr/bin/env python
"""C1: Run all Paper 2 "Ours" experiments and export tables II–VI.

Default job list covers:
  - Table II: imdb2, mr, sst2, amazon2
  - Table III: imdb3, imdb4
  - Table IV: sst5 (bare), amazon5
  - Table V: sst5 + smote, sst5 + nlpaug
  - Table VI: OP fields from every successful run

Examples:
  # Smoke (fast, not paper-quality)
  python -m reproduce_paper2.run_all_ours --smoke

  # Full paper defaults (long, GPU recommended)
  python -m reproduce_paper2.run_all_ours --device cuda

  # Subset
  python -m reproduce_paper2.run_all_ours --only imdb2,sst2,sst5 --smoke

  # Export only from existing results
  python -m reproduce_paper2.run_all_ours --export-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduce_paper2.config import config_for_dataset
from reproduce_paper2.experiment import run_experiment
from reproduce_paper2.export_tables import main as export_main


@dataclass
class Job:
    dataset: str
    augment: str = "none"
    out_name: Optional[str] = None

    @property
    def run_name(self) -> str:
        if self.out_name:
            return self.out_name
        if self.augment and self.augment != "none":
            return f"{self.dataset}_{self.augment}"
        return self.dataset


DEFAULT_JOBS: List[Job] = [
    # Table II
    Job("imdb2"),
    Job("mr"),
    Job("sst2"),
    Job("amazon2"),
    # Table III
    Job("imdb3"),
    Job("imdb4"),
    # Table IV / V
    Job("sst5"),
    Job("sst5", augment="smote"),
    Job("sst5", augment="nlpaug"),
    Job("amazon5"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all Ours experiments + export tables")
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--tables-dir", type=str, default="outputs/tables")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch, small limits (pipeline check, not paper numbers)",
    )
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma list of run names or datasets (e.g. imdb2,sst5_nlpaug)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run if results.json already exists",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help="Only build tables from existing results",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue remaining jobs if one fails (default True)",
    )
    p.add_argument("--bilstm-input", choices=["cls", "sequence"], default="cls")
    p.add_argument("--no-freeze-bert", action="store_true")
    return p.parse_args()


def _filter_jobs(jobs: List[Job], only: Optional[str]) -> List[Job]:
    if not only:
        return jobs
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    out = []
    for j in jobs:
        if j.run_name in wanted or j.dataset in wanted:
            out.append(j)
    return out


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    tables_dir = Path(args.tables_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not args.export_only:
        jobs = _filter_jobs(DEFAULT_JOBS, args.only)
        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Device: {device}")
        print(f"Jobs ({len(jobs)}): {[j.run_name for j in jobs]}")

        summary = []
        for job in jobs:
            run_dir = out_root / job.run_name
            results_path = run_dir / "results.json"
            if args.skip_existing and results_path.is_file():
                print(f"\n[skip] {job.run_name} (exists)")
                try:
                    summary.append(json.loads(results_path.read_text(encoding="utf-8")))
                except Exception:  # noqa: BLE001
                    pass
                continue

            print(f"\n######## RUN {job.run_name} ########")
            overrides = {
                "seed": args.seed,
                "bilstm_input": args.bilstm_input,
                "freeze_bert": not args.no_freeze_bert,
                "augment": job.augment,
                "output_dir": str(run_dir),
            }
            if args.smoke:
                overrides["epochs"] = 1
                overrides["limit_train_samples"] = args.limit_train or 256
                overrides["limit_test_samples"] = args.limit_test or 128
                overrides["batch_size"] = 8
            else:
                if args.epochs is not None:
                    overrides["epochs"] = args.epochs
                if args.limit_train is not None:
                    overrides["limit_train_samples"] = args.limit_train
                if args.limit_test is not None:
                    overrides["limit_test_samples"] = args.limit_test

            try:
                set_seed(args.seed)
                cfg = config_for_dataset(job.dataset, **overrides)
                # Amazon full can be huge — if not smoke and no limit, still OK but warn
                if job.dataset.startswith("amazon") and not args.smoke and args.limit_train is None:
                    print(
                        "[warn] Full Amazon can be very large/slow. "
                        "Consider --limit-train 50000 for a first pass."
                    )
                result = run_experiment(cfg, device=device, verbose=True)
                summary.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {job.run_name}: {exc}")
                traceback.print_exc()
                err_path = run_dir / "error.txt"
                run_dir.mkdir(parents=True, exist_ok=True)
                err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
                if not args.continue_on_error:
                    raise

        summary_path = tables_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "dataset": s.get("dataset"),
                        "augment": s.get("augment"),
                        "test_accuracy_pct": s.get("test_accuracy_pct"),
                        "op": s.get("overall_polarity"),
                    }
                    for s in summary
                ],
                f,
                indent=2,
            )
        print(f"\nWrote {summary_path}")

    # Always export tables
    print("\n######## EXPORT TABLES ########")
    sys.argv = [
        "export_tables",
        "--results-root",
        str(out_root),
        "--out-dir",
        str(tables_dir),
    ]
    export_main()
    print("\nDone. See", tables_dir)


if __name__ == "__main__":
    main()
