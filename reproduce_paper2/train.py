#!/usr/bin/env python
"""Unified Paper 2 trainer (Phases A–C).

Examples:
  python -m reproduce_paper2.train --dataset imdb2
  python -m reproduce_paper2.train --dataset sst5 --augment nlpaug
  python -m reproduce_paper2.train --dataset sst5 --augment smote
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduce_paper2.config import DATASET_SPECS, config_for_dataset
from reproduce_paper2.experiment import run_experiment


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce Paper 2 BERT+BiLSTM (binary + fine-grained)"
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="imdb2",
        choices=sorted(DATASET_SPECS.keys()),
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bilstm-input", choices=["cls", "sequence"], default="cls")
    p.add_argument("--no-freeze-bert", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--augment",
        choices=["none", "nlpaug", "smote", "smote_bert"],
        default="none",
        help="smote/smote_bert = SMOTE in BERT [CLS] space (Paper Table V)",
    )
    p.add_argument("--nlpaug-factor", type=float, default=0.5)
    p.add_argument("--list-datasets", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        for key, spec in DATASET_SPECS.items():
            target = (
                f"{spec.paper_accuracy_pct:.2f}%"
                if spec.paper_accuracy_pct is not None
                else "n/a"
            )
            print(
                f"  {key:10s}  labels={spec.num_labels}  task={spec.task:13s}  "
                f"OP={spec.op_mode:6s}  paper_acc={target:8s}  {spec.description}"
            )
        return

    overrides = {
        "seed": args.seed,
        "bilstm_input": args.bilstm_input,
        "freeze_bert": not args.no_freeze_bert,
        "val_ratio": args.val_ratio,
        "limit_train_samples": args.limit_train,
        "limit_test_samples": args.limit_test,
        "augment": "smote" if args.augment == "smote_bert" else args.augment,
        "nlpaug_factor": args.nlpaug_factor,
    }
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    cfg = config_for_dataset(args.dataset, **overrides)
    if args.output_dir is None and cfg.augment != "none":
        cfg.output_dir = f"outputs/{cfg.dataset}_{cfg.augment}"

    set_seed(cfg.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_experiment(cfg, device=device, verbose=True)


if __name__ == "__main__":
    main()
