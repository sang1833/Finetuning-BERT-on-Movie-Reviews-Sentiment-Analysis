#!/usr/bin/env python
"""Train / evaluate BERT+BiLSTM on IMDb-2 with Paper 2 binary hyperparameters.

Uses PyTorch + HuggingFace Transformers (portable; original paper code was TF).

Example:
  python -m reproduce_paper2.train
  python -m reproduce_paper2.train --epochs 1 --limit-train 512 --limit-test 256
  python -m reproduce_paper2.train --bilstm-input sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import Adam

# Allow `python reproduce_paper2/train.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproduce_paper2.config import default_binary_imdb_config
from reproduce_paper2.data import load_imdb2, train_val_split
from reproduce_paper2.dataset_torch import make_loader
from reproduce_paper2.model import build_bert_bilstm
from reproduce_paper2.polarity import compare_original_vs_computed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Paper 2 IMDb-2 binary run")
    p.add_argument("--output-dir", type=str, default="outputs/imdb2_binary")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bilstm-input",
        choices=["cls", "sequence"],
        default="cls",
        help="cls = Paper 2 description; sequence = legacy full-token BiLSTM",
    )
    p.add_argument("--no-freeze-bert", action="store_true")
    p.add_argument("--limit-train", type=int, default=None, help="Smoke-test subset")
    p.add_argument("--limit-test", type=int, default=None, help="Smoke-test subset")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 ...")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, binary_head: bool):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    n = 0
    criterion = nn.BCEWithLogitsLoss() if binary_head else nn.CrossEntropyLoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        if binary_head:
            loss = criterion(logits, labels.float())
            pred = (torch.sigmoid(logits) >= 0.5).long()
        else:
            loss = criterion(logits, labels)
            pred = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    acc = accuracy_score(y_true, y_pred)
    return {"loss": total_loss / max(n, 1), "accuracy": acc, "y_true": y_true, "y_pred": y_pred}


def train_one_epoch(model, loader, optimizer, scheduler, device, binary_head: bool):
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.BCEWithLogitsLoss() if binary_head else nn.CrossEntropyLoss()

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        if binary_head:
            loss = criterion(logits, labels.float())
        else:
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return total_loss / max(n, 1)


def main() -> None:
    args = parse_args()
    overrides = {
        "output_dir": args.output_dir,
        "seed": args.seed,
        "bilstm_input": args.bilstm_input,
        "freeze_bert": not args.no_freeze_bert,
        "val_ratio": args.val_ratio,
        "limit_train_samples": args.limit_train,
        "limit_test_samples": args.limit_test,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    cfg = default_binary_imdb_config(**overrides)
    set_seed(cfg.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg.to_dict(), "device": str(device)}, f, indent=2)

    print("=== Paper 2 reproduction: IMDb-2 binary (PyTorch) ===")
    print(json.dumps({**cfg.to_dict(), "device": str(device)}, indent=2))

    print("\n[1/5] Loading ACL IMDb (official train/test)...")
    dataset = load_imdb2(
        limit_train=cfg.limit_train_samples,
        limit_test=cfg.limit_test_samples,
        seed=cfg.seed,
    )
    train_split, val_split = train_val_split(
        dataset.train, val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    print(
        f"  train={len(train_split.texts)} val={len(val_split.texts)} "
        f"test={len(dataset.test.texts)} labels={dataset.label_names}"
    )

    print("\n[2/5] Tokenizer + DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_loader = make_loader(
        train_split.texts,
        train_split.labels,
        tokenizer,
        cfg.max_length,
        cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    val_loader = make_loader(
        val_split.texts,
        val_split.labels,
        tokenizer,
        cfg.max_length,
        cfg.batch_size,
        shuffle=False,
        seed=cfg.seed,
    )
    test_loader = make_loader(
        dataset.test.texts,
        dataset.test.labels,
        tokenizer,
        cfg.max_length,
        cfg.batch_size,
        shuffle=False,
        seed=cfg.seed,
    )

    print("\n[3/5] Building BERT+BiLSTM...")
    model = build_bert_bilstm(cfg).to(device)
    binary_head = model.binary_sigmoid_head
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  device={device} trainable_params={sum(p.numel() for p in trainable):,}")

    # Paper 2: Adam lr=3e-5, eps=1e-8
    optimizer = Adam(trainable, lr=cfg.learning_rate, eps=cfg.epsilon)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    history_rows = []
    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    print("\n[4/5] Training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, binary_head
        )
        val_stats = evaluate(model, val_loader, device, binary_head)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
        }
        history_rows.append(row)
        print(
            f"  epoch {epoch:02d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_stats['loss']:.4f}  "
            f"val_acc={val_stats['accuracy'] * 100:.2f}%"
        )
        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "val_accuracy": best_val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(history_rows)

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best checkpoint (val_acc={best_val_acc * 100:.2f}%)")

    print("\n[5/5] Evaluating on official IMDb test split...")
    test_stats = evaluate(model, test_loader, device, binary_head)
    y_true, y_pred = test_stats["y_true"], test_stats["y_pred"]
    acc = test_stats["accuracy"]
    report = classification_report(
        y_true, y_pred, target_names=dataset.label_names, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  Test accuracy: {acc * 100:.4f}%")
    print("  Paper 2 reported (IMDb-2): 97.67%")
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", cm)

    op = compare_original_vs_computed(y_true, y_pred, mode="binary")
    print("\nOverall polarity (Algorithm 2, coef=1.2):")
    print(f"  Original OP (gold labels):   {op['original_op']}")
    print(f"  Computed OP (predictions):   {op['computed_op']}")
    print(f"  Match: {op['match']}")
    print("  Paper 2 Table VI (IMDb-2): Original=Neutral, Computed=Neutral")

    results = {
        "dataset": "imdb2",
        "framework": "pytorch",
        "device": str(device),
        "test_accuracy": acc,
        "test_accuracy_pct": acc * 100.0,
        "paper2_target_pct": 97.67,
        "best_val_accuracy": best_val_acc,
        "overall_polarity": op,
        "num_test": int(len(y_true)),
        "config": cfg.to_dict(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), out_dir / "final.weights.pt")
    print(f"\nWrote results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
