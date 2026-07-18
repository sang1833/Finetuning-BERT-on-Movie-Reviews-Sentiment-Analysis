"""Shared training experiment used by CLI and table runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import Adam
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .augment import apply_augment
from .config import TrainConfig
from .data import load_dataset_by_key, train_val_split
from .dataset_torch import make_loader
from .model import build_bert_bilstm
from .polarity import compare_original_vs_computed


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

    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


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


def run_experiment(
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train + evaluate one Paper 2 configuration; write artifacts to cfg.output_dir."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg.to_dict(), "device": str(device)}, f, indent=2)

    if verbose:
        print(f"=== Paper 2: {cfg.dataset} ({cfg.task}) augment={cfg.augment} ===")
        print(json.dumps({**cfg.to_dict(), "device": str(device)}, indent=2))

    if verbose:
        print(f"\n[1/5] Loading '{cfg.dataset}'...")
    dataset = load_dataset_by_key(
        cfg.dataset,
        limit_train=cfg.limit_train_samples,
        limit_test=cfg.limit_test_samples,
        seed=cfg.seed,
    )
    label_names = cfg.label_names or dataset.label_names
    train_split, val_split = train_val_split(
        dataset.train, val_ratio=cfg.val_ratio, seed=cfg.seed, stratified=True
    )

    if cfg.augment and cfg.augment != "none":
        if verbose:
            print(f"  Applying train augment: {cfg.augment}")
        train_split = apply_augment(
            train_split,
            mode=cfg.augment,
            factor=cfg.nlpaug_factor,
            seed=cfg.seed,
            model_name=cfg.model_name,
            max_length=min(cfg.max_length, 128),
            device=str(device),
        )

    if verbose:
        print(
            f"  train={len(train_split.texts)} val={len(val_split.texts)} "
            f"test={len(dataset.test.texts)} labels={label_names}"
        )

    if verbose:
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

    if verbose:
        print("\n[3/5] Building BERT+BiLSTM...")
    model = build_bert_bilstm(cfg).to(device)
    binary_head = model.binary_sigmoid_head
    trainable = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(
            f"  device={device} num_labels={cfg.num_labels} "
            f"trainable={sum(p.numel() for p in trainable):,}"
        )

    optimizer = Adam(
        trainable, lr=cfg.learning_rate, eps=cfg.epsilon, weight_decay=cfg.decay
    )
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )

    history_rows = []
    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    if verbose:
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
        if verbose:
            print(
                f"  epoch {epoch:02d}/{cfg.epochs}  "
                f"train_loss={train_loss:.4f}  val_acc={val_stats['accuracy'] * 100:.2f}%"
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

    if verbose:
        print("\n[5/5] Evaluating...")
    test_stats = evaluate(model, test_loader, device, binary_head)
    y_true, y_pred = test_stats["y_true"], test_stats["y_pred"]
    acc = test_stats["accuracy"]

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    names = [label_names[i] for i in present if i < len(label_names)]
    report = classification_report(
        y_true,
        y_pred,
        labels=present,
        target_names=names if len(names) == len(present) else None,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    op = compare_original_vs_computed(
        y_true, y_pred, mode=cfg.op_mode, label_names=label_names
    )

    if verbose:
        print(f"  Test accuracy: {acc * 100:.4f}%")
        if cfg.paper_accuracy_pct is not None:
            print(f"  Paper target: {cfg.paper_accuracy_pct}%")
        print(f"  OP original={op['original_op']} computed={op['computed_op']} match={op['match']}")

    results: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "framework": "pytorch",
        "device": str(device),
        "test_accuracy": acc,
        "test_accuracy_pct": acc * 100.0,
        "paper2_target_pct": cfg.paper_accuracy_pct,
        "best_val_accuracy": best_val_acc,
        "overall_polarity": dict(op),
        "paper_op_expected": cfg.paper_op,
        "num_test": int(len(y_true)),
        "label_names": label_names,
        "augment": cfg.augment,
        "config": cfg.to_dict(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), out_dir / "final.weights.pt")
    if verbose:
        print(f"  Wrote {out_dir / 'results.json'}")
    return results
