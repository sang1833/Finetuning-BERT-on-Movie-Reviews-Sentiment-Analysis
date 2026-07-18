"""PyTorch Dataset / DataLoader helpers."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EncodedDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def make_loader(
    texts: List[str],
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    ds = EncodedDataset(texts, labels, tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )
