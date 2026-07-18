"""Dataset loaders for Paper 2 reproduction.

IMDb-2 uses the official ACL IMDb train/test split (25k / 25k).
"""

from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TextClassificationSplit:
    texts: List[str]
    labels: np.ndarray  # int labels
    name: str = ""


@dataclass
class TextClassificationDataset:
    train: TextClassificationSplit
    test: TextClassificationSplit
    label_names: List[str]


_RATING_RE = re.compile(r"(\d+)_(\d+)\.txt$")
_IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reproduce_paper2" / "datasets"


def _read_imdb_dir(directory: Path) -> pd.DataFrame:
    """Read pos/neg folders; keep star rating from filename when present."""
    rows = []
    for sentiment_name, polarity in (("pos", 1), ("neg", 0)):
        folder = directory / sentiment_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing IMDb folder: {folder}")
        for file_name in os.listdir(folder):
            path = folder / file_name
            text = path.read_text(encoding="utf-8", errors="ignore")
            rating = None
            match = _RATING_RE.match(file_name)
            if match:
                rating = int(match.group(2))
            rows.append(
                {
                    "text": text,
                    "label": polarity,
                    "rating": rating,
                    "split_dir": str(directory),
                }
            )
    return pd.DataFrame(rows)


def download_acl_imdb(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Download ACL IMDb and return path to extracted ``aclImdb`` root."""
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "aclImdb_v1.tar.gz"
    extract_root = cache_dir
    imdb_root = extract_root / "aclImdb"

    if imdb_root.is_dir() and not force_download:
        return imdb_root

    if force_download or not archive_path.is_file():
        print(f"Downloading IMDb to {archive_path} ...")
        urllib.request.urlretrieve(_IMDB_URL, archive_path)

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_root)

    if not imdb_root.is_dir():
        # Some extract layouts nest differently
        for candidate in extract_root.rglob("aclImdb"):
            if candidate.is_dir() and (candidate / "train").is_dir():
                return candidate
        raise FileNotFoundError(f"Could not locate aclImdb under {extract_root}")
    return imdb_root


def load_imdb2(
    force_download: bool = False,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> TextClassificationDataset:
    """Official IMDb binary split used as IMDb-2 in Paper 2."""
    root = download_acl_imdb(cache_dir=cache_dir, force_download=force_download)
    train_df = _read_imdb_dir(root / "train")
    test_df = _read_imdb_dir(root / "test")

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if limit_train is not None:
        train_df = train_df.iloc[:limit_train].reset_index(drop=True)
    if limit_test is not None:
        test_df = test_df.iloc[:limit_test].reset_index(drop=True)

    label_names = ["negative", "positive"]
    train = TextClassificationSplit(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].to_numpy(dtype=np.int64),
        name="imdb2_train",
    )
    test = TextClassificationSplit(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].to_numpy(dtype=np.int64),
        name="imdb2_test",
    )
    return TextClassificationDataset(train=train, test=test, label_names=label_names)


def train_val_split(
    split: TextClassificationSplit,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[TextClassificationSplit, TextClassificationSplit]:
    """Shuffle-then-cut split of the official train set (test stays official)."""
    n = len(split.texts)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def _take(idxs: np.ndarray, name: str) -> TextClassificationSplit:
        idxs = np.sort(idxs)
        return TextClassificationSplit(
            texts=[split.texts[i] for i in idxs],
            labels=split.labels[idxs],
            name=name,
        )

    return _take(train_idx, split.name + "_tr"), _take(val_idx, split.name + "_val")
