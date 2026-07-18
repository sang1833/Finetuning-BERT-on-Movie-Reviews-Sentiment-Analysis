"""Dataset loaders for Paper 2 Phase B (binary + fine-grained)."""

from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BINARY_LABELS,
    FIVE_LABELS,
    FOUR_LABELS,
    THREE_LABELS,
)


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


def _limit_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    limit_train: Optional[int],
    limit_test: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit_train is not None:
        train_df = train_df.iloc[:limit_train].reset_index(drop=True)
    if limit_test is not None:
        test_df = test_df.iloc[:limit_test].reset_index(drop=True)
    return train_df, test_df


def _df_to_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: List[str],
    name: str,
) -> TextClassificationDataset:
    return TextClassificationDataset(
        train=TextClassificationSplit(
            texts=train_df["text"].astype(str).tolist(),
            labels=train_df["label"].to_numpy(dtype=np.int64),
            name=f"{name}_train",
        ),
        test=TextClassificationSplit(
            texts=test_df["text"].astype(str).tolist(),
            labels=test_df["label"].to_numpy(dtype=np.int64),
            name=f"{name}_test",
        ),
        label_names=label_names,
    )


# ---------------------------------------------------------------------------
# IMDb
# ---------------------------------------------------------------------------


def _read_imdb_dir(directory: Path) -> pd.DataFrame:
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
                    "binary": polarity,
                    "rating": rating,
                }
            )
    return pd.DataFrame(rows)


def download_acl_imdb(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "aclImdb_v1.tar.gz"
    imdb_root = cache_dir / "aclImdb"

    if imdb_root.is_dir() and not force_download:
        return imdb_root

    if force_download or not archive_path.is_file():
        print(f"Downloading IMDb to {archive_path} ...")
        urllib.request.urlretrieve(_IMDB_URL, archive_path)

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    if not imdb_root.is_dir():
        for candidate in cache_dir.rglob("aclImdb"):
            if candidate.is_dir() and (candidate / "train").is_dir():
                return candidate
        raise FileNotFoundError(f"Could not locate aclImdb under {cache_dir}")
    return imdb_root


def _map_imdb3(rating: int) -> Optional[int]:
    """Paper 2: 1–3 neg, 4&7 neutral, 8–10 pos (scores 5–6 absent)."""
    if rating in (1, 2, 3):
        return 0  # negative
    if rating in (4, 7):
        return 1  # neutral
    if rating in (8, 9, 10):
        return 2  # positive
    return None


def _map_imdb4(rating: int) -> Optional[int]:
    """Paper 2 binary-tree: 1–2 HN, 3–4 N, 7–8 P, 9–10 HP."""
    if rating in (1, 2):
        return 0
    if rating in (3, 4):
        return 1
    if rating in (7, 8):
        return 2
    if rating in (9, 10):
        return 3
    return None


def load_imdb2(
    force_download: bool = False,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> TextClassificationDataset:
    root = download_acl_imdb(cache_dir=cache_dir, force_download=force_download)
    train_df = _read_imdb_dir(root / "train")
    test_df = _read_imdb_dir(root / "test")
    train_df["label"] = train_df["binary"]
    test_df["label"] = test_df["binary"]
    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, BINARY_LABELS, "imdb2")


def load_imdb3(
    force_download: bool = False,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> TextClassificationDataset:
    root = download_acl_imdb(cache_dir=cache_dir, force_download=force_download)
    train_df = _read_imdb_dir(root / "train")
    test_df = _read_imdb_dir(root / "test")

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["rating"]).copy()
        df["label"] = df["rating"].map(_map_imdb3)
        return df.dropna(subset=["label"]).astype({"label": int})

    train_df, test_df = _prep(train_df), _prep(test_df)
    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, THREE_LABELS, "imdb3")


def load_imdb4(
    force_download: bool = False,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> TextClassificationDataset:
    root = download_acl_imdb(cache_dir=cache_dir, force_download=force_download)
    train_df = _read_imdb_dir(root / "train")
    test_df = _read_imdb_dir(root / "test")

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["rating"]).copy()
        df["label"] = df["rating"].map(_map_imdb4)
        return df.dropna(subset=["label"]).astype({"label": int})

    train_df, test_df = _prep(train_df), _prep(test_df)
    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, FOUR_LABELS, "imdb4")


# ---------------------------------------------------------------------------
# SST / MR / Amazon via HuggingFace datasets
# ---------------------------------------------------------------------------


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Package 'datasets' is required for SST/MR/Amazon loaders. "
            "Install with: pip install datasets"
        ) from exc
    return load_dataset


def load_sst2(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> TextClassificationDataset:
    """SST-2 binary. Tries several HF ids (glue path broke on newer hub clients)."""
    load_dataset = _require_datasets()
    errors: List[str] = []
    train_df = test_df = None

    candidates = (
        # (repo, config, train_split, eval_split, text_key, label_key)
        ("nyu-mll/glue", "sst2", "train", "validation", "sentence", "label"),
        ("SetFit/sst2", None, "train", "test", "text", "label"),
        ("gpt3mix/sst2", None, "train", "test", "text", "label"),
        ("glue", "sst2", "train", "validation", "sentence", "label"),
    )
    for repo, config, tr, ev, text_key, label_key in candidates:
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
            eval_split = ev if ev in ds else ("validation" if "validation" in ds else "test")
            train_df = pd.DataFrame(
                {"text": ds[tr][text_key], "label": ds[tr][label_key]}
            )
            test_df = pd.DataFrame(
                {"text": ds[eval_split][text_key], "label": ds[eval_split][label_key]}
            )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}: {exc}")
            train_df = test_df = None

    if train_df is None or test_df is None:
        raise RuntimeError("Could not load SST-2. Errors:\n" + "\n".join(errors))

    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, BINARY_LABELS, "sst2")


def load_sst5(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> TextClassificationDataset:
    """SST-5 fine-grained. Tries SetFit/sst5 then falls back to glue-style sources."""
    load_dataset = _require_datasets()
    train_df = test_df = None
    errors: List[str] = []

    for name, train_split, test_split, text_key, label_key in (
        ("SetFit/sst5", "train", "test", "text", "label"),
        ("sst", "train", "test", "sentence", "label"),
    ):
        try:
            if name == "sst":
                ds = load_dataset("sst", "default")
                # continuous labels in [0,1] → 5 bins
                def _to_5(y: float) -> int:
                    # map [0,1] → {0,1,2,3,4}
                    return min(4, int(float(y) * 5)) if float(y) < 1.0 else 4

                train_df = pd.DataFrame(
                    {
                        "text": ds[train_split]["sentence"],
                        "label": [_to_5(y) for y in ds[train_split]["label"]],
                    }
                )
                # sst has validation+test
                test_parts = []
                for sp in ("validation", "test"):
                    if sp in ds:
                        test_parts.append(
                            pd.DataFrame(
                                {
                                    "text": ds[sp]["sentence"],
                                    "label": [_to_5(y) for y in ds[sp]["label"]],
                                }
                            )
                        )
                test_df = pd.concat(test_parts, ignore_index=True)
            else:
                ds = load_dataset(name)
                train_df = pd.DataFrame(
                    {
                        "text": ds[train_split][text_key],
                        "label": ds[train_split][label_key],
                    }
                )
                test_df = pd.DataFrame(
                    {
                        "text": ds[test_split][text_key],
                        "label": ds[test_split][label_key],
                    }
                )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            train_df = test_df = None

    if train_df is None or test_df is None:
        raise RuntimeError(
            "Could not load SST-5. Tried SetFit/sst5 and sst. Errors:\n"
            + "\n".join(errors)
        )

    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)
    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, FIVE_LABELS, "sst5")


def load_mr(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    paper_like_split: bool = True,
) -> TextClassificationDataset:
    """MR-style binary via HuggingFace (rotten_tomatoes / cornell variants).

    Paper Table I uses ~8529 train / ~2133 test (≈80/20) balanced binary.
    When ``paper_like_split`` is True we merge available RT splits and re-split
    stratified 80/20 to approximate that protocol (counts may still differ from
    the classic 5331+5331 Pang & Lee subset).
    """
    load_dataset = _require_datasets()
    errors: List[str] = []
    train_df = test_df = None

    for repo in ("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes"):
        try:
            ds = load_dataset(repo)
            frames = []
            for split in ds:
                frames.append(
                    pd.DataFrame(
                        {"text": ds[split]["text"], "label": ds[split]["label"]}
                    )
                )
            full = pd.concat(frames, ignore_index=True)
            if paper_like_split:
                train_df, test_df = stratified_train_test_split_df(
                    full, test_ratio=test_ratio, seed=seed
                )
            else:
                train_df = pd.DataFrame(
                    {"text": ds["train"]["text"], "label": ds["train"]["label"]}
                )
                test_frames = []
                for split in ("validation", "test"):
                    if split in ds:
                        test_frames.append(
                            pd.DataFrame(
                                {
                                    "text": ds[split]["text"],
                                    "label": ds[split]["label"],
                                }
                            )
                        )
                test_df = (
                    pd.concat(test_frames, ignore_index=True)
                    if test_frames
                    else full.iloc[:0]
                )
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{repo}: {exc}")
            train_df = test_df = None

    if train_df is None or test_df is None:
        raise RuntimeError("Could not load MR. Errors:\n" + "\n".join(errors))

    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)
    return _df_to_dataset(train_df, test_df, BINARY_LABELS, "mr")


def _load_amazon_video(
    binary: bool,
    limit_train: Optional[int],
    limit_test: Optional[int],
    seed: int,
) -> TextClassificationDataset:
    """Amazon video reviews from amazon_us_reviews Video_v1_00 when available."""
    load_dataset = _require_datasets()
    errors: List[str] = []
    ds = None
    for path in (
        ("amazon_us_reviews", "Video_v1_00"),
        ("amazon_polarity", None),
    ):
        try:
            if path[1] is None:
                ds = load_dataset(path[0])
                # amazon_polarity: already binary, content + label
                full = pd.DataFrame(
                    {
                        "text": ds["train"]["content"],
                        "label": ds["train"]["label"],
                    }
                )
                if not binary:
                    raise ValueError("amazon_polarity is binary-only; need Video stars for amazon5")
            else:
                ds = load_dataset(path[0], path[1])
                stars = ds["train"]["star_rating"]
                bodies = ds["train"]["review_body"]
                rows = []
                for text, star in zip(bodies, stars):
                    star = int(star)
                    if binary:
                        if star <= 2:
                            rows.append({"text": text, "label": 0})
                        elif star >= 4:
                            rows.append({"text": text, "label": 1})
                        # drop neutral star=3
                    else:
                        # 1..5 → 0..4
                        rows.append({"text": text, "label": star - 1})
                full = pd.DataFrame(rows)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
            ds = None
            full = None  # type: ignore

    if full is None or len(full) == 0:
        raise RuntimeError(
            "Could not load Amazon reviews. Tried amazon_us_reviews Video_v1_00 "
            "and amazon_polarity (binary only). Errors:\n" + "\n".join(errors)
        )

    full = full.dropna(subset=["text"]).reset_index(drop=True)
    full["text"] = full["text"].astype(str)
    full = full[full["text"].str.len() > 0].reset_index(drop=True)

    # Stratified ~20% test — paper Table I is ~20% holdout on Amazon video
    train_df, test_df = stratified_train_test_split_df(
        full, test_ratio=0.2, seed=seed, label_col="label"
    )
    train_df, test_df = _limit_df(train_df, test_df, limit_train, limit_test, seed)

    labels = BINARY_LABELS if binary else FIVE_LABELS
    name = "amazon2" if binary else "amazon5"
    return _df_to_dataset(train_df, test_df, labels, name)


def load_amazon2(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> TextClassificationDataset:
    return _load_amazon_video(True, limit_train, limit_test, seed)


def load_amazon5(
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> TextClassificationDataset:
    return _load_amazon_video(False, limit_train, limit_test, seed)


# ---------------------------------------------------------------------------
# Registry + split helpers
# ---------------------------------------------------------------------------

LOADERS: Dict[str, Callable[..., TextClassificationDataset]] = {
    "imdb2": load_imdb2,
    "imdb3": load_imdb3,
    "imdb4": load_imdb4,
    "sst2": load_sst2,
    "sst5": load_sst5,
    "mr": load_mr,
    "amazon2": load_amazon2,
    "amazon5": load_amazon5,
}


def load_dataset_by_key(
    key: str,
    limit_train: Optional[int] = None,
    limit_test: Optional[int] = None,
    seed: int = 42,
) -> TextClassificationDataset:
    if key not in LOADERS:
        known = ", ".join(sorted(LOADERS))
        raise ValueError(f"Unknown dataset '{key}'. Choose from: {known}")
    return LOADERS[key](
        limit_train=limit_train,
        limit_test=limit_test,
        seed=seed,
    )


def train_val_split(
    split: TextClassificationSplit,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[TextClassificationSplit, TextClassificationSplit]:
    """Hold out a validation slice from train (test stays held out).

    Uses stratified sampling by default so class ratios match the train set.
    """
    n = len(split.texts)
    if n <= 1:
        return split, TextClassificationSplit(texts=[], labels=np.array([], dtype=np.int64), name=split.name + "_val")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    if stratified and len(np.unique(split.labels)) > 1:
        val_idx_list: List[int] = []
        train_idx_list: List[int] = []
        for lab in np.unique(split.labels):
            lab_idx = indices[split.labels == lab]
            rng.shuffle(lab_idx)
            n_val_lab = max(1, int(round(len(lab_idx) * val_ratio))) if len(lab_idx) > 1 else 0
            # leave at least one train sample when possible
            if n_val_lab >= len(lab_idx) and len(lab_idx) > 1:
                n_val_lab = len(lab_idx) - 1
            val_idx_list.extend(lab_idx[:n_val_lab].tolist())
            train_idx_list.extend(lab_idx[n_val_lab:].tolist())
        val_idx = np.array(val_idx_list, dtype=np.int64)
        train_idx = np.array(train_idx_list, dtype=np.int64)
    else:
        rng.shuffle(indices)
        n_val = max(1, int(round(n * val_ratio)))
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


def count_by_label(
    labels: np.ndarray,
    label_names: List[str],
) -> Dict[str, int]:
    counts = {name: 0 for name in label_names}
    for y in labels.tolist():
        if 0 <= int(y) < len(label_names):
            counts[label_names[int(y)]] += 1
    return counts


def dataset_label_stats(dataset: TextClassificationDataset) -> Dict[str, Dict[str, int]]:
    return {
        "train": count_by_label(dataset.train.labels, dataset.label_names),
        "test": count_by_label(dataset.test.labels, dataset.label_names),
    }


def stratified_train_test_split_df(
    df: pd.DataFrame,
    test_ratio: float,
    seed: int,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of a single dataframe."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for lab, group in df.groupby(label_col):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = max(1, int(round(len(group) * test_ratio))) if len(group) > 1 else 0
        if n_test >= len(group) and len(group) > 1:
            n_test = len(group) - 1
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
