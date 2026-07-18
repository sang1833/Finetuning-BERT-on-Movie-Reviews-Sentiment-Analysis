"""Hyperparameters and dataset registry aligned with Paper 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional


ModelInputMode = Literal["cls", "sequence"]
TaskMode = Literal["binary", "fine_grained"]
AugmentMode = Literal["none", "nlpaug", "smote"]


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a Paper 2 experiment dataset."""

    key: str
    num_labels: int
    task: TaskMode
    op_mode: str  # binary | three | four | five
    label_names: List[str]
    paper_accuracy_pct: Optional[float]
    paper_op: Optional[str]  # Table VI expected overall polarity
    description: str


# Paper 2 label conventions
BINARY_LABELS = ["negative", "positive"]
THREE_LABELS = ["negative", "neutral", "positive"]
FOUR_LABELS = ["highly_negative", "negative", "positive", "highly_positive"]
FIVE_LABELS = [
    "highly_negative",
    "negative",
    "neutral",
    "positive",
    "highly_positive",
]

# Paper 2 Table I — train/test counts per label (H_POS, POS, NEU, NEG, H_NEG).
# Binary rows use only POS/NEG; fine-grained fill more columns.
# Values copied from the paper table for comparison scripts.
PAPER_TABLE_I: Dict[str, Dict[str, Dict[str, int]]] = {
    "imdb2": {
        "train": {"positive": 12500, "negative": 12500},
        "test": {"positive": 12500, "negative": 12500},
    },
    "mr": {
        "train": {"positive": 4264, "negative": 4265},
        "test": {"positive": 1067, "negative": 1066},
    },
    "sst2": {
        "train": {"positive": 4300, "negative": 4244},
        "test": {"positive": 886, "negative": 1116},
    },
    "amazon2": {
        "train": {"positive": 239660, "negative": 37056},
        "test": {"positive": 59949, "negative": 9231},
    },
    "imdb3": {
        "train": {"positive": 18227, "neutral": 4816, "negative": 14958},
        "test": {"positive": 4556, "neutral": 1204, "negative": 3739},
    },
    "imdb4": {
        "train": {
            "highly_positive": 11471,
            "positive": 8530,
            "negative": 8234,
            "highly_negative": 11767,
        },
        "test": {
            "highly_positive": 2867,
            "positive": 2132,
            "negative": 2058,
            "highly_negative": 2941,
        },
    },
    "sst5": {
        "train": {
            "highly_positive": 1482,
            "positive": 2489,
            "neutral": 1794,
            "negative": 2512,
            "highly_negative": 1208,
        },
        "test": {
            "highly_positive": 370,
            "positive": 622,
            "neutral": 448,
            "negative": 628,
            "highly_negative": 302,
        },
    },
    "amazon5": {
        "train": {
            "highly_positive": 182000,
            "positive": 57688,
            "neutral": 27767,
            "negative": 15168,
            "highly_negative": 21863,
        },
        "test": {
            "highly_positive": 45500,
            "positive": 14421,
            "neutral": 6941,
            "negative": 3791,
            "highly_negative": 5465,
        },
    },
}

# Paper Table II–V SOTA rows (for export side-by-side; not re-trained).
PAPER_TABLE_II_BASELINES: Dict[str, Dict[str, Optional[float]]] = {
    "RNN-Capsule": {"imdb2": 84.12, "mr": 83.80, "sst2": 82.77, "amazon2": 82.68},
    "coRNN": {"imdb2": 87.4, "mr": 87.11, "sst2": 88.97, "amazon2": 89.32},
    "TL-CNN": {"imdb2": 87.70, "mr": 81.5, "sst2": 87.70, "amazon2": 88.12},
    "Modified LMU": {"imdb2": 93.20, "mr": 93.15, "sst2": 93.10, "amazon2": 93.67},
    "DualCL": {"imdb2": None, "mr": 94.31, "sst2": 94.91, "amazon2": 94.98},
    "L Mixed": {"imdb2": 95.68, "mr": 95.72, "sst2": None, "amazon2": 95.81},
    "EFL": {"imdb2": 96.10, "mr": 96.90, "sst2": 96.90, "amazon2": 96.91},
    "NB-weighted-BON+dv-cosine": {
        "imdb2": 97.40,
        "mr": None,
        "sst2": 96.55,
        "amazon2": 97.55,
    },
    "SMART-RoBERTa Large": {"imdb2": 96.34, "mr": 97.5, "sst2": 96.61, "amazon2": None},
    "Ours (paper)": {"imdb2": 97.67, "mr": 97.88, "sst2": 97.62, "amazon2": 98.76},
}

PAPER_TABLE_III_BASELINES: Dict[str, Dict[str, Optional[float]]] = {
    "CNN-RNF-LSTM": {"imdb3": 73.71, "imdb4": 63.78},
    "DPCNN": {"imdb3": 76.24, "imdb4": 66.17},
    "BERT-large": {"imdb3": 77.21, "imdb4": 66.87},
    "Ours (paper)": {"imdb3": 81.87, "imdb4": 70.75},
}

PAPER_TABLE_IV_BASELINES: Dict[str, Dict[str, Optional[float]]] = {
    "CNN+word2vec": {"sst5": 46.4, "amazon5": 48.85},
    "TL-CNN": {"sst5": 47.2, "amazon5": 58.1},
    "DRNN": {"sst5": None, "amazon5": 64.43},
    "BERT-large": {"sst5": 55.5, "amazon5": 65.83},
    "BCN+Suffix+BiLSTM-Tied+Cove": {"sst5": 56.2, "amazon5": 65.92},
    "RoBERTa+large+Self-explaining": {"sst5": 59.10, "amazon5": None},
    "Ours (paper)": {"sst5": 60.48, "amazon5": 69.68},
}

PAPER_TABLE_V: Dict[str, float] = {
    "BERT+BiLSTM": 58.44,
    "BERT+BiLSTM+SMOTE": 58.36,
    "BERT+BiLSTM+NLPAUG": 60.48,
}

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "imdb2": DatasetSpec(
        key="imdb2",
        num_labels=2,
        task="binary",
        op_mode="binary",
        label_names=BINARY_LABELS,
        paper_accuracy_pct=97.67,
        paper_op="neutral",
        description="IMDb binary (official 25k/25k)",
    ),
    "mr": DatasetSpec(
        key="mr",
        num_labels=2,
        task="binary",
        op_mode="binary",
        label_names=BINARY_LABELS,
        paper_accuracy_pct=97.88,
        paper_op="neutral",
        description="MR movie reviews binary (rotten_tomatoes / Pang&Lee-style)",
    ),
    "sst2": DatasetSpec(
        key="sst2",
        num_labels=2,
        task="binary",
        op_mode="binary",
        label_names=BINARY_LABELS,
        paper_accuracy_pct=97.62,
        paper_op="neutral",
        description="Stanford Sentiment Treebank binary (GLUE SST-2)",
    ),
    "amazon2": DatasetSpec(
        key="amazon2",
        num_labels=2,
        task="binary",
        op_mode="binary",
        label_names=BINARY_LABELS,
        paper_accuracy_pct=98.76,
        paper_op="positive",
        description="Amazon video reviews binary (stars 1-2 vs 4-5)",
    ),
    "imdb3": DatasetSpec(
        key="imdb3",
        num_labels=3,
        task="fine_grained",
        op_mode="three",
        label_names=THREE_LABELS,
        paper_accuracy_pct=81.87,
        paper_op="neutral",
        description="IMDb 3-class from rating scores",
    ),
    "imdb4": DatasetSpec(
        key="imdb4",
        num_labels=4,
        task="fine_grained",
        op_mode="four",
        label_names=FOUR_LABELS,
        paper_accuracy_pct=70.75,
        paper_op="neutral",
        description="IMDb 4-class binary-tree split from ratings",
    ),
    "sst5": DatasetSpec(
        key="sst5",
        num_labels=5,
        task="fine_grained",
        op_mode="five",
        label_names=FIVE_LABELS,
        paper_accuracy_pct=60.48,
        paper_op="neutral",
        description="SST-5 fine-grained (NLPAUG target in paper)",
    ),
    "amazon5": DatasetSpec(
        key="amazon5",
        num_labels=5,
        task="fine_grained",
        op_mode="five",
        label_names=FIVE_LABELS,
        paper_accuracy_pct=69.68,
        paper_op="positive",
        description="Amazon video reviews 5-star fine-grained",
    ),
}


@dataclass
class TrainConfig:
    """Training / model configuration (Paper 2 Section IV-C)."""

    task: TaskMode = "binary"
    num_labels: int = 2
    dataset: str = "imdb2"
    op_mode: str = "binary"
    label_names: List[str] | None = None

    model_name: str = "bert-base-uncased"
    max_length: int = 128
    bilstm_input: ModelInputMode = "cls"
    bilstm_units: int = 768
    dropout: float = 0.5
    freeze_bert: bool = True

    learning_rate: float = 3e-5
    epsilon: float = 1e-8
    decay: float = 0.0
    batch_size: int = 32
    epochs: int = 10
    val_ratio: float = 0.1

    seed: int = 42
    output_dir: str = "outputs/imdb2_binary"
    limit_train_samples: int | None = None
    limit_test_samples: int | None = None

    # Phase B: augmentation for fine-grained (esp. SST-5)
    augment: AugmentMode = "none"
    nlpaug_factor: float = 0.5  # fraction of minority-class samples to augment
    paper_accuracy_pct: float | None = None
    paper_op: str | None = None

    def apply_task_defaults(self) -> "TrainConfig":
        if self.task == "binary":
            self.max_length = 128
            self.learning_rate = 3e-5
            self.epsilon = 1e-8
            self.decay = 0.0
            self.batch_size = 32
            self.epochs = 10
        else:
            self.max_length = 256
            self.learning_rate = 1e-4
            self.epsilon = 1e-7
            self.decay = 1e-5
            self.batch_size = 64
            self.epochs = 15
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_for_dataset(dataset_key: str, **overrides: Any) -> TrainConfig:
    """Build a TrainConfig for a registered Paper 2 dataset."""
    if dataset_key not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {known}")

    spec = DATASET_SPECS[dataset_key]
    cfg = TrainConfig(
        task=spec.task,
        num_labels=spec.num_labels,
        dataset=spec.key,
        op_mode=spec.op_mode,
        label_names=list(spec.label_names),
        output_dir=f"outputs/{spec.key}",
        paper_accuracy_pct=spec.paper_accuracy_pct,
        paper_op=spec.paper_op,
    )
    cfg.apply_task_defaults()

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config field: {key}")
        setattr(cfg, key, value)
    return cfg


# Backward-compatible alias
def default_binary_imdb_config(**overrides: Any) -> TrainConfig:
    return config_for_dataset("imdb2", **overrides)
