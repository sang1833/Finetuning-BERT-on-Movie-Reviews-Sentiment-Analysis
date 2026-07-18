"""Hyperparameters aligned with Paper 2 experimental settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal


ModelInputMode = Literal["cls", "sequence"]
TaskMode = Literal["binary", "fine_grained"]


@dataclass
class TrainConfig:
    """Training / model configuration.

    Binary settings follow Paper 2 Section IV-C (binary classification).
    Fine-grained settings follow the same section for multi-class runs.
    """

    # Task
    task: TaskMode = "binary"
    num_labels: int = 2
    dataset: str = "imdb2"

    # Backbone
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    # Paper 2 text: only [CLS] (C) is fed to BiLSTM.
    # Use "sequence" to match the original repo (full token sequence → BiLSTM).
    bilstm_input: ModelInputMode = "cls"
    bilstm_units: int = 768
    dropout: float = 0.5
    freeze_bert: bool = True

    # Optim / schedule (binary defaults from Paper 2)
    learning_rate: float = 3e-5
    epsilon: float = 1e-8
    decay: float = 0.0
    batch_size: int = 32
    epochs: int = 10
    # Hold out a slice of official train for early monitoring (test stays official).
    val_ratio: float = 0.1

    # Runtime
    seed: int = 42
    output_dir: str = "outputs/imdb2_binary"
    limit_train_samples: int | None = None  # for smoke tests
    limit_test_samples: int | None = None

    def apply_task_defaults(self) -> "TrainConfig":
        """Fill hyperparams from Paper 2 for the selected task mode."""
        if self.task == "binary":
            self.max_length = 128
            self.learning_rate = 3e-5
            self.epsilon = 1e-8
            self.decay = 0.0
            self.batch_size = 32
            self.epochs = 10
        else:
            # Fine-grained (3/4/5-class) — ready for later stages
            self.max_length = 256
            self.learning_rate = 1e-4
            self.epsilon = 1e-7
            self.decay = 1e-5
            self.batch_size = 64
            self.epochs = 15
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_binary_imdb_config(**overrides: Any) -> TrainConfig:
    cfg = TrainConfig(task="binary", num_labels=2, dataset="imdb2")
    cfg.apply_task_defaults()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config field: {key}")
        setattr(cfg, key, value)
    return cfg
