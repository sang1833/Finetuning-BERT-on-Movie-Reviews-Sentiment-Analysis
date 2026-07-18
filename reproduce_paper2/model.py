"""BERT + BiLSTM classifier (PyTorch) for Paper 2 reproduction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from .config import TrainConfig


class BertBiLSTMClassifier(nn.Module):
    """BERT_BASE + BiLSTM + classification head.

    Paper 2: feed [CLS] vector C into BiLSTM, then dense.
    ``bilstm_input='sequence'`` uses full token sequence (legacy repo style).
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        hidden = self.bert.config.hidden_size  # 768 for base

        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(config.dropout)
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=config.bilstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        bilstm_out = config.bilstm_units * 2
        self.head_dropout = nn.Dropout(config.dropout)

        if config.task == "binary" and config.num_labels == 2:
            # Binary CE with logits (more stable than sigmoid+BCE in training loop)
            self.classifier = nn.Linear(bilstm_out, 1)
            self.binary_sigmoid_head = True
        else:
            self.classifier = nn.Linear(bilstm_out, config.num_labels)
            self.binary_sigmoid_head = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        if self.config.bilstm_input == "cls":
            bilstm_in = sequence_output[:, 0:1, :]  # [B, 1, H]
        else:
            bilstm_in = sequence_output

        bilstm_in = self.dropout(bilstm_in)
        lstm_out, _ = self.bilstm(bilstm_in)  # [B, T, 2H]
        # last time step
        pooled = lstm_out[:, -1, :]
        pooled = self.head_dropout(pooled)
        logits = self.classifier(pooled)
        if self.binary_sigmoid_head:
            return logits.squeeze(-1)  # [B]
        return logits  # [B, C]


def build_bert_bilstm(config: TrainConfig) -> BertBiLSTMClassifier:
    return BertBiLSTMClassifier(config)
