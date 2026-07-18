"""Overall polarity heuristics from Paper 2 (Algorithms 1–4)."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np


LabelLike = Union[int, str]

POS = "positive"
NEG = "negative"
NEU = "neutral"
H_POS = "highly_positive"
H_NEG = "highly_negative"


def int_labels_to_names(
    labels: Iterable[LabelLike],
    label_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Map integer class ids to canonical string names."""
    out: List[str] = []
    for lab in labels:
        if isinstance(lab, str):
            out.append(lab.lower().replace(" ", "_").replace("-", "_"))
            continue
        idx = int(lab)
        if label_names is not None:
            out.append(label_names[idx])
        else:
            # binary fallback
            out.append(POS if idx == 1 else NEG)
    return out


def _as_label_list(
    labels: Iterable[LabelLike],
    label_names: Optional[Sequence[str]] = None,
) -> List[str]:
    return int_labels_to_names(labels, label_names)


def overall_polarity_binary(
    labels: Iterable[LabelLike],
    majority_coef: float = 1.2,
    label_names: Optional[Sequence[str]] = None,
) -> str:
    """Algorithm 2: binary overall polarity."""
    labs = _as_label_list(labels, label_names)
    pos = sum(1 for x in labs if x in {POS, "1", "pos"})
    neg = sum(1 for x in labs if x in {NEG, "0", "neg"})
    if pos > majority_coef * neg:
        return POS
    if neg > majority_coef * pos:
        return NEG
    return NEU


def overall_polarity_three_class(
    labels: Iterable[LabelLike],
    neutral_ratio: float = 0.85,
    majority_coef: float = 1.5,
    label_names: Optional[Sequence[str]] = None,
) -> str:
    """Algorithm 1: three-class overall polarity."""
    labs = _as_label_list(labels, label_names)
    n = len(labs) or 1
    counts = Counter(labs)
    neu = counts.get(NEU, 0)
    pos = counts.get(POS, 0)
    neg = counts.get(NEG, 0)
    if neu > neutral_ratio * n:
        return NEU
    if pos > majority_coef * neg:
        return POS
    if neg > majority_coef * pos:
        return NEG
    return NEU


def overall_polarity_four_class(
    labels: Iterable[LabelLike],
    super_coef: float = 1.2,
    sub_coef: float = 1.5,
    label_names: Optional[Sequence[str]] = None,
) -> str:
    """Algorithm 3: hierarchical four-class overall polarity."""
    labs = _as_label_list(labels, label_names)
    c = Counter(labs)
    hneg = c.get(H_NEG, 0)
    neg = c.get(NEG, 0)
    pos = c.get(POS, 0)
    hpos = c.get(H_POS, 0)
    neg_super = hneg + neg
    pos_super = hpos + pos
    if neg_super > super_coef * pos_super:
        return H_NEG if hneg > sub_coef * neg else NEG
    if pos_super > super_coef * neg_super:
        return H_POS if hpos > sub_coef * pos else POS
    return NEU


def overall_polarity_five_class(
    labels: Iterable[LabelLike],
    neutral_ratio: float = 0.85,
    super_coef: float = 1.2,
    sub_coef: float = 1.5,
    label_names: Optional[Sequence[str]] = None,
) -> str:
    """Algorithm 4: five-class overall polarity."""
    labs = _as_label_list(labels, label_names)
    n = len(labs) or 1
    neu = sum(1 for x in labs if x == NEU)
    if neu > neutral_ratio * n:
        return NEU
    return overall_polarity_four_class(
        labs, super_coef=super_coef, sub_coef=sub_coef, label_names=None
    )


def compare_original_vs_computed(
    gold: Sequence[LabelLike],
    pred: Sequence[LabelLike],
    mode: str = "binary",
    label_names: Optional[Sequence[str]] = None,
) -> Mapping[str, str]:
    """Mirror Paper 2 Table VI style comparison."""
    fn_map = {
        "binary": overall_polarity_binary,
        "three": overall_polarity_three_class,
        "four": overall_polarity_four_class,
        "five": overall_polarity_five_class,
    }
    if mode not in fn_map:
        raise ValueError(f"Unknown OP mode: {mode}")
    fn = fn_map[mode]
    orig = fn(gold, label_names=label_names)
    comp = fn(pred, label_names=label_names)
    return {
        "original_op": orig,
        "computed_op": comp,
        "match": str(orig == comp),
        "mode": mode,
    }
