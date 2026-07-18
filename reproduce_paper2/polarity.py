"""Overall polarity heuristics from Paper 2 (Algorithms 1–4).

Binary path (Algorithm 2) is used by the IMDb-2 reproduction entrypoint.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Mapping, Sequence, Union

import numpy as np


LabelLike = Union[int, str]


# Canonical names used in reports
POS = "positive"
NEG = "negative"
NEU = "neutral"
H_POS = "highly_positive"
H_NEG = "highly_negative"


def _as_label_list(labels: Iterable[LabelLike]) -> List[str]:
    out: List[str] = []
    for lab in labels:
        if isinstance(lab, (int, np.integer)):
            # binary default: 0=neg, 1=pos
            out.append(POS if int(lab) == 1 else NEG)
        else:
            out.append(str(lab).lower().replace(" ", "_"))
    return out


def overall_polarity_binary(
    labels: Iterable[LabelLike],
    majority_coef: float = 1.2,
) -> str:
    """Algorithm 2 (Paper 2): overall polarity from binary predictions.

    if #pos > coef * #neg → positive
    elif #neg > coef * #pos → negative
    else → neutral
    """
    labs = _as_label_list(labels)
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
) -> str:
    """Algorithm 1 (Paper 2): three-class overall polarity."""
    labs = _as_label_list(labels)
    n = len(labs) or 1
    counts = Counter(labs)
    neu = counts.get(NEU, 0) + counts.get("2", 0)
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
) -> str:
    """Algorithm 3 (Paper 2): hierarchical four-class overall polarity."""
    labs = _as_label_list(labels)
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
) -> str:
    """Algorithm 4 (Paper 2): five-class overall polarity."""
    labs = _as_label_list(labels)
    n = len(labs) or 1
    neu = sum(1 for x in labs if x == NEU)
    if neu > neutral_ratio * n:
        return NEU
    # drop neutrals for hierarchical comparison (paper keeps counts as-is
    # on full vector; hierarchical step still uses fine labels)
    return overall_polarity_four_class(labs, super_coef=super_coef, sub_coef=sub_coef)


def compare_original_vs_computed(
    gold: Sequence[LabelLike],
    pred: Sequence[LabelLike],
    mode: str = "binary",
) -> Mapping[str, str]:
    """Mirror Paper 2 Table VI style comparison."""
    fn = {
        "binary": overall_polarity_binary,
        "three": overall_polarity_three_class,
        "four": overall_polarity_four_class,
        "five": overall_polarity_five_class,
    }[mode]
    return {
        "original_op": fn(gold),
        "computed_op": fn(pred),
        "match": str(fn(gold) == fn(pred)),
    }
