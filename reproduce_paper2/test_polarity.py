"""Unit checks for overall-polarity helpers (no GPU / no downloads)."""

from reproduce_paper2.polarity import (
    overall_polarity_binary,
    overall_polarity_four_class,
    overall_polarity_three_class,
    compare_original_vs_computed,
)


def test_binary_balanced_is_neutral():
    # 50/50 → neutral under coef 1.2
    labels = [0] * 50 + [1] * 50
    assert overall_polarity_binary(labels) == "neutral"


def test_binary_positive_dominates():
    labels = [1] * 60 + [0] * 40  # 60 > 1.2*40=48
    assert overall_polarity_binary(labels) == "positive"


def test_binary_negative_dominates():
    labels = [0] * 60 + [1] * 40
    assert overall_polarity_binary(labels) == "negative"


def test_three_class_neutral_threshold():
    labels = ["neutral"] * 90 + ["positive"] * 5 + ["negative"] * 5
    assert overall_polarity_three_class(labels) == "neutral"


def test_four_class_highly_positive():
    labels = ["highly_positive"] * 40 + ["positive"] * 10 + ["negative"] * 10 + ["highly_negative"] * 5
    assert overall_polarity_four_class(labels) == "highly_positive"


def test_compare_match():
    gold = [0, 0, 1, 1]
    pred = [0, 0, 1, 1]
    out = compare_original_vs_computed(gold, pred, mode="binary")
    assert out["match"] == "True"
    assert out["original_op"] == out["computed_op"]


if __name__ == "__main__":
    test_binary_balanced_is_neutral()
    test_binary_positive_dominates()
    test_binary_negative_dominates()
    test_three_class_neutral_threshold()
    test_four_class_highly_positive()
    test_compare_match()
    print("All polarity tests passed.")
