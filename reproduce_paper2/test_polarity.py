"""Unit checks for overall-polarity helpers (no GPU / no downloads)."""

from reproduce_paper2.polarity import (
    overall_polarity_binary,
    overall_polarity_four_class,
    overall_polarity_three_class,
    overall_polarity_five_class,
    compare_original_vs_computed,
    int_labels_to_names,
)


def test_binary_balanced_is_neutral():
    labels = [0] * 50 + [1] * 50
    assert overall_polarity_binary(labels) == "neutral"


def test_binary_positive_dominates():
    labels = [1] * 60 + [0] * 40
    assert overall_polarity_binary(labels) == "positive"


def test_binary_negative_dominates():
    labels = [0] * 60 + [1] * 40
    assert overall_polarity_binary(labels) == "negative"


def test_three_class_with_names():
    names = ["negative", "neutral", "positive"]
    labels = [1] * 90 + [2] * 5 + [0] * 5  # mostly neutral
    assert overall_polarity_three_class(labels, label_names=names) == "neutral"


def test_four_class_highly_positive():
    names = ["highly_negative", "negative", "positive", "highly_positive"]
    # many HP + P
    labels = [3] * 40 + [2] * 10 + [1] * 10 + [0] * 5
    assert overall_polarity_four_class(labels, label_names=names) == "highly_positive"


def test_five_class_neutral_gate():
    names = [
        "highly_negative",
        "negative",
        "neutral",
        "positive",
        "highly_positive",
    ]
    labels = [2] * 90 + [3] * 5 + [0] * 5
    assert overall_polarity_five_class(labels, label_names=names) == "neutral"


def test_compare_match():
    gold = [0, 0, 1, 1]
    pred = [0, 0, 1, 1]
    out = compare_original_vs_computed(gold, pred, mode="binary")
    assert out["match"] == "True"
    assert out["original_op"] == out["computed_op"]


def test_int_labels_to_names():
    names = ["negative", "neutral", "positive"]
    assert int_labels_to_names([0, 1, 2], names) == names


if __name__ == "__main__":
    test_binary_balanced_is_neutral()
    test_binary_positive_dominates()
    test_binary_negative_dominates()
    test_three_class_with_names()
    test_four_class_highly_positive()
    test_five_class_neutral_gate()
    test_compare_match()
    test_int_labels_to_names()
    print("All polarity tests passed.")
