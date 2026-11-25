import pytest
import torch
from src.ml.class_weights import compute_class_weights

def test_weights_by_hand():
    # 4 segments of type 0 and 1 of type 1, so 5 segments and 2 road types:
    # type 0: 5 / (2 * 4) = 0.625, type 1: 5 / (2 * 1) = 2.5
    # and then the square root of those, so the rare types don't get too much weight
    weights = compute_class_weights(torch.tensor([0, 0, 0, 0, 1]), num_classes=2)
    assert weights.tolist() == pytest.approx([0.625 ** 0.5, 2.5 ** 0.5])

def test_rare_type_gets_bigger_weight():
    weights = compute_class_weights(torch.tensor([0, 0, 0, 1]), num_classes=2)
    assert weights[1] > weights[0]

def test_equal_types_get_weight_one():
    weights = compute_class_weights(torch.tensor([0, 1, 2, 0, 1, 2]), num_classes=3)
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])

def test_missing_type_gets_weight_zero():
    # Road type 1 has no training segments, so it gets weight 0 instead of infinity
    weights = compute_class_weights(torch.tensor([0, 0, 2]), num_classes=3)
    assert weights[1] == 0
    assert torch.isfinite(weights).all()

def test_power_half_gives_square_root():
    # The same labels as test_weights_by_hand: the full weights are 0.625 and 2.5, so power 0.5 gives their roots
    weights = compute_class_weights(torch.tensor([0, 0, 0, 0, 1]), num_classes=2, power=0.5)
    assert weights.tolist() == pytest.approx([0.625 ** 0.5, 2.5 ** 0.5])


def test_power_zero_gives_no_weighting():
    weights = compute_class_weights(torch.tensor([0, 0, 0, 1]), num_classes=2, power=0)
    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_missing_type_stays_zero_with_any_power():
    # 0 ** 0 is 1, so the power must only be applied to road types that occur
    for power in (0, 0.5, 1):
        weights = compute_class_weights(torch.tensor([0, 0, 2]), num_classes=3, power=power)
        assert weights[1] == 0
