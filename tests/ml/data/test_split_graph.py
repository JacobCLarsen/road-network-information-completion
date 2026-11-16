# pytest tests

import random

import pytest
import torch

from src.ml.data import split_graph
from src.ml.data.split_graph import make_split_masks
from src.ml.config import TRAIN_SHARE, VAL_SHARE

STRATEGIES = ["random", "spatial"]
SPLITS = ["train", "validation", "test"]

def _graph_from(y_list, geometry):
    """Minimal graph dictionary: only the parts of get_static_components() that the split reads."""
    y = torch.tensor(y_list, dtype=torch.long)
    return {"y": y, "mask": y >= 0, "geometry": geometry}


@pytest.fixture
def graph():
    """
    600 segments with unbalanced classes and 64 unlabeled segments (y = -1), in random order and at random
    positions. Class sizes: 300, 150, 60, 20, 5 and 1, to include both big and rare classes.
    """
    rng = random.Random(0)
    y_list = [0] * 300 + [1] * 150 + [2] * 60 + [3] * 20 + [4] * 5 + [5] * 1 + [-1] * 64
    rng.shuffle(y_list)
    geometry = []
    for _ in y_list:
        lon, lat = rng.uniform(116.0, 117.0), rng.uniform(39.5, 40.5)
        geometry.append([(lon, lat), (lon + 0.001, lat + 0.001)])
    return _graph_from(y_list, geometry)


@pytest.fixture
def grid_graph():
    """
    800 labeled segments placed in 16 groups of 50. Every group sits on one exact spot, chosen so that with
    cells_per_side=4 each group falls in its own grid cell. Used to test that whole cells stay together.
    Returns the graph and the list of index tensors, one per group.
    """
    positions = [0.5, 1.5, 2.5, 3.5]
    y_list, geometry, groups = [], [], []
    for lon in positions:
        for lat in positions:
            start = len(y_list)
            for k in range(50):
                y_list.append(k % 3)
                geometry.append([(lon, lat), (lon, lat)])
            groups.append(torch.arange(start, len(y_list)))
    return _graph_from(y_list, geometry), groups


def _call(graph, strategy, seed=42, **kwargs):
    return make_split_masks(graph, seed=seed, strategy=strategy, train_share=TRAIN_SHARE, val_share=VAL_SHARE, **kwargs)


# Property tests: must hold for every split, both strategies

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_masks_have_expected_keys_shape_and_type(graph, strategy):
    masks = _call(graph, strategy)
    assert set(masks) == set(SPLITS)
    for name in SPLITS:
        assert masks[name].dtype == torch.bool
        assert masks[name].shape == graph["y"].shape


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_splits_are_mutually_exclusive(graph, strategy):
    masks = _call(graph, strategy)
    overlap_count = masks["train"].int() + masks["validation"].int() + masks["test"].int()
    assert overlap_count.max() <= 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_splits_cover_exactly_the_labeled_segments(graph, strategy):
    masks = _call(graph, strategy)
    union = masks["train"] | masks["validation"] | masks["test"]
    assert torch.equal(union, graph["mask"])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unlabeled_segments_are_in_no_split(graph, strategy):
    masks = _call(graph, strategy)
    unlabeled = graph["y"] == -1
    for name in SPLITS:
        assert not (masks[name] & unlabeled).any()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_split_is_empty(graph, strategy):
    masks = _call(graph, strategy)
    for name in SPLITS:
        assert masks[name].sum() > 0


# Reproducibility

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_same_seed_gives_same_split(graph, strategy):
    first = _call(graph, strategy, seed=7)
    second = _call(graph, strategy, seed=7)
    for name in SPLITS:
        assert torch.equal(first[name], second[name])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_different_seeds_give_different_splits(graph, strategy):
    base = _call(graph, strategy, seed=0)
    others = [_call(graph, strategy, seed=s) for s in range(1, 6)]
    # A single other seed could coincide by chance in theory, so five are checked
    assert any(not torch.equal(base["train"], other["train"]) for other in others)


# Random strategy: shares and stratification

def test_random_split_sizes_match_the_shares(graph):
    masks = _call(graph, "random")
    total = int(graph["mask"].sum())
    assert abs(int(masks["train"].sum()) - TRAIN_SHARE * total) <= 6   # at most about one segment of rounding per class
    assert abs(int(masks["validation"].sum()) - VAL_SHARE * total) <= 6


def test_random_split_is_stratified_per_class(graph):
    masks = _call(graph, "random")
    for c in [0, 1, 2, 3]:  # the classes with enough segments for the shares to be meaningful
        n = int((graph["y"] == c).sum())
        n_train = int((masks["train"] & (graph["y"] == c)).sum())
        n_val = int((masks["validation"] & (graph["y"] == c)).sum())
        assert abs(n_train - TRAIN_SHARE * n) <= 1
        assert abs(n_val - VAL_SHARE * n) <= 1


def test_random_split_keeps_every_class_in_training(graph):
    masks = _call(graph, "random")
    for c in range(6):  # includes class 5, which has a single segment
        assert (masks["train"] & (graph["y"] == c)).any(), f"class {c} is missing from training"

# Spatial strategy: whole cells stay together

def test_spatial_split_keeps_each_grid_cell_in_one_split(grid_graph):
    graph, groups = grid_graph
    masks = _call(graph, "spatial", cells_per_side=4)
    for idx in groups:
        counts = sorted(int(masks[name][idx].sum()) for name in SPLITS)
        assert counts == [0, 0, len(idx)], "a grid cell was divided between splits"


def test_spatial_split_sizes_are_close_to_the_shares(grid_graph):
    graph, _ = grid_graph
    masks = _call(graph, "spatial", cells_per_side=4)
    total = int(graph["mask"].sum())
    # Whole cells move together (one cell is 6.25 percent here), so the shares are only approximate
    assert abs(int(masks["train"].sum()) / total - TRAIN_SHARE) <= 0.15
    assert abs(int(masks["validation"].sum()) / total - VAL_SHARE) <= 0.15
    assert abs(int(masks["test"].sum()) / total - (1 - TRAIN_SHARE - VAL_SHARE)) <= 0.15


# Configuration: defaults come from the config, arguments override them

def test_shares_default_to_the_config_values(graph):
    masks = make_split_masks(graph, seed=42, strategy="random")
    total = int(graph["mask"].sum())
    assert abs(int(masks["train"].sum()) - split_graph.TRAIN_SHARE * total) <= 6
    assert abs(int(masks["validation"].sum()) - split_graph.VAL_SHARE * total) <= 6


def test_test_share_is_the_remainder(graph):
    masks = make_split_masks(graph, seed=42, strategy="random", train_share=0.6, val_share=0.2)
    total = int(graph["mask"].sum())
    assert abs(int(masks["train"].sum()) - 0.6 * total) <= 6
    assert abs(int(masks["validation"].sum()) - 0.2 * total) <= 6
    assert abs(int(masks["test"].sum()) - 0.2 * total) <= 6


def test_strategy_defaults_to_the_config_value(graph):
    default = make_split_masks(graph, seed=42, train_share=TRAIN_SHARE, val_share=VAL_SHARE)
    explicit = _call(graph, split_graph.STRATEGY)
    for name in SPLITS:
        assert torch.equal(default[name], explicit[name])


def test_strategy_argument_overrides_the_config(graph):
    random_masks = _call(graph, "random")
    spatial_masks = _call(graph, "spatial")
    assert any(not torch.equal(random_masks[name], spatial_masks[name]) for name in SPLITS)


# Error handling

def test_unknown_strategy_is_rejected(graph):
    with pytest.raises(ValueError):
        make_split_masks(graph, strategy="spacial", train_share=TRAIN_SHARE, val_share=VAL_SHARE)


@pytest.mark.parametrize("train_share, val_share", [(-0.1, 0.2), (0.5, -0.1), (0.8, 0.3)])
def test_invalid_shares_are_rejected(graph, train_share, val_share):
    with pytest.raises(ValueError):
        make_split_masks(graph, strategy="random", train_share=train_share, val_share=val_share)


# Edge cases

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_graph_without_any_labels_gives_empty_masks(strategy):
    graph = _graph_from([-1, -1, -1], [[(0.0, 0.0), (0.1, 0.1)]] * 3)
    masks = _call(graph, strategy)
    for name in SPLITS:
        assert masks[name].shape == (3,)
        assert not masks[name].any()


def test_class_with_one_segment_goes_to_training():
    y_list = [0] * 20 + [1]
    graph = _graph_from(y_list, [[(0.0, 0.0), (0.1, 0.1)]] * len(y_list))
    masks = _call(graph, "random")
    assert masks["train"][20]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_graph_is_not_modified(graph, strategy):
    y_before, mask_before = graph["y"].clone(), graph["mask"].clone()
    _call(graph, strategy)
    assert torch.equal(graph["y"], y_before)
    assert torch.equal(graph["mask"], mask_before)


def test_accepts_an_object_with_get_static_components(graph):
    class FakeBuilder:
        def get_static_components(self):
            return graph

    from_object = _call(FakeBuilder(), "random")
    from_dict = _call(graph, "random")
    for name in SPLITS:
        assert torch.equal(from_object[name], from_dict[name])


# Integration test with the real graph builder

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_works_with_the_output_of_roadgraphbuilder(strategy):
    from src.ml.data.road_graph_builder import RoadGraphBuilder, Segment

    # A chain of 60 segments, where segment i runs from junction i to junction i + 1. Every sixth one has a
    # road type that is not in ROAD_CLASSES, so it is unlabeled.
    types = ["residential", "tertiary", "primary"]
    segments = []
    for i in range(60):
        highway = "footway" if i % 6 == 5 else types[i % 3]
        points = [(116.0 + i * 0.001, 39.9), (116.0 + (i + 1) * 0.001, 39.9)]
        segments.append(Segment([i], i, i + 1, points, highway=highway, oneway="False"))

    graph = RoadGraphBuilder(segments).get_static_components()
    masks = _call(graph, strategy)

    union = masks["train"] | masks["validation"] | masks["test"]
    assert masks["train"].shape[0] == 60
    assert torch.equal(union, graph["mask"])
    assert int(graph["mask"].sum()) == 50