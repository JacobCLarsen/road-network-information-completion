import json
from pathlib import Path

import pytest
import torch

from src.ml.data.split_graph import (
    _check_masks,
    _grid_cell_ids,
    _new_masks,
    _segment_centroids,
    _split_counts,
    make_split_masks,
)

# Regression tests for make_split_masks.
#
# Every model is trained and scored on the split this function returns, so a code change that moves even one segment
# to another split makes new results impossible to compare with old ones. The snapshot file holds the exact segment
# indices of each split for a fixed synthetic graph, for several settings. The tests fail if any of them change.
#
# The snapshot was made with torch 2.14.1. torch.randperm is only guaranteed to give the same numbers for the same torch version, so after a torch upgrade
# these tests may fail even though the code did not change. If the change in the split is intended, make a new
# snapshot with:  python -m tests.ml.data.test_split_graph

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "split_graph_snapshot.json"

NUM_SEGMENTS = 80
NUM_ROAD_TYPES = 4

# The settings the snapshot covers: name -> keyword arguments for make_split_masks.
# All arguments are given explicitly, so a change in the config does not break the tests.
CASES = {
    "random_stratified": dict(seed=42, strategy="random", stratify=True, train_share=0.6, val_share=0.2),
    "random_not_stratified": dict(seed=42, strategy="random", stratify=False, train_share=0.6, val_share=0.2),
    "random_other_seed": dict(seed=1, strategy="random", stratify=True, train_share=0.6, val_share=0.2),
    "random_no_test": dict(seed=42, strategy="random", stratify=True, train_share=0.7, val_share=0.3),
    "random_uneven_shares": dict(seed=3, strategy="random", stratify=True, train_share=0.5, val_share=0.35),
    "spatial_default_grid": dict(seed=42, strategy="spatial", cells_per_side=20, train_share=0.6, val_share=0.2),
    "spatial_small_grid": dict(seed=42, strategy="spatial", cells_per_side=5, train_share=0.6, val_share=0.2),
    "spatial_other_seed": dict(seed=7, strategy="spatial", cells_per_side=8, train_share=0.5, val_share=0.25),
    "spatial_no_validation": dict(seed=42, strategy="spatial", cells_per_side=6, train_share=0.8, val_share=0.0),
}


def build_graph():
    """
    A fixed synthetic graph made with plain arithmetic (no random generator), so it is the same on every machine.
    Every 7th segment has no label. Segments have 2 or 3 points spread over a small area.
    """
    y = []
    mask = []
    geometry = []
    for i in range(NUM_SEGMENTS):
        # Uneven road types, so the stratified split has groups of different sizes
        y.append((i * i + 2 * i + i // 5) % NUM_ROAD_TYPES)
        mask.append(i % 7 != 3)

        lon = 9.9 + ((i * 37) % 101) / 1000
        lat = 57.0 + ((i * 53) % 97) / 1000
        points = [[lon, lat], [lon + 0.0004, lat + 0.0002]]
        if i % 3 == 0:
            points.append([lon + 0.0008, lat - 0.0003])
        geometry.append(points)

    return {
        "y": torch.tensor(y, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "geometry": geometry,
    }


def masks_to_indices(masks):
    """Turns the masks into lists of segment indices, which are easy to store and to compare."""
    return {name: mask.nonzero().flatten().tolist() for name, mask in masks.items()}


def load_snapshot():
    with open(SNAPSHOT_PATH, encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def graph():
    return build_graph()


def test_snapshot_covers_every_case():
    assert sorted(load_snapshot()) == sorted(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_make_split_masks_matches_snapshot(graph, case):
    """The split for each setting must be exactly the stored one, segment for segment."""
    # Arrange
    expected = load_snapshot()[case]

    # Act
    result = masks_to_indices(make_split_masks(graph, **CASES[case]))

    # Assert
    for name in ("train", "validation", "test"):
        assert result[name] == expected[name], f"{case}: the {name} split has changed"


@pytest.mark.parametrize("case", sorted(CASES))
def test_make_split_masks_same_result_twice(graph, case):
    first = masks_to_indices(make_split_masks(graph, **CASES[case]))
    second = masks_to_indices(make_split_masks(graph, **CASES[case]))
    assert first == second


@pytest.mark.parametrize("case", sorted(CASES))
def test_make_split_masks_covers_labeled_segments_once(graph, case):
    masks = make_split_masks(graph, **CASES[case])
    times_used = masks["train"].int() + masks["validation"].int() + masks["test"].int()
    assert torch.equal(times_used, graph["mask"].int())


def test_make_split_masks_accepts_graph_builder(graph):
    """A RoadGraphBuilder (anything with get_static_components) gives the same split as its dictionary."""
    class FakeBuilder:
        def get_static_components(self):
            return graph

    kwargs = CASES["random_stratified"]
    assert masks_to_indices(make_split_masks(FakeBuilder(), **kwargs)) == masks_to_indices(
        make_split_masks(graph, **kwargs))


def test_random_stratified_keeps_road_type_shares(graph):
    """Every road type is split on its own, so train gets about 60% of each road type."""
    masks = make_split_masks(graph, **CASES["random_stratified"])
    labeled = graph["mask"]
    for road_type in range(NUM_ROAD_TYPES):
        of_type = labeled & (graph["y"] == road_type)
        share = (masks["train"] & of_type).sum().item() / of_type.sum().item()
        assert share == pytest.approx(0.6, abs=0.1)


def test_spatial_split_keeps_grid_cells_together(graph):
    """Segments with the same centroid lie in the same cell and must end up in the same split."""
    graph["geometry"][1] = graph["geometry"][0]
    graph["geometry"][2] = graph["geometry"][0]
    masks = make_split_masks(graph, **CASES["spatial_default_grid"])
    for name in ("train", "validation", "test"):
        assert masks[name][0] == masks[name][1] == masks[name][2]


def test_spatial_split_with_all_segments_in_one_place(graph):
    """All segments at the same position must not give a division by zero; they all go to one split."""
    graph["geometry"] = [[[10.0, 57.0]] for _ in range(NUM_SEGMENTS)]
    masks = make_split_masks(graph, **CASES["spatial_default_grid"])
    assert torch.equal(masks["train"], graph["mask"])


@pytest.mark.parametrize("strategy", ["random", "spatial"])
def test_make_split_masks_with_no_labels(graph, strategy):
    graph["mask"] = torch.zeros(NUM_SEGMENTS, dtype=torch.bool)
    masks = make_split_masks(graph, strategy=strategy)
    for mask in masks.values():
        assert not mask.any()


@pytest.mark.parametrize("train_share,val_share", [(-0.1, 0.2), (0.6, -0.1), (0.8, 0.3)])
def test_make_split_masks_raises_error_for_invalid_shares(graph, train_share, val_share):
    with pytest.raises(ValueError, match="Invalid shares"):
        make_split_masks(graph, train_share=train_share, val_share=val_share)


def test_make_split_masks_raises_error_for_unknown_strategy(graph):
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_split_masks(graph, strategy="alphabetical")


@pytest.mark.parametrize("count,train_share,val_share,expected", [
    (10, 0.6, 0.2, (6, 2, 2)),
    (0, 0.6, 0.2, (0, 0, 0)),
    # Train always gets at least one item
    (1, 0.1, 0.5, (1, 0, 0)),
    # 3 * 0.6 = 1.8 rounds to 2, 3 * 0.2 = 0.6 rounds to 1, so nothing is left for test
    (3, 0.6, 0.2, (2, 1, 0)),
    (5, 1.0, 0.0, (5, 0, 0)),
])
def test_split_counts_by_hand(count, train_share, val_share, expected):
    assert _split_counts(count, train_share, val_share) == expected


def test_segment_centroids_by_hand():
    # The centroid is the mean of the points: (0 + 2) / 2 = 1 and (0 + 4) / 2 = 2. A single point is its own centroid.
    geometry = [[[0.0, 0.0], [2.0, 4.0]], [[1.0, 1.0]], [[0.0, 0.0], [3.0, 0.0], [0.0, 6.0]]]
    centroids = _segment_centroids(geometry)
    assert centroids.dtype == torch.float64
    assert centroids.tolist() == [[1.0, 2.0], [1.0, 1.0], [1.0, 2.0]]


def test_grid_cell_ids_by_hand():
    # A 2 x 2 grid over the box from (0, 0) to (1, 1). The cell number is column * 2 + row.
    # (0, 0) is in column 0, row 0 -> 0. (1, 1) lies on the far edge, so it goes to the last cell -> 3.
    # (0.5, 0.25) is in column 1, row 0 -> 2. (0.25, 0.75) is in column 0, row 1 -> 1.
    centroids = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25], [0.25, 0.75]], dtype=torch.float64)
    assert _grid_cell_ids(centroids, cells_per_side=2).tolist() == [0, 3, 2, 1]


def test_grid_cell_ids_with_all_segments_in_one_place():
    centroids = torch.tensor([[10.0, 57.0], [10.0, 57.0]], dtype=torch.float64)
    assert _grid_cell_ids(centroids, cells_per_side=4).tolist() == [0, 0]


def test_check_masks_raises_error_for_overlap():
    masks = _new_masks(3)
    masks["train"][0] = True
    masks["test"][0] = True
    with pytest.raises(AssertionError, match="more than one split"):
        _check_masks(masks, torch.tensor([True, False, False]))


def test_check_masks_raises_error_for_missing_segment():
    masks = _new_masks(3)
    masks["train"][0] = True
    with pytest.raises(AssertionError, match="cover exactly"):
        _check_masks(masks, torch.tensor([True, True, False]))


def write_snapshot():
    """Makes a new snapshot from the current code. Only run this when a change in the split is intended."""
    graph = build_graph()
    snapshot = {case: masks_to_indices(make_split_masks(graph, **kwargs)) for case, kwargs in CASES.items()}
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)
        file.write("\n")
    print(f"Wrote {SNAPSHOT_PATH}")


if __name__ == "__main__":
    write_snapshot()
