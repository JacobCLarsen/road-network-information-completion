import math
import pytest
import torch
from src.ml.data.road_graph_builder import (
    LINK_ROAD_MAPPINGS,
    RARE_ROAD_TYPES,
    ROAD_CLASSES,
    RoadGraphBuilder,
    Segment,
    compute_features,
    compute_raw_features,
    fit_standardizer,
    haversine,
    mean_and_std,
    standardize,
)

A, B, C = (10.000, 55.000), (10.001, 55.000), (10.002, 55.000)
BEND, D, E = (10.001, 54.999), (10.002, 54.999), (10.002, 54.998)

# The degree of each segment in the segments fixture
DEGREES = [2, 2, 3, 1]


@pytest.fixture
def segments():
    """
    A small network of four segments between junctions 1, 2, 3, 5, 6:

        1 ---- 2 ---- 3        segments 0 and 1 (primary, oneway)
               |
               + -- 5          segment 2 (residential, two-way), bends on the way to 5
                    |
                    6          segment 3 (no highway tag)
    """
    return [
        Segment(edge_ids=[10], u=1, v=2, points=[A, B], highway="primary", oneway="True"),
        Segment(edge_ids=[11], u=2, v=3, points=[B, C], highway="primary", oneway="True"),
        Segment(edge_ids=[20, 21], u=2, v=5, points=[B, BEND, D], highway="residential", oneway="False"),
        Segment(edge_ids=[30], u=5, v=6, points=[D, E], highway=None, oneway=None),
    ]


def test_keys_and_geometry(segments):
    builder = RoadGraphBuilder(segments)

    assert builder.seg_key == [(10, 1, 2), (11, 2, 3), (20, 2, 5), (30, 5, 6)]
    assert builder.geometry[2] == [B, BEND, D]


def test_both_directions_map_to_the_same_segment(segments):
    builder = RoadGraphBuilder(segments)
    assert builder.edge_to_segment == {10: 0, 11: 1, 20: 2, 21: 2, 30: 3}


def test_labels_and_mask(segments):
    builder = RoadGraphBuilder(segments)

    primary, residential = ROAD_CLASSES.index("primary"), ROAD_CLASSES.index("residential")
    assert builder.y.tolist() == [primary, primary, residential, -1]
    assert builder.mask.tolist() == [True, True, True, False]


def test_multiple_road_types_are_unlabelled():
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway="['residential', 'tertiary']")
    builder = RoadGraphBuilder([seg])
    assert builder.y.tolist() == [-1]
    assert builder.mask.tolist() == [False]


def test_features(segments):
    builder = RoadGraphBuilder(segments)
    assert builder.x.shape == (4, 4)

    # Straight segment: no curvature, oneway
    straight = builder.x[0]
    assert straight[0].item() == pytest.approx(math.log(haversine(A, B)), rel=1e-5)
    assert straight[1].item() == pytest.approx(0.0, abs=1e-5)
    assert straight[2].item() == 1.0

    # Bent segment B -> BEND -> D: curvature is 1 - straight / length
    length = haversine(B, BEND) + haversine(BEND, D)
    bent = builder.x[2]
    assert bent[0].item() == pytest.approx(math.log(length), rel=1e-5)
    assert bent[1].item() == pytest.approx(1 - haversine(B, D) / length, rel=1e-4)
    assert bent[2].item() == 0.0


def test_line_graph_edges(segments):
    builder = RoadGraphBuilder(segments)

    # Segments 0, 1, 2 meet at junction 2; segments 2 and 3 meet at junction 5
    edges = set(map(tuple, builder.edge_index.t().tolist()))
    assert edges == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 3), (3, 2)}


def test_connection_count_feature(segments):
    builder = RoadGraphBuilder(segments)
    assert builder.x[:, 3].tolist() == [2, 2, 3, 1]


def test_parallel_segments_are_connected_once():
    segs = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B]),
        Segment(edge_ids=[2], u=1, v=2, points=[A, BEND, B]),
    ]
    builder = RoadGraphBuilder(segs)
    assert builder.edge_index.tolist() == [[0, 1], [1, 0]]


def test_loop_segment_is_not_connected_to_itself():
    loop = Segment(edge_ids=[1], u=1, v=1, points=[A, B, BEND, A], highway="service")
    builder = RoadGraphBuilder([loop])

    assert builder.edge_index.shape == (2, 0)
    assert builder.x[0, 1].item() == pytest.approx(1.0)
    assert builder.x[0, 3] == 0


def test_zero_length_segment_has_finite_features():
    builder = RoadGraphBuilder([Segment(edge_ids=[1], u=1, v=2, points=[A, A], highway="road")])
    assert torch.isfinite(builder.x).all()


def test_straight_segment_has_curvature_in_range():
    builder = RoadGraphBuilder([Segment(edge_ids=[1], u=1, v=2, points=[A, B, C], highway="road")])
    assert 0.0 <= builder.x[0, 1].item() <= 1.0


def test_duplicate_edge_id_raises():
    duplicates = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway="road"),
        Segment(edge_ids=[1], u=2, v=3, points=[B, C], highway="road"),
    ]
    with pytest.raises(AssertionError, match="Edge ID 1"):
        RoadGraphBuilder(duplicates)


def test_get_static_components_keys(segments):
    components = RoadGraphBuilder(segments).get_static_components()
    assert set(components.keys()) == {'x', 'y', 'mask', 'edge_index', 'seg_key', 'geometry', 'edge_to_segment'}


def test_mean_and_std_known_values():
    mean, std = mean_and_std([2, 4, 4, 4, 5, 5, 7, 9])

    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(2.0)


def test_mean_and_std_identical_values_use_std_of_one():
    mean, std = mean_and_std([3.0, 3.0, 3.0])

    assert mean == pytest.approx(3.0)
    assert std == 1.0


def test_mean_and_std_single_value_uses_std_of_one():
    assert mean_and_std([7.0]) == (7.0, 1.0)


@pytest.mark.parametrize("x,mean,std,expected", [
    (5.0, 5.0, 2.0, 0.0),
    (7.0, 5.0, 2.0, 1.0),
    (1.0, 5.0, 2.0, -2.0),
    (3.0, 3.0, 1.0, 0.0),
])
def test_standardize(x, mean, std, expected):
    assert standardize(x, mean, std) == pytest.approx(expected)


def test_compute_raw_features_are_unscaled(segments):
    log_length, curvature, one_way = compute_raw_features(segments[0])

    assert log_length == pytest.approx(math.log(haversine(A, B)))
    assert curvature == pytest.approx(0.0, abs=1e-9)
    assert one_way == 1


def test_fit_standardizer_matches_raw_features(segments):
    raw = [compute_raw_features(segment) for segment in segments]

    length_mean, length_std, curv_mean, curv_std, degree_mean, degree_std = fit_standardizer(segments, DEGREES)

    assert (length_mean, length_std) == pytest.approx(mean_and_std([row[0] for row in raw]))
    assert (curv_mean, curv_std) == pytest.approx(mean_and_std([row[1] for row in raw]))
    assert (degree_mean, degree_std) == pytest.approx(mean_and_std(DEGREES))


def test_standardized_training_features_have_mean_zero_and_std_one(segments):
    stats = fit_standardizer(segments, DEGREES)

    rows = [compute_features(segment, degree, *stats) for segment, degree in zip(segments, DEGREES)]

    for column in (0, 1, 3):
        z_values = [row[column] for row in rows]
        assert sum(z_values) / len(z_values) == pytest.approx(0.0, abs=1e-9)
        assert math.sqrt(sum(z ** 2 for z in z_values) / len(z_values)) == pytest.approx(1.0)


def test_compute_features_applies_z_score_to_length_curvature_and_degree(segments):
    log_length, curvature, _ = compute_raw_features(segments[2])

    z_length, z_curvature, _, z_degree = compute_features(segments[2], 3, 1.0, 2.0, 0.5, 0.25, 2.0, 4.0)

    assert z_length == pytest.approx((log_length - 1.0) / 2.0)
    assert z_curvature == pytest.approx((curvature - 0.5) / 0.25)
    assert z_degree == pytest.approx((3 - 2.0) / 4.0)


@pytest.mark.parametrize("index,expected", [(0, 1), (2, 0)])
def test_compute_features_does_not_standardize_one_way(segments, index, expected):
    stats = fit_standardizer(segments, DEGREES)

    assert compute_features(segments[index], DEGREES[index], *stats)[2] == expected


def test_compute_features_keeps_identical_training_values_finite():
    identical = [Segment(edge_ids=[i], u=i, v=i + 1, points=[A, B]) for i in range(3)]
    stats = fit_standardizer(identical, [1, 1, 1])

    row = compute_features(identical[0], 1, *stats)

    assert [row[0], row[1], row[3]] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_standardize_features_uses_stats_from_training_segments_only(segments):
    builder = RoadGraphBuilder(segments)
    train_mask = torch.tensor([True, False, True, False])
    stats = fit_standardizer([segments[0], segments[2]], [DEGREES[0], DEGREES[2]])

    builder.standardize_features(segments, train_mask)

    # Every segment is scaled, also the ones outside the training set
    expected = [compute_features(segment, degree, *stats) for segment, degree in zip(segments, DEGREES)]
    assert torch.equal(builder.x, torch.tensor(expected, dtype=torch.float))


def test_standardize_features_is_not_affected_by_non_training_segments(segments):
    train_mask = torch.tensor([True, True, True, False])
    changed = list(segments)
    changed[3] = Segment(edge_ids=[30], u=5, v=6, points=[D, (10.5, 54.5)], highway=None, oneway=None)
    builder, changed_builder = RoadGraphBuilder(segments), RoadGraphBuilder(changed)

    builder.standardize_features(segments, train_mask)
    changed_builder.standardize_features(changed, train_mask)

    assert torch.equal(builder.x[train_mask], changed_builder.x[train_mask])


def test_standardize_features_keeps_one_way_binary(segments):
    builder = RoadGraphBuilder(segments)

    builder.standardize_features(segments, torch.tensor([True, True, True, True]))

    assert builder.x[:, 2].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_standardize_features_updates_static_components(segments):
    builder = RoadGraphBuilder(segments)
    graph = builder.get_static_components()

    builder.standardize_features(segments, torch.tensor([True, True, True, True]))

    assert graph["x"] is builder.x
    assert graph["x"][:, [0, 1, 3]].mean(dim=0).tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)


def test_standardize_features_twice_gives_the_same_result(segments):
    builder = RoadGraphBuilder(segments)
    train_mask = torch.tensor([True, True, False, False])

    builder.standardize_features(segments, train_mask)
    once = builder.x.clone()
    builder.standardize_features(segments, train_mask)

    assert torch.equal(once, builder.x)


@pytest.mark.parametrize("link,parent", LINK_ROAD_MAPPINGS.items())
def test_link_road_gets_the_label_of_its_parent(link, parent):
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway=link)

    builder = RoadGraphBuilder([seg])

    assert builder.y.tolist() == [ROAD_CLASSES.index(parent)]
    assert builder.mask.tolist() == [True]


@pytest.mark.parametrize("road_type", RARE_ROAD_TYPES)
def test_rare_road_type_is_removed_from_training(road_type):
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway=road_type)

    builder = RoadGraphBuilder([seg])

    assert builder.y.tolist() == [-1]
    assert builder.mask.tolist() == [False]


@pytest.mark.parametrize("road_type", [road_type for road_type in ROAD_CLASSES
                                       if road_type not in RARE_ROAD_TYPES and road_type not in LINK_ROAD_MAPPINGS])
def test_regular_road_type_keeps_its_own_label(road_type):
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway=road_type)

    builder = RoadGraphBuilder([seg])

    assert builder.y.tolist() == [ROAD_CLASSES.index(road_type)]
    assert builder.mask.tolist() == [True]


def test_rare_road_segment_stays_in_the_graph_as_a_neighbour():
    segs = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway="primary"),
        Segment(edge_ids=[2], u=2, v=3, points=[B, C], highway="service"),
    ]

    builder = RoadGraphBuilder(segs)

    assert builder.x.shape[0] == 2
    assert builder.edge_index.tolist() == [[0, 1], [1, 0]]
    assert builder.mask.tolist() == [True, False]


def test_link_road_is_unlabelled_when_its_parent_is_not_in_classes():
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], highway="primary_link")

    builder = RoadGraphBuilder([seg], classes=["primary_link", "residential"])

    assert builder.y.tolist() == [-1]
    assert builder.mask.tolist() == [False]


def test_link_parents_are_kept_classes():
    for parent in LINK_ROAD_MAPPINGS.values():
        assert parent in ROAD_CLASSES
        assert parent not in RARE_ROAD_TYPES


def test_road_classes_have_no_link_or_rare_types():
    assert not set(ROAD_CLASSES) & set(LINK_ROAD_MAPPINGS)
    assert not set(ROAD_CLASSES) & set(RARE_ROAD_TYPES)
