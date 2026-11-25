import math
import pytest
import torch
from src.ml.data.road_graph_builder import (
    ROAD_CLASSES,
    RoadGraphBuilder,
    Segment,
    compute_features,
    compute_raw_features,
    compute_raw_traffic_features,
    compute_traffic_features,
    fit_traffic_standardizer,
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
                    6          segment 3 (no label)
    """
    return [
        Segment(edge_ids=[10], u=1, v=2, points=[A, B], label="primary", oneway="True"),
        Segment(edge_ids=[11], u=2, v=3, points=[B, C], label="primary", oneway="True"),
        Segment(edge_ids=[20, 21], u=2, v=5, points=[B, BEND, D], label="residential", oneway="False"),
        Segment(edge_ids=[30], u=5, v=6, points=[D, E], label=None, oneway=None),
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


def test_features(segments):
    builder = RoadGraphBuilder(segments)
    assert builder.x.shape == (4, 10)

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
    loop = Segment(edge_ids=[1], u=1, v=1, points=[A, B, BEND, A], label=None)
    builder = RoadGraphBuilder([loop])

    assert builder.edge_index.shape == (2, 0)
    assert builder.x[0, 1].item() == pytest.approx(1.0)
    assert builder.x[0, 3] == 0


def test_zero_length_segment_has_finite_features():
    builder = RoadGraphBuilder([Segment(edge_ids=[1], u=1, v=2, points=[A, A], label=None)])
    assert torch.isfinite(builder.x).all()


def test_straight_segment_has_curvature_in_range():
    builder = RoadGraphBuilder([Segment(edge_ids=[1], u=1, v=2, points=[A, B, C], label=None)])
    assert 0.0 <= builder.x[0, 1].item() <= 1.0


def test_duplicate_edge_id_raises():
    duplicates = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B], label=None),
        Segment(edge_ids=[1], u=2, v=3, points=[B, C], label=None),
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
    traffic_stats = fit_traffic_standardizer([segments[0], segments[2]])

    builder.standardize_features(segments, train_mask)

    # Every segment is scaled, also the ones outside the training set
    expected = []
    for segment, degree in zip(segments, DEGREES):
        expected.append(compute_features(segment, degree, *stats) + compute_traffic_features(segment, traffic_stats))
    assert torch.equal(builder.x, torch.tensor(expected, dtype=torch.float))


def test_standardize_features_is_not_affected_by_non_training_segments(segments):
    train_mask = torch.tensor([True, True, True, False])
    changed = list(segments)
    changed[3] = Segment(edge_ids=[30], u=5, v=6, points=[D, (10.5, 54.5)], label=None, oneway=None)
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


# Which road types become which label is decided in the database (src/db/create_road_type_labels.py,
# tested in tests/db/unit). The graph builder only turns the label name into a number.

@pytest.mark.parametrize("road_type", ROAD_CLASSES)
def test_label_is_the_index_in_road_classes(road_type):
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], label=road_type)

    builder = RoadGraphBuilder([seg])

    assert builder.y.tolist() == [ROAD_CLASSES.index(road_type)]
    assert builder.mask.tolist() == [True]


def test_label_not_in_classes_is_unlabelled():
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], label="primary")

    builder = RoadGraphBuilder([seg], classes=["residential"])

    assert builder.y.tolist() == [-1]
    assert builder.mask.tolist() == [False]


def test_unlabelled_segment_stays_in_the_graph_as_a_neighbour():
    segs = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B], label="primary"),
        Segment(edge_ids=[2], u=2, v=3, points=[B, C], label=None),
    ]

    builder = RoadGraphBuilder(segs)

    assert builder.x.shape[0] == 2
    assert builder.edge_index.tolist() == [[0, 1], [1, 0]]
    assert builder.mask.tolist() == [True, False]


# Traffic features

def test_segment_without_traffic_has_zero_traffic_features():
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B])

    assert compute_raw_traffic_features(seg) == [0.0, 0, 0.0, 0.0, 0.0, 0.0]


def test_raw_traffic_features():
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], traversal_count=9, avg_speed=12.5, avg_acceleration=-0.5,
                  speed_std=3.0, slow_share=0.25)

    assert compute_raw_traffic_features(seg) == [pytest.approx(math.log(10)), 1, 12.5, -0.5, 3.0, 0.25]


def test_traffic_features_are_columns_4_to_9(segments):
    segments[0].traversal_count = 3
    segments[0].avg_speed = 10.0
    segments[0].avg_acceleration = 0.5
    segments[0].speed_std = 2.0
    segments[0].slow_share = 0.25

    builder = RoadGraphBuilder(segments)

    assert builder.x[0, 4:].tolist() == pytest.approx([math.log(4), 1.0, 10.0, 0.5, 2.0, 0.25])
    assert builder.x[1, 4:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_traffic_standardizer_only_uses_known_speeds():
    segs = [
        Segment(edge_ids=[1], u=1, v=2, points=[A, B], traversal_count=1, avg_speed=10.0),
        Segment(edge_ids=[2], u=2, v=3, points=[B, C], traversal_count=1, avg_speed=20.0),
        Segment(edge_ids=[3], u=3, v=4, points=[A, C]),
    ]

    stats = fit_traffic_standardizer(segs)
    count_mean, count_std = stats[0]
    speed_mean, speed_std = stats[1]
    acc_mean, acc_std = stats[2]

    # the segment without traffic does not pull the speed mean down
    assert (speed_mean, speed_std) == pytest.approx((15.0, 5.0))
    # no segment has an acceleration, so nothing is changed
    assert (acc_mean, acc_std) == (0.0, 1.0)
    # but the count uses all three segments
    assert count_mean == pytest.approx(2 * math.log(2) / 3)


def test_missing_speed_gets_z_zero_and_has_traffic_stays_binary():
    known = Segment(edge_ids=[1], u=1, v=2, points=[A, B], traversal_count=4, avg_speed=20.0, avg_acceleration=1.0,
                    speed_std=3.0, slow_share=0.5)
    missing = Segment(edge_ids=[2], u=2, v=3, points=[B, C])
    # (mean, std) for count, speed, acceleration, speed_std and slow_share
    stats = [(0.0, 1.0), (10.0, 5.0), (0.0, 2.0), (1.0, 1.0), (0.25, 0.25)]

    assert compute_traffic_features(known, stats) == pytest.approx([math.log(5), 1, 2.0, 0.5, 2.0, 1.0])
    assert compute_traffic_features(missing, stats) == [0.0, 0, 0.0, 0.0, 0.0, 0.0]


# Neighbour label features

def test_neighbour_labels_are_shares_of_training_neighbours(segments):
    # segments 0 and 1 are primary, 2 is residential, 3 has no label (see the fixture)
    builder = RoadGraphBuilder(segments)
    train_mask = torch.tensor([True, True, True, False])

    builder.add_neighbour_label_features(train_mask)

    primary, residential = ROAD_CLASSES.index("primary"), ROAD_CLASSES.index("residential")
    shares = builder.x[:, 10:]
    assert shares.shape == (4, len(ROAD_CLASSES))
    # segment 2 has neighbours 0, 1 (primary) and 3 (no training label): 2 of 3 are primary
    assert shares[2, primary].item() == pytest.approx(2 / 3)
    assert shares[2].sum().item() == pytest.approx(2 / 3)
    # segment 3 only has neighbour 2, which is residential
    assert shares[3, residential].item() == 1.0


def test_neighbour_labels_only_use_training_segments(segments):
    builder = RoadGraphBuilder(segments)
    # segment 2 (residential) is not a training segment, so segment 3 must not see its label
    train_mask = torch.tensor([True, True, False, False])

    builder.add_neighbour_label_features(train_mask)

    assert builder.x[3, 10:].tolist() == [0.0] * len(ROAD_CLASSES)


def test_neighbour_labels_never_use_the_own_label():
    # one training segment without neighbours: its own label must not show up
    seg = Segment(edge_ids=[1], u=1, v=2, points=[A, B], label="primary")
    builder = RoadGraphBuilder([seg])

    builder.add_neighbour_label_features(torch.tensor([True]))

    assert builder.x[0, 10:].tolist() == [0.0] * len(ROAD_CLASSES)


def test_neighbour_labels_keep_the_other_features(segments):
    builder = RoadGraphBuilder(segments)
    train_mask = torch.tensor([True, True, True, False])
    builder.standardize_features(segments, train_mask)
    before = builder.x.clone()

    builder.add_neighbour_label_features(train_mask)

    assert torch.equal(builder.x[:, :10], before)
