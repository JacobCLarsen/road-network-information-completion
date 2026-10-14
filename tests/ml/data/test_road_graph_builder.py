import math
import pytest
import torch
from src.ml.data.road_graph_builder import RoadGraphBuilder, Segment, ROAD_CLASSES, haversine

A, B, C = (10.000, 55.000), (10.001, 55.000), (10.002, 55.000)
BEND, D, E = (10.001, 54.999), (10.002, 54.999), (10.002, 54.998)


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


def test_get_static_components_keys(segments):
    components = RoadGraphBuilder(segments).get_static_components()
    assert set(components.keys()) == {'x', 'y', 'mask', 'edge_index', 'seg_key', 'geometry', 'edge_to_segment'}
