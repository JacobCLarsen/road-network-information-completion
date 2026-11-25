# Tests for edges_to_segments in database_loader.py. These do not need a database,
# we just make the edges by hand like read_edges() would return them.

import pytest

from src.ml.data.database_loader import edges_to_segments

A, B, C = (10.000, 55.000), (10.001, 55.000), (10.002, 55.000)


def edge(fid, u, v, oneway, key=0, label=None):
    points = [A, B]
    if u > v:
        points = [B, A]
    return {"fid": fid, "u": u, "v": v, "key": key, "oneway": oneway, "points": points, "label": label}


def test_two_way_road_becomes_one_segment():
    edges = [edge(1, 1, 2, False, label="residential"), edge(2, 2, 1, False, label="residential")]

    segments = edges_to_segments(edges)

    assert len(segments) == 1
    # the one with u < v is kept, and it gets both edge ids
    assert segments[0].edge_ids == [1, 2]
    assert (segments[0].u, segments[0].v) == (1, 2)
    assert segments[0].points == [A, B]
    assert segments[0].label == "residential"


def test_one_way_roads_are_not_merged():
    # two one-way roads going opposite ways are two different segments
    edges = [edge(1, 1, 2, True), edge(2, 2, 1, True)]

    segments = edges_to_segments(edges)

    assert [s.edge_ids for s in segments] == [[1], [2]]


def test_different_key_is_not_merged():
    # same junctions but a different key means it is another road
    edges = [edge(1, 1, 2, False, key=0), edge(2, 2, 1, False, key=1)]

    segments = edges_to_segments(edges)

    assert [s.edge_ids for s in segments] == [[1], [2]]


def test_loop_is_kept_once():
    loop = {"fid": 7, "u": 3, "v": 3, "key": 0, "oneway": False, "points": [A, B, C, A], "label": None}

    segments = edges_to_segments([loop])

    assert len(segments) == 1
    assert segments[0].edge_ids == [7]


def test_traffic_of_both_directions_is_added_together():
    edges = [edge(1, 1, 2, False), edge(2, 2, 1, False)]
    # {segment_id: [traversals, speed sum, acceleration sum, acceleration count, squared speed sum, slow traversals]}
    traffic = {1: [2, 20.0, 1.0, 2, 200.0, 0], 2: [3, 45.0, -1.0, 1, 675.0, 1]}

    segments = edges_to_segments(edges, traffic)

    assert segments[0].traversal_count == 5
    assert segments[0].avg_speed == 13.0  # (20 + 45) / 5
    assert segments[0].avg_acceleration == 0.0  # (1 - 1) / 3
    assert segments[0].slow_share == 0.2  # 1 of 5


def test_segment_without_traffic():
    segments = edges_to_segments([edge(1, 1, 2, True)], {})

    assert segments[0].traversal_count == 0
    assert segments[0].avg_speed is None
    assert segments[0].avg_acceleration is None
    assert segments[0].speed_std is None
    assert segments[0].slow_share is None


def test_traffic_without_any_acceleration():
    # e.g. a segment that was only ever the first one of a trip
    segments = edges_to_segments([edge(1, 1, 2, True)], {1: [2, 16.0, 0.0, 0, 128.0, 0]})

    assert segments[0].avg_speed == 8.0
    assert segments[0].avg_acceleration is None


def test_speed_std():
    # speeds 4 and 8: mean 6, std 2
    segments = edges_to_segments([edge(1, 1, 2, True)], {1: [2, 12.0, 0.0, 0, 80.0, 0]})

    assert segments[0].avg_speed == 6.0
    assert segments[0].speed_std == pytest.approx(2.0)


def test_speed_std_is_zero_when_all_speeds_are_the_same():
    # 3 traversals at 0.1 m/s, rounding can make the variance a tiny bit negative
    segments = edges_to_segments([edge(1, 1, 2, True)], {1: [3, 0.3, 0.0, 0, 0.03, 3]})

    assert segments[0].speed_std == pytest.approx(0.0, abs=1e-6)
    assert segments[0].slow_share == 1.0
