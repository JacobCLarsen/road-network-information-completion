# Tests for edges_to_segments in database_loader.py. These do not need a database,
# we just make the edges by hand like read_edges() would return them.

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
