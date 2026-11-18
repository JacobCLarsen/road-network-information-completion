# src/ml/data/database_loader.py

# Loads the road network and the road type labels from the database, and turns them into a list of
# Segment objects for the RoadGraphBuilder.
#
# The road segments are in the attribute_prediction.segment view and the labels in attribute_prediction.segment_label.
# The labels are made by src/db/create_road_type_labels.py, so run that first on a new database.

import asyncio
import json

from sqlalchemy import text

from src.db.engine import engine, session_scope
from src.ml.data.road_graph_builder import Segment

# Which labels we train on. Version 1 = the highway tag, with link roads merged into their parent type.
LABEL_RULE_VERSION = 1

# One row per segment in the map, with the name of its road type (or NULL if it has no label).
# Note that a segment in the view is one direction of a road, so a two-way road is two rows.
# ST_AsGeoJSON gives us the shape of the road as JSON, so we can read the points with json.loads.
# We sort by segment_id so the segments always come in the same order, otherwise the split would change.
SELECT_EDGES = text("""
    SELECT s.segment_id, s.u, s.v, s.key, s.oneway, ST_AsGeoJSON(s.geom), c.name
    FROM attribute_prediction.segment s
    LEFT JOIN attribute_prediction.segment_label l
        ON l.segment_id = s.segment_id
        AND l.label_rule_version = :version
        AND l.attribute_id = (SELECT attribute_id FROM attribute_prediction.attribute WHERE name = 'road_type')
    LEFT JOIN attribute_prediction.attribute_class c ON c.class_id = l.class_id
    ORDER BY s.segment_id
""")


async def read_edges():
    # reads all the edges from the database. Every edge is a dict with its columns.
    # oneway is True or False (the view already turned the oneway tag into a boolean)
    async with session_scope() as session:
        result = await session.execute(SELECT_EDGES, {"version": LABEL_RULE_VERSION})
        rows = result.all()

    edges = []
    for fid, u, v, key, oneway, geojson, label in rows:
        # the geometry is a LINESTRING, so coordinates is a list of [lon, lat]
        points = []
        for point in json.loads(geojson)["coordinates"]:
            points.append((point[0], point[1]))
        edges.append({"fid": fid, "u": u, "v": v, "key": key, "oneway": oneway, "points": points, "label": label})
    return edges


def edges_to_segments(edges):
    # The edges are already split at junctions, but a two-way road is in the view twice:
    # once from u to v and once from v to u. We merge the two into one segment that keeps both edge ids.

    # Find the fid of every two-way edge by (u, v, key).
    # key is there because two different roads can have the same u and v.
    two_way_ids = {}
    for edge in edges:
        if not edge["oneway"]:
            two_way_ids[(edge["u"], edge["v"], edge["key"])] = edge["fid"]

    segments = []
    for edge in edges:
        u = edge["u"]
        v = edge["v"]

        # look for the edge that goes the other way. Loops (u == v) don't have one
        reverse_id = None
        if not edge["oneway"] and u != v:
            reverse_id = two_way_ids.get((v, u, edge["key"]))

        if reverse_id is None:
            edge_ids = [edge["fid"]]
        else:
            # both directions exist, so we only keep the one where u < v, otherwise we add the road twice
            if u > v:
                continue
            edge_ids = [edge["fid"], reverse_id]

        segments.append(Segment(edge_ids, u, v, edge["points"], edge["label"], edge["oneway"]))

    return segments


def load_segments():
    # The training scripts are not async, so this function runs the async database code for them
    edges = asyncio.run(read_edges_and_close())
    return edges_to_segments(edges)


async def read_edges_and_close():
    edges = await read_edges()
    # close the connections, because asyncio.run closes the event loop they belong to
    await engine.dispose()
    return edges
