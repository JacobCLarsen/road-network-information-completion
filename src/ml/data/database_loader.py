# src/ml/data/database_loader.py

# Loads the road network and the road type labels from the database, and turns them into a list of
# Segment objects for the RoadGraphBuilder.
#
# The road segments are in the attribute_prediction.segment view and the labels in attribute_prediction.segment_label.
# The labels are made by src/db/create_road_type_labels.py, so run that first on a new database.
# The traffic features come from attribute_prediction.traversal, which is made by src/db/create_trips.py.

import asyncio
import json
import math

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

# Acceleration above this (in m/s^2) is not possible for a car, so it must be GPS noise and we skip it.
# About 0.1 % of the values are bigger than this.
MAX_ACCELERATION = 10.0

# A traversal slower than this (in m/s, 2 m/s = 7.2 km/h) counts as slow: the taxi was stopped or crawling,
# e.g. at a traffic light or in a traffic jam.
SLOW_SPEED = 2.0

# The traffic on every segment that has at least one traversal (segments without traffic are not in the result).
# For each segment we get: number of traversals, sum of the speeds, sum of the accelerations, how many
# accelerations there are, sum of the squared speeds (for the standard deviation) and how many traversals were slow.
# We return sums instead of averages, so we can add the two directions of a road together in Python and then divide.
#
# The acceleration of a traversal is the change in speed from the segment before it in the same trip,
# divided by the time between entering the two segments. lag() gives the value from the row before in the
# same trip (sorted by seq). If seq - lag(seq) is not 1, a road point was skipped in between, so we don't
# know the segment before and the acceleration is NULL. The first traversal of a trip is also NULL.
SELECT_TRAFFIC = text("""
    WITH steps AS (
        SELECT
            segment_id,
            speed_mps,
            CASE WHEN seq - lag(seq) OVER w = 1 THEN
                (speed_mps - lag(speed_mps) OVER w) / extract(epoch FROM enter_time - lag(enter_time) OVER w)
            END AS acceleration
        FROM attribute_prediction.traversal
        WINDOW w AS (PARTITION BY trip_id ORDER BY seq)
    )
    SELECT
        segment_id,
        count(*),
        sum(speed_mps),
        coalesce(sum(acceleration) FILTER (WHERE abs(acceleration) <= :max_acceleration), 0),
        count(acceleration) FILTER (WHERE abs(acceleration) <= :max_acceleration),
        sum(speed_mps * speed_mps),
        count(*) FILTER (WHERE speed_mps < :slow_speed)
    FROM steps
    GROUP BY segment_id
""")


async def read_traffic():
    # returns a dict {segment_id: [traversals, speed sum, acceleration sum, acceleration count,
    #                                squared speed sum, slow traversals]}
    async with session_scope() as session:
        result = await session.execute(SELECT_TRAFFIC, {"max_acceleration": MAX_ACCELERATION, "slow_speed": SLOW_SPEED})
        rows = result.all()

    traffic = {}
    for row in rows:
        segment_id = row[0]
        traffic[segment_id] = list(row[1:])
    return traffic


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


def edges_to_segments(edges, traffic=None):
    # The edges are already split at junctions, but a two-way road is in the view twice:
    # once from u to v and once from v to u. We merge the two into one segment that keeps both edge ids.
    # traffic is the dict from read_traffic(). The traffic of both directions is added together.
    if traffic is None:
        traffic = {}

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

        segment = Segment(edge_ids, u, v, edge["points"], edge["label"], edge["oneway"])
        add_traffic(segment, traffic)
        segments.append(segment)

    return segments


def add_traffic(segment, traffic):
    # adds up the traffic of all the edge ids of the segment, and saves the averages on the segment
    count = 0
    speed_sum = 0.0
    acceleration_sum = 0.0
    acceleration_count = 0
    squared_speed_sum = 0.0
    slow_count = 0
    for edge_id in segment.edge_ids:
        if edge_id in traffic:
            count = count + traffic[edge_id][0]
            speed_sum = speed_sum + traffic[edge_id][1]
            acceleration_sum = acceleration_sum + traffic[edge_id][2]
            acceleration_count = acceleration_count + traffic[edge_id][3]
            squared_speed_sum = squared_speed_sum + traffic[edge_id][4]
            slow_count = slow_count + traffic[edge_id][5]

    segment.traversal_count = count
    # no traffic means we don't know the speed, so it stays None
    if count > 0:
        segment.avg_speed = speed_sum / count
        # std = sqrt(mean of the squares - square of the mean). Rounding errors can make it a tiny bit
        # negative when all speeds are the same, so we use max(..., 0)
        variance = squared_speed_sum / count - segment.avg_speed ** 2
        segment.speed_std = math.sqrt(max(variance, 0.0))
        segment.slow_share = slow_count / count
    if acceleration_count > 0:
        segment.avg_acceleration = acceleration_sum / acceleration_count


def load_segments():
    # The training scripts are not async, so this function runs the async database code for them
    edges, traffic = asyncio.run(read_from_database())
    return edges_to_segments(edges, traffic)


async def read_from_database():
    edges = await read_edges()
    # this one takes a minute or two, because it goes through all the traversals
    traffic = await read_traffic()
    # close the connections, because asyncio.run closes the event loop they belong to
    await engine.dispose()
    return edges, traffic
