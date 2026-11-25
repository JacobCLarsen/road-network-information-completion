# Script that fills the trip and traversal tables from the raw taxi data (public.trajectories).
# Run it with: python -m src.db.create_trips
# Old trips are deleted first, so it is fine to run it again.
#
# Every trajectory becomes one trip, and every road point in it becomes one traversal.
# road point = [segment_id, enter_time, speed_mps, duration_s]
# gps point  = [offset_m, time, lon, lat]
# The times are unix timestamps and we save them in UTC.
#
# offset_m is (we think) how many meters along its matched road segment the gps point is, counted from
# the start of the segment. Although we could not find any information on the OSM website.
#
# The trip ids are 1, 2, 3 ... sorted by taxi_id and then trajectory_id, so we get the same ids every time.

import asyncio

from sqlalchemy import text

from src.db.engine import engine, session_scope

# The file from the dataset we use. traj_train_20 to traj_train_80 are just smaller parts
# of traj_train_100, so we only need this one.
DATASET = "traj_train_100"

# How many taxis we do in one batch
TAXIS_PER_BATCH = 500

# Makes the trips. first_trip_id is the id of the first trip in this batch.
# jsonb_array_elements(r.road_points) below unpacks the JSON list of road points,
# and p ->> 1 takes element 1 (counting from 0) of each road point, which is the enter_time.
# This is used to find the earliest start_time and latest end_time of a road segment,
# which is the start and end times for the total trip
# ST_MakePoint builds a new PostGIS point. 4326 marks the format, which is normal GPS points (WGS84)
# Note that single arrow (->) returns JSON and double arrow (->>) returns a text value from that JSON.
INSERT_TRIPS = text("""
    INSERT INTO attribute_prediction.trip (trip_id, start_time, end_time, start_pos, end_pos, trajectory_id)
    SELECT
        :first_trip_id - 1 + row_number() OVER (ORDER BY r.taxi_id, r.trajectory_id),
        -- start = when the trip entered its first segment.
        (SELECT min(to_timestamp((p->>1)::float8)) FROM jsonb_array_elements(r.road_points) p) AT TIME ZONE 'UTC',
        -- end = when the trip left its last segment (enter time + time on the segment)
        (SELECT max(to_timestamp((p->>1)::float8 + (p->>3)::float8)) FROM jsonb_array_elements(r.road_points) p)
            AT TIME ZONE 'UTC',
        -- first and last gps point, they are [offset_m, time, lon, lat]
        ST_SetSRID(ST_MakePoint((r.gps_points->0->>2)::float8, (r.gps_points->0->>3)::float8), 4326),
        ST_SetSRID(ST_MakePoint((r.gps_points->-1->>2)::float8, (r.gps_points->-1->>3)::float8), 4326),
        r.trajectory_id
    FROM public.trajectories r
    WHERE r.dataset = :dataset AND r.taxi_id = ANY(:taxi_ids)
""")

# One traversal per road point.
# We skip points with a negative speed or time, or a segment that is not in osm_edges,
# because the database would not accept them anyway.
# seq is the position in road_points, so there can be a gap in seq if we skipped a point.
INSERT_TRAVERSALS = text("""
    INSERT INTO attribute_prediction.traversal (trip_id, seq, segment_id, enter_time, duration_s, speed_mps)
    SELECT
        t.trip_id,
        p.ord,
        (p.value->>0)::bigint,
        to_timestamp((p.value->>1)::float8) AT TIME ZONE 'UTC',
        (p.value->>3)::float8,
        (p.value->>2)::float8
    FROM public.trajectories r
    JOIN attribute_prediction.trip t ON t.trajectory_id = r.trajectory_id
    CROSS JOIN LATERAL jsonb_array_elements(r.road_points) WITH ORDINALITY p(value, ord)
    WHERE r.dataset = :dataset AND r.taxi_id = ANY(:taxi_ids)
      AND (p.value->>2)::float8 >= 0
      AND (p.value->>3)::float8 >= 0
      AND EXISTS (SELECT 1 FROM public.osm_edges e WHERE e.fid = (p.value->>0)::bigint)
""")

# coalesce(x, 0) returns the first value in the list that is not NULL, so if x is NULL we get 0.
# We need it because the sum of no road points gives NULL instead of 0.
COUNT_ROAD_POINTS = text("""
    SELECT coalesce(sum(jsonb_array_length(road_points)), 0) FROM public.trajectories
    WHERE dataset = :dataset AND taxi_id = ANY(:taxi_ids)
""")


async def get_taxi_ids(session, dataset):
    # all the taxis in the dataset, sorted the same way as the trip ids
    result = await session.execute(
        text("SELECT DISTINCT taxi_id FROM public.trajectories WHERE dataset = :dataset ORDER BY taxi_id"),
        {"dataset": dataset},
    )
    taxi_ids = []
    for row in result.all():
        taxi_ids.append(row[0])
    return taxi_ids


async def create_trips_for_taxis(session, dataset, taxi_ids, first_trip_id):
    # makes the trips and traversals for a list of taxis
    # returns how many trips and traversals we made, and how many road points were skipped
    params = {"dataset": dataset, "taxi_ids": taxi_ids, "first_trip_id": first_trip_id}

    result = await session.execute(INSERT_TRIPS, params)
    trips = result.rowcount

    result = await session.execute(INSERT_TRAVERSALS, params)
    traversals = result.rowcount

    road_points = await session.scalar(COUNT_ROAD_POINTS, params)
    skipped = road_points - traversals

    return trips, traversals, skipped


async def create_trips():
    async with session_scope() as session:
        taxi_ids = await get_taxi_ids(session, DATASET)
        if len(taxi_ids) == 0:
            raise RuntimeError("No trajectories found for " + DATASET + ", run the import in the data repo first")

        # delete the old trips (CASCADE also deletes the traversals)
        await session.execute(text("TRUNCATE attribute_prediction.trip CASCADE"))

    print("Making trips for", len(taxi_ids), "taxis in", DATASET)

    next_trip_id = 1
    total_trips = 0
    total_traversals = 0
    total_skipped = 0

    # we do 500 taxis at a time, and save after each batch,
    # so we don't have one huge transaction with 34 million rows
    for i in range(0, len(taxi_ids), TAXIS_PER_BATCH):
        batch = taxi_ids[i:i + TAXIS_PER_BATCH]

        async with session_scope() as session:
            trips, traversals, skipped = await create_trips_for_taxis(session, DATASET, batch, next_trip_id)

        next_trip_id = next_trip_id + trips
        total_trips = total_trips + trips
        total_traversals = total_traversals + traversals
        total_skipped = total_skipped + skipped

        done = i + len(batch)
        print("  taxis", done, "/", len(taxi_ids), "-", total_trips, "trips,", total_traversals, "traversals")

    print("Done!", total_trips, "trips,", total_traversals, "traversals,", total_skipped, "road points skipped")


async def main():
    await create_trips()
    # close the database connections before the script ends
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
