"""Fills the trip and traversal tables from the raw taxi data in public.trajectories.

Run it from the repo root (it uses DATABASE_URL from .env):
    python -m src.db.create_trips

It is safe to run more than once, all old trips and traversals are deleted first.

Each row in trajectories becomes one trip, and each of its road points becomes one traversal.
A road point looks like [segment_id, enter_time, speed_mps, duration_s] and a gps point looks like
[?, time, lon, lat]. All times are unix timestamps, we save them as UTC.

Trip ids are 1, 2, 3, ... with the trajectories sorted by taxi_id and trajectory_id, so the same data
always gives the same ids. trip.trajectory_id points back to the raw row.
"""

import asyncio

from sqlalchemy import text

from src.db.engine import engine, session_scope

# Which file from the dataset to make trips from. traj_train_20 to traj_train_80 are smaller copies
# of traj_train_100, so we only use the full one.
DATASET = "traj_train_100"

# How many taxis we do in one batch (500 taxis is around 70.000 trips)
TAXIS_PER_BATCH = 500

# Makes the trips. :first_trip_id is the id of the first trip in this batch.
INSERT_TRIPS = text("""
    INSERT INTO attribute_prediction.trip (trip_id, start_time, end_time, start_pos, end_pos, trajectory_id)
    SELECT
        :first_trip_id - 1 + row_number() OVER (ORDER BY r.taxi_id, r.trajectory_id),
        -- start = when the trip entered its first segment
        (SELECT min(to_timestamp((p->>1)::float8)) FROM jsonb_array_elements(r.road_points) p) AT TIME ZONE 'UTC',
        -- end = when the trip left its last segment (enter time + time on the segment)
        (SELECT max(to_timestamp((p->>1)::float8 + (p->>3)::float8)) FROM jsonb_array_elements(r.road_points) p)
            AT TIME ZONE 'UTC',
        -- first and last gps point, they are [?, time, lon, lat]
        ST_SetSRID(ST_MakePoint((r.gps_points->0->>2)::float8, (r.gps_points->0->>3)::float8), 4326),
        ST_SetSRID(ST_MakePoint((r.gps_points->-1->>2)::float8, (r.gps_points->-1->>3)::float8), 4326),
        r.trajectory_id
    FROM public.trajectories r
    WHERE r.dataset = :dataset AND r.taxi_id = ANY(:taxi_ids)
""")

# Makes one traversal per road point. Points with a negative speed or time, or a segment that does
# not exist, are skipped (the database would reject them). seq is the position in road_points, so it
# can have gaps if a point was skipped.
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

COUNT_ROAD_POINTS = text("""
    SELECT coalesce(sum(jsonb_array_length(road_points)), 0) FROM public.trajectories
    WHERE dataset = :dataset AND taxi_id = ANY(:taxi_ids)
""")


async def get_taxi_ids(session, dataset):
    """All taxi ids in the dataset, sorted the same way as the trip ids."""
    result = await session.execute(
        text("SELECT DISTINCT taxi_id FROM public.trajectories WHERE dataset = :dataset ORDER BY taxi_id"),
        {"dataset": dataset},
    )
    return [row[0] for row in result.all()]


async def create_trips_for_taxis(session, dataset, taxi_ids, first_trip_id):
    """Makes the trips and traversals for some taxis. Returns (trips, traversals, skipped road points)."""
    values = {"dataset": dataset, "taxi_ids": taxi_ids, "first_trip_id": first_trip_id}
    trips = (await session.execute(INSERT_TRIPS, values)).rowcount
    traversals = (await session.execute(INSERT_TRAVERSALS, values)).rowcount
    road_points = await session.scalar(COUNT_ROAD_POINTS, values)
    return trips, traversals, road_points - traversals


async def create_trips():
    async with session_scope() as session:
        taxi_ids = await get_taxi_ids(session, DATASET)
        if not taxi_ids:
            raise RuntimeError(f"There are no trajectories with dataset = {DATASET}. Run the import in the data repo first.")
        # Delete the old data. CASCADE also empties traversal.
        await session.execute(text("TRUNCATE attribute_prediction.trip CASCADE"))
    print(f"Making trips for {len(taxi_ids)} taxis in {DATASET}")

    next_trip_id = 1
    total_traversals = 0
    total_skipped = 0
    for start in range(0, len(taxi_ids), TAXIS_PER_BATCH):
        batch = taxi_ids[start:start + TAXIS_PER_BATCH]
        # one transaction per batch, so we do not keep one giant transaction open
        async with session_scope() as session:
            trips, traversals, skipped = await create_trips_for_taxis(session, DATASET, batch, next_trip_id)
        next_trip_id += trips
        total_traversals += traversals
        total_skipped += skipped
        print(f"  taxis {start + len(batch)}/{len(taxi_ids)}: {next_trip_id - 1} trips, {total_traversals} traversals")

    print(f"Done: {next_trip_id - 1} trips, {total_traversals} traversals, {total_skipped} road points skipped")


async def main():
    try:
        await create_trips()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
