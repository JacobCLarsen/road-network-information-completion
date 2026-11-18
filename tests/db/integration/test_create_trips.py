import json
from datetime import datetime, timezone

from sqlalchemy import func, select, text

from src.db.create_trips import create_trips_for_taxis, get_taxi_ids
from src.db.models import Traversal, Trip
from tests.db.integration.conftest import SEGMENT_IDS

DATASET = "test_dataset"


def utc(seconds):
    # unix time -> datetime in UTC without time zone, like in the database
    return datetime.fromtimestamp(seconds, timezone.utc).replace(tzinfo=None)


async def add_trajectory(session, taxi_id, trajectory_id, gps_points, road_points):
    await session.execute(
        text("""
            INSERT INTO public.trajectories (dataset, taxi_id, trajectory_id, gps_points, road_points)
            VALUES (:dataset, :taxi_id, :trajectory_id, CAST(:gps AS jsonb), CAST(:road AS jsonb))
        """),
        {"dataset": DATASET, "taxi_id": taxi_id, "trajectory_id": trajectory_id,
         "gps": json.dumps(gps_points), "road": json.dumps(road_points)},
    )


async def test_create_trips(session, segments):
    # taxi "b" comes after "a", so its trip should get the higher id
    await add_trajectory(
        session, "b", "200:b",
        gps_points=[[1.0, 200, 116.0, 39.0], [2.0, 260, 116.1, 39.1]],
        road_points=[[SEGMENT_IDS[0], 200.0, 5.0, 20.0]],
    )
    await add_trajectory(
        session, "a", "100:a",
        gps_points=[[1.0, 100, 116.2, 39.2], [2.0, 150, 116.3, 39.3]],
        road_points=[
            [SEGMENT_IDS[0], 100.0, 10.0, 10.0],
            [SEGMENT_IDS[1], 110.0, 8.0, 25.0],
            [-1, 135.0, 8.0, 5.0],  # segment that does not exist, should be skipped
        ],
    )

    taxi_ids = await get_taxi_ids(session, DATASET)
    assert taxi_ids == ["a", "b"]

    trips, traversals, skipped = await create_trips_for_taxis(session, DATASET, taxi_ids, first_trip_id=1)
    assert (trips, traversals, skipped) == (2, 3, 1)

    trip_a = (await session.execute(select(Trip).where(Trip.trajectory_id == "100:a"))).scalar_one()
    assert trip_a.trip_id == 1
    assert trip_a.start_time == utc(100)
    assert trip_a.end_time == utc(140)  # 135 + 5, the skipped point still counts for the trip time
    lon = await session.scalar(func.ST_X(Trip.start_pos).select().where(Trip.trip_id == 1))
    assert lon == 116.2

    trip_b = (await session.execute(select(Trip).where(Trip.trajectory_id == "200:b"))).scalar_one()
    assert trip_b.trip_id == 2

    result = await session.execute(select(Traversal).where(Traversal.trip_id == 1).order_by(Traversal.seq))
    steps = list(result.scalars().all())
    assert [(s.seq, s.segment_id, s.speed_mps, s.duration_s) for s in steps] == [
        (1, SEGMENT_IDS[0], 10.0, 10.0),
        (2, SEGMENT_IDS[1], 8.0, 25.0),
    ]
    assert steps[1].enter_time == utc(110)
