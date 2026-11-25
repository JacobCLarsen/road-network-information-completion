from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, DateTime, Double, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import SCHEMA, Base


class Trip(Base):
    # One taxi trip. Made from a row in public.trajectories by src/db/create_trips.py
    __tablename__ = "trip"
    __table_args__ = {"schema": SCHEMA}

    trip_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    start_time: Mapped[datetime] = mapped_column(DateTime)  # in UTC
    end_time: Mapped[datetime] = mapped_column(DateTime)  # in UTC
    start_pos = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    end_pos = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    # The trajectory_id of the raw row in public.trajectories, so we can find it again
    trajectory_id: Mapped[str] = mapped_column(Text)


class Traversal(Base):
    # One segment a trip drove on. seq is the order in the trip (1, 2, 3, ...)
    __tablename__ = "traversal"
    __table_args__ = {"schema": SCHEMA}

    trip_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.trip.trip_id"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    segment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("public.osm_edges.fid"))
    enter_time: Mapped[datetime] = mapped_column(DateTime)  # in UTC
    duration_s: Mapped[float] = mapped_column(Double)  # seconds spent on the segment
    speed_mps: Mapped[float] = mapped_column(Double)  # meters per second
