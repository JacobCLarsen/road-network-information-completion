from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, Boolean, Double, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import SCHEMA, Base


class OsmEdge(Base):
    # The raw road segments from the dataset (filled by the import script in the data repo).
    # We only have this class because our other tables have foreign keys to osm_edges.fid.
    # To read segments, use the Segment view below instead.
    __tablename__ = "osm_edges"
    __table_args__ = {"schema": "public"}

    fid: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    u: Mapped[int | None] = mapped_column(BigInteger)
    v: Mapped[int | None] = mapped_column(BigInteger)
    key: Mapped[int | None] = mapped_column(BigInteger)
    osmid: Mapped[str | None] = mapped_column(Text)
    oneway: Mapped[str | None] = mapped_column(Text)
    lanes: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str | None] = mapped_column(Text)
    highway: Mapped[str | None] = mapped_column(Text)
    length: Mapped[float | None] = mapped_column(Double)
    bridge: Mapped[str | None] = mapped_column(Text)
    ref: Mapped[str | None] = mapped_column(Text)
    tunnel: Mapped[str | None] = mapped_column(Text)
    junction: Mapped[str | None] = mapped_column(Text)
    maxspeed: Mapped[str | None] = mapped_column(Text)
    access: Mapped[str | None] = mapped_column(Text)
    service: Mapped[str | None] = mapped_column(Text)
    width: Mapped[str | None] = mapped_column(Text)
    geometry = mapped_column(Geometry("LINESTRING", srid=4326, spatial_index=False))


class Segment(Base):
    # This is a VIEW (read only) on top of osm_edges with nicer column names.
    # segment_id is the same as osm_edges.fid. u and v are the ids of the start and end node.
    __tablename__ = "segment"
    __table_args__ = {"schema": SCHEMA, "info": {"is_view": True}}

    segment_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    u: Mapped[int | None] = mapped_column(BigInteger)
    v: Mapped[int | None] = mapped_column(BigInteger)
    length_m: Mapped[float | None] = mapped_column(Double)
    oneway: Mapped[bool | None] = mapped_column(Boolean)
    geom = mapped_column(Geometry("LINESTRING", srid=4326, spatial_index=False))
    highway: Mapped[str | None] = mapped_column(Text)
    lanes: Mapped[str | None] = mapped_column(Text)
    maxspeed: Mapped[str | None] = mapped_column(Text)
    ref: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str | None] = mapped_column(Text)
    bridge: Mapped[str | None] = mapped_column(Text)
    tunnel: Mapped[str | None] = mapped_column(Text)
