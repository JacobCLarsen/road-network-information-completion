from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Double, ForeignKey, ForeignKeyConstraint, Identity, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import SCHEMA, Base, Partition, TaskType, partition_enum, task_type_enum


class Attribute(Base):
    # Something we want to predict for a segment, for example "road_type"
    __tablename__ = "attribute"
    __table_args__ = {"schema": SCHEMA}

    attribute_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    task_type: Mapped[TaskType] = mapped_column(task_type_enum)
    description: Mapped[str | None] = mapped_column(Text)


class AttributeClass(Base):
    # The classes of an attribute, e.g. "primary" or "residential" for road_type.
    # category is the coarse group the class belongs to.
    __tablename__ = "attribute_class"
    __table_args__ = {"schema": SCHEMA}

    class_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    attribute_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.attribute.attribute_id"))
    name: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)


class LabelRule(Base):
    # Describes how the labels were made. Has a version so we can change the rule later.
    __tablename__ = "label_rule"
    __table_args__ = {"schema": SCHEMA}

    attribute_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.attribute.attribute_id"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SegmentLabel(Base):
    # The true label of a segment. For classification class_id is set, for regression value is set.
    __tablename__ = "segment_label"
    __table_args__ = (
        # foreign keys with two columns have to be written like this
        ForeignKeyConstraint(
            ["attribute_id", "class_id"],
            [f"{SCHEMA}.attribute_class.attribute_id", f"{SCHEMA}.attribute_class.class_id"],
        ),
        ForeignKeyConstraint(
            ["attribute_id", "label_rule_version"],
            [f"{SCHEMA}.label_rule.attribute_id", f"{SCHEMA}.label_rule.version"],
        ),
        {"schema": SCHEMA},
    )

    segment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("public.osm_edges.fid"), primary_key=True)
    attribute_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label_rule_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_id: Mapped[int | None] = mapped_column(Integer)
    value: Mapped[float | None] = mapped_column(Double)


class Split(Base):
    # A train/val/test split, method is e.g. "random" or "spatial"
    __tablename__ = "split"
    __table_args__ = {"schema": SCHEMA}

    split_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    method: Mapped[str] = mapped_column(Text)
    attribute_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.attribute.attribute_id"))


class SplitAssignment(Base):
    # Which partition (train, val or test) each segment is in for a split
    __tablename__ = "split_assignment"
    __table_args__ = {"schema": SCHEMA}

    split_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.split.split_id"), primary_key=True)
    segment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("public.osm_edges.fid"), primary_key=True)
    partition: Mapped[Partition] = mapped_column(partition_enum)
