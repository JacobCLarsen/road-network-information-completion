from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Double, ForeignKey, ForeignKeyConstraint, Identity, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import (
    SCHEMA,
    Base,
    MetricLevel,
    Partition,
    RunStatus,
    metric_level_enum,
    partition_enum,
    run_status_enum,
)


class FeatureSet(Base):
    # A list of input features with a name and a version, e.g. "geometry" version 1
    __tablename__ = "feature_set"
    __table_args__ = {"schema": SCHEMA}

    feature_set_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)


class FeatureSetFeature(Base):
    # One feature in a feature set
    __tablename__ = "feature_set_feature"
    __table_args__ = {"schema": SCHEMA}

    feature_set_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.feature_set.feature_set_id"), primary_key=True)
    feature_name: Mapped[str] = mapped_column(Text, primary_key=True)


class ModelRun(Base):
    # One training run of a model. Saves everything we need to run it again
    # (which split, features, seed, hyperparameters and git commit).
    # When a run is deleted the database also deletes its epochs, metrics and predictions.
    __tablename__ = "model_run"
    __table_args__ = (
        ForeignKeyConstraint(
            ["attribute_id", "label_rule_version"],
            [f"{SCHEMA}.label_rule.attribute_id", f"{SCHEMA}.label_rule.version"],
        ),
        {"schema": SCHEMA},
    )

    run_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    attribute_id: Mapped[int] = mapped_column(Integer)
    label_rule_version: Mapped[int] = mapped_column(Integer)
    split_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.split.split_id"))
    feature_set_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.feature_set.feature_set_id"))
    architecture: Mapped[str] = mapped_column(Text)
    hyperparameters: Mapped[dict | None] = mapped_column(JSONB)
    seed: Mapped[int] = mapped_column(Integer)
    code_version: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)  # in UTC
    status: Mapped[RunStatus] = mapped_column(run_status_enum)


class Epoch(Base):
    # The losses and validation score after each epoch (the training curve)
    __tablename__ = "epoch"
    __table_args__ = {"schema": SCHEMA}

    run_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.model_run.run_id"), primary_key=True)
    epoch: Mapped[int] = mapped_column(Integer, primary_key=True)
    train_loss: Mapped[float | None] = mapped_column(Double)
    val_loss: Mapped[float | None] = mapped_column(Double)
    val_macro_f1: Mapped[float | None] = mapped_column(Double)


class Metric(Base):
    # A score of a run, e.g. macro_f1 on the test set.
    # class_id is empty for overall scores and set for scores of one class.
    __tablename__ = "metric"
    __table_args__ = {"schema": SCHEMA}

    metric_id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.model_run.run_id"))
    partition: Mapped[Partition] = mapped_column(partition_enum)
    level: Mapped[MetricLevel] = mapped_column(metric_level_enum)
    metric_name: Mapped[str] = mapped_column(Text)
    class_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.attribute_class.class_id"))
    value: Mapped[float] = mapped_column(Double)


class Prediction(Base):
    # What a run predicted for each segment
    __tablename__ = "prediction"
    __table_args__ = {"schema": SCHEMA}

    run_id: Mapped[int] = mapped_column(ForeignKey(f"{SCHEMA}.model_run.run_id"), primary_key=True)
    segment_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("public.osm_edges.fid"), primary_key=True)
    predicted_class_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SCHEMA}.attribute_class.class_id"))
    predicted_value: Mapped[float | None] = mapped_column(Double)
    probabilities: Mapped[list[float] | None] = mapped_column(ARRAY(Double))
