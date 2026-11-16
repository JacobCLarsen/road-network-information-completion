"""What the API sends back. FastAPI uses these classes to check the responses and to make the /docs page."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.db.models import MetricLevel, Partition, RunStatus


class EpochResponse(BaseModel):
    # from_attributes lets us make it straight from a database row
    model_config = ConfigDict(from_attributes=True)

    epoch: int
    train_loss: float | None
    val_loss: float | None
    val_macro_f1: float | None


class MetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    partition: Partition
    level: MetricLevel
    metric_name: str
    class_id: int | None
    value: float


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    attribute_id: int
    label_rule_version: int
    split_id: int
    feature_set_id: int
    architecture: str
    hyperparameters: dict | None
    seed: int
    code_version: str | None
    started_at: datetime | None
    status: RunStatus


class RunDetailResponse(RunResponse):
    # a run together with its training curve and its scores
    epochs: list[EpochResponse]
    metrics: list[MetricResponse]
