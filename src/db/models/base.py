import enum

from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import DeclarativeBase

# Our tables are in this schema. The raw dataset tables (osm_edges etc.) are in "public".
SCHEMA = "attribute_prediction"


class Base(DeclarativeBase):
    # All our table classes inherit from this.
    # NOTE: the tables are made by the migrations in the data repo, we never create them from here.
    pass


# The enum types that exist in the database. create_type=False because they already exist.

class TaskType(enum.StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Partition(enum.StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class MetricLevel(enum.StrEnum):
    FINE = "fine"  # the 8 road types
    COARSE = "coarse"  # the 3 road categories


def _values(enum_class):
    # store the value ("train") in the database and not the name ("TRAIN")
    return [member.value for member in enum_class]


task_type_enum = ENUM(TaskType, name="task_type", schema=SCHEMA, create_type=False, values_callable=_values)
partition_enum = ENUM(Partition, name="partition", schema=SCHEMA, create_type=False, values_callable=_values)
run_status_enum = ENUM(RunStatus, name="run_status", schema=SCHEMA, create_type=False, values_callable=_values)
metric_level_enum = ENUM(MetricLevel, name="metric_level", schema=SCHEMA, create_type=False, values_callable=_values)
