# Data layer (`src/db`)

This is the code we use to talk to the database, both from the API and from the ML scripts.
The database is PostGIS and the tables are made by the migrations in the
[astep-aau/data](https://github.com/astep-aau/data) repo. We never create tables from this repo.

```
src/db/
├── config.py       # reads DATABASE_URL from the environment or .env
├── engine.py       # the connection, session_scope(), get_session() and check_database()
├── exceptions.py   # NotFoundError
├── models/         # one class per table, they have to match the migrations
└── repositories/   # the functions we use to read and write data
```

## Connecting

Everything is set with `DATABASE_URL` (see `.env.example`):

| Where | DATABASE_URL |
|---|---|
| Local (docker in the data repo) | `postgresql+asyncpg://postgres:postgres@localhost:5432/data` |
| On the server (k8s secret `db-secret`) | `postgresql+asyncpg://USER:PASSWORD@dataset-postgres.main.svc.cluster.local:5432/data` |
| Server database from your own machine | run `kubectl port-forward svc/dataset-postgres -n main 5433:5432` and use port 5433 |

When the API starts it checks that the database has run the migration in `EXPECTED_MIGRATION`
(`config.py`). When the data repo gets new migrations, update the models and that line.

To check that a connection works (local or server), run:

```bash
python -m src.db.check_connection
```

### Using the server database from your own machine

1. Get a kubeconfig for the cluster from the group and check it works: `kubectl get svc -n main`
2. Forward the database to your machine (keep this terminal open):
   `kubectl port-forward svc/dataset-postgres -n main 5433:5432`
3. Set `DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5433/data` in `.env`
   (the user and password are in the `postgres-secret` secret in namespace `main`)
4. Run `python -m src.db.check_connection`

The API on the server uses the same database through the `db-secret` secret (see `k8s/secret.example.yaml`).

## Labels

The road type classes and the label of every segment are saved in the database by a script.
Run it once on every new database (it is safe to run again):

```bash
python -m src.db.create_road_type_labels
```

It uses the same rule as the ML code: the highway tag, with link roads merged into their parent
type. Segments with other tags get no label. The class ids are made by the database, so they are
not the same as the index in `ROAD_CLASSES`. Always look classes up by name.

## How to use it

The repositories never commit. `session_scope()` commits at the end of the with-block, and if an
error happens it rolls everything back, so a training run is saved completely or not at all.

```python
import asyncio
from src.db.engine import session_scope
from src.db.models import MetricLevel, Partition
from src.db.repositories import AttributeRepository, MetricRepository, ModelRunRepository, PredictionRepository

async def save_results(split_id, feature_set_id, scores, segment_ids, predictions):
    async with session_scope() as session:
        attribute = await AttributeRepository(session).get_by_name("road_type")
        run = await ModelRunRepository(session).start_run(
            attribute_id=attribute.attribute_id, label_rule_version=1, split_id=split_id,
            feature_set_id=feature_set_id, architecture="graphsage", seed=0, hyperparameters={"lr": 0.01},
        )
        await MetricRepository(session).save_metrics(run.run_id, Partition.TEST, MetricLevel.FINE, scores)
        await PredictionRepository(session).save_predictions(run.run_id, segment_ids, predictions)
        await ModelRunRepository(session).finish_run(run.run_id)

asyncio.run(save_results(...))
```

In a FastAPI route use `session: AsyncSession = Depends(get_session)` instead.

## Functions

All repositories have the basic CRUD functions: `get(id)`, `get_all(**filters)`, `create(**values)`,
`update(id, **values)` and `delete(id)`. On top of that:

| Repository | Functions |
|---|---|
| `SegmentRepository` | `get_highway_tags` (read only, `get_all()` gives all segments) |
| `AttributeRepository` | `get_by_name`, `get_classes`, `add_class` |
| `LabelRepository` | `get_labels`, `save_labels`, `get_or_create_rule` |
| `SplitRepository` | `create_split`, `get_partitions` |
| `FeatureSetRepository` | `get_or_create` |
| `ModelRunRepository` | `start_run`, `finish_run`, `fail_run` |
| `EpochRepository` | only the basic ones |
| `MetricRepository` | `save_metrics` |
| `PredictionRepository` | `save_predictions` |

We only added the functions we need right now. New ones can be added when we need them.

## Tests

```bash
pytest tests/db/unit           # no database needed
pytest tests/db/integration    # needs TEST_DATABASE_URL in .env
```

The integration tests roll back everything they do, so they do not change your database.
`test_schema.py` checks that the models still match the real tables.
