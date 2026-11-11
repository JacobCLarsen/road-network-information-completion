import subprocess
from datetime import datetime, timezone

from sqlalchemy import insert, select

from src.db.models import Epoch, FeatureSet, FeatureSetFeature, Metric, ModelRun, Prediction, RunStatus
from src.db.repositories.base import BaseRepository


def get_git_commit():
    """Returns the current git commit hash, so we know which code a run was trained with."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip() or None
    except OSError:
        return None


class FeatureSetRepository(BaseRepository):
    model = FeatureSet

    async def get_or_create(self, name, version, features):
        """Returns the feature set with this name and version. If it does not exist yet it is created
        with the given list of feature names."""
        result = await self.session.execute(
            select(FeatureSet).where(FeatureSet.name == name, FeatureSet.version == version)
        )
        feature_set = result.scalars().first()
        if feature_set is not None:
            return feature_set

        feature_set = await self.create(name=name, version=version)
        for feature in features:
            self.session.add(FeatureSetFeature(feature_set_id=feature_set.feature_set_id, feature_name=feature))
        await self.session.flush()
        return feature_set


class ModelRunRepository(BaseRepository):
    model = ModelRun

    async def start_run(self, attribute_id, label_rule_version, split_id, feature_set_id, architecture, seed,
                        hyperparameters=None):
        """Saves a new run with status "running". Call finish_run or fail_run when training is done."""
        return await self.create(
            attribute_id=attribute_id,
            label_rule_version=label_rule_version,
            split_id=split_id,
            feature_set_id=feature_set_id,
            architecture=architecture,
            seed=seed,
            hyperparameters=hyperparameters,
            code_version=get_git_commit(),
            # the column has no time zone, so we always save the time in UTC
            started_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status=RunStatus.RUNNING,
        )

    async def finish_run(self, run_id):
        return await self.update(run_id, status=RunStatus.FINISHED)

    async def fail_run(self, run_id):
        return await self.update(run_id, status=RunStatus.FAILED)


class EpochRepository(BaseRepository):
    # Only needs the basic functions, e.g. create(run_id=1, epoch=1, train_loss=0.5)
    model = Epoch


class MetricRepository(BaseRepository):
    model = Metric

    async def save_metrics(self, run_id, partition, level, scores, class_id=None):
        """Saves a dict of scores, e.g. {"macro_f1": 0.7, "accuracy": 0.8}.
        Give class_id if the scores are for one class only."""
        for metric_name, value in scores.items():
            self.session.add(Metric(
                run_id=run_id,
                partition=partition,
                level=level,
                metric_name=metric_name,
                class_id=class_id,
                value=float(value),
            ))
        await self.session.flush()


class PredictionRepository(BaseRepository):
    model = Prediction

    async def save_predictions(self, run_id, segment_ids, predicted_class_ids, probabilities=None):
        """Saves the predicted class for every segment. The lists must be in the same order.
        probabilities is optional (one list of class probabilities per segment)."""
        if len(segment_ids) != len(predicted_class_ids):
            raise ValueError("segment_ids and predicted_class_ids must have the same length")

        rows = []
        for i in range(len(segment_ids)):
            rows.append({
                "run_id": run_id,
                # int() and float() because numpy numbers can not be sent to the database
                "segment_id": int(segment_ids[i]),
                "predicted_class_id": int(predicted_class_ids[i]),
                "probabilities": [float(p) for p in probabilities[i]] if probabilities is not None else None,
            })
        if rows:
            await self.session.execute(insert(Prediction), rows)
