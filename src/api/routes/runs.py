from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import EpochResponse, MetricResponse, RunDetailResponse, RunResponse
from src.db.engine import get_session
from src.db.repositories import EpochRepository, MetricRepository, ModelRunRepository

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/", response_model=list[RunResponse])
async def list_runs(session: AsyncSession = Depends(get_session)):
    """All the training runs, newest run first."""
    runs = await ModelRunRepository(session).get_all()
    runs.sort(key=lambda run: run.run_id, reverse=True)
    return runs


@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: int, session: AsyncSession = Depends(get_session)):
    """One run with its epochs (the training curve) and its metrics (the scores)."""
    run = await ModelRunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} was not found")

    epochs = await EpochRepository(session).get_all(run_id=run_id)
    epochs.sort(key=lambda epoch: epoch.epoch)
    metrics = await MetricRepository(session).get_all(run_id=run_id)

    # turn the database rows into the response classes
    return RunDetailResponse(
        **RunResponse.model_validate(run).model_dump(),
        epochs=[EpochResponse.model_validate(epoch) for epoch in epochs],
        metrics=[MetricResponse.model_validate(metric) for metric in metrics],
    )
