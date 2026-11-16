from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.create_road_type_labels import LABEL_RULE_VERSION
from src.db.engine import get_session
from src.db.repositories import AttributeRepository, SegmentRepository

router = APIRouter(prefix="/map", tags=["map"])

# Most segments we send in one answer. Zoomed out the whole city is too much for the browser.
MAX_SEGMENTS = 20000


async def get_road_type(session):
    road_type = await AttributeRepository(session).get_by_name("road_type")
    if road_type is None:
        raise HTTPException(status_code=404, detail="The attribute road_type does not exist")
    return road_type


@router.get("/road-types")
async def get_road_types(session: AsyncSession = Depends(get_session)):
    """The road type classes, so the front end can turn a class_id into a name."""
    road_type = await get_road_type(session)
    classes = await AttributeRepository(session).get_classes(road_type.attribute_id)
    return [{"class_id": c.class_id, "name": c.name, "category": c.category} for c in classes]


@router.get("/missing")
async def get_missing_segments(run_id: int | None = None, session: AsyncSession = Depends(get_session)):
    """All segments without a road type label (the missing values) as GeoJSON.
    With run_id every segment also gets the road type that run predicted for it."""
    road_type = await get_road_type(session)
    features = await SegmentRepository(session).get_map_features(
        road_type.attribute_id, LABEL_RULE_VERSION, run_id=run_id, only_missing=True, limit=MAX_SEGMENTS
    )
    return {"type": "FeatureCollection", "features": features}


@router.get("/segments")
async def get_segments(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    run_id: int | None = None,
    session: AsyncSession = Depends(get_session),
):
    """All segments inside the box (the part of the map you look at) as GeoJSON, with their true road type
    (null if missing) and, with run_id, the predicted road type. At most MAX_SEGMENTS are sent."""
    road_type = await get_road_type(session)
    features = await SegmentRepository(session).get_map_features(
        road_type.attribute_id, LABEL_RULE_VERSION, run_id=run_id,
        bbox=(min_lon, min_lat, max_lon, max_lat), limit=MAX_SEGMENTS,
    )
    return {"type": "FeatureCollection", "features": features, "truncated": len(features) == MAX_SEGMENTS}
