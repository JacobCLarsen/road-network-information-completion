import pytest

from src.db.models import OsmEdge, Prediction
from src.db.repositories import AttributeRepository
from tests.api.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")

# A fake segment far away from Beijing, so it is the only one in its box
FAKE_ID = 9000000010
LINE = "SRID=4326;LINESTRING(-170 -80, -169.9 -79.9)"
BOX = "min_lon=-171&min_lat=-81&max_lon=-169&max_lat=-79"


async def test_road_types(db_client):
    response = await db_client.get("/map/road-types")
    assert response.status_code == 200
    assert all("class_id" in road_type and "name" in road_type for road_type in response.json())


async def test_segment_without_label_is_missing(db_client, session):
    session.add(OsmEdge(fid=FAKE_ID, highway="service", geometry=LINE))
    await session.flush()

    # it has no label, so it is in the missing segments
    missing = (await db_client.get("/map/missing")).json()["features"]
    assert FAKE_ID in [feature["properties"]["segment_id"] for feature in missing]

    # and it is the only segment in the box, with label null
    data = (await db_client.get(f"/map/segments?{BOX}")).json()
    assert data["truncated"] is False
    assert len(data["features"]) == 1
    feature = data["features"][0]
    assert feature["properties"] == {"segment_id": FAKE_ID, "name": None, "label": None}
    assert feature["geometry"]["type"] == "LineString"


async def test_segments_with_prediction(db_client, session, run):
    session.add(OsmEdge(fid=FAKE_ID, highway="service", geometry=LINE))
    await session.flush()
    # the run in the fixture is for a test attribute, but the prediction only needs a class id
    road_type = await AttributeRepository(session).get_by_name("road_type")
    classes = await AttributeRepository(session).get_classes(road_type.attribute_id)
    if not classes:
        pytest.skip("No road type classes, run python -m src.db.create_road_type_labels")
    session.add(Prediction(run_id=run.run_id, segment_id=FAKE_ID, predicted_class_id=classes[0].class_id))
    await session.flush()

    data = (await db_client.get(f"/map/segments?{BOX}&run_id={run.run_id}")).json()
    assert data["features"][0]["properties"]["predicted"] == classes[0].name
