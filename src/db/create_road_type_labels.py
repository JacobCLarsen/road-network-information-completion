"""Fills the database with the road type classes and a label for every segment.

It is safe to run more than once. Existing classes are kept and the labels of label rule
version 1 are deleted and saved again.

The ML code reads these labels from the database (src/ml/data/database_loader.py), so this is the
rule for the labels we train on:
  - the label is the highway tag of the segment
  - link roads are merged into their parent type, e.g. "primary_link" becomes "primary"
  - all other tags (rare types, or segments with more than one tag) get no label
If you change the rule, use a new LABEL_RULE_VERSION instead of changing version 1.
"""

import asyncio

from src.db.engine import engine, session_scope
from src.db.repositories import AttributeRepository, LabelRepository, SegmentRepository

# Chang the rule version to add a new labeling rule. Always increment by one
LABEL_RULE_VERSION = 1
LABEL_RULE_DESCRIPTION = "highway tag, link roads merged into parent type, other tags not labelled"

# The names must be the same as ROAD_CLASSES in src/ml/data/road_graph_builder.py, and the categories
# the same as ROAD_GROUPS in src/ml/evaluation.py.
ROAD_CLASSES = {
    "motorway": "highways",
    "trunk": "highways",
    "primary": "main_roads",
    "secondary": "main_roads",
    "tertiary": "main_roads",
    "unclassified": "minor_roads",
    "residential": "minor_roads",
    "living_street": "minor_roads",
}

LINK_ROAD_MAPPINGS = {
    "motorway_link": "motorway",
    "trunk_link": "trunk",
    "primary_link": "primary",
    "secondary_link": "secondary",
    "tertiary_link": "tertiary",
}


def road_type_from_tag(highway):
    """Returns the road type for a highway tag, or None if the segment should not get a label."""
    if highway in LINK_ROAD_MAPPINGS:
        highway = LINK_ROAD_MAPPINGS[highway]
    if highway in ROAD_CLASSES:
        return highway
    return None


async def create_road_type_labels():
    async with session_scope() as session:
        attributes = AttributeRepository(session)
        label_repo = LabelRepository(session)

        # The road_type attribute is made by the migrations in the data repo
        road_type = await attributes.get_by_name("road_type")
        if road_type is None:
            raise RuntimeError("The attribute road_type does not exist. Run the migrations in the data repo first.")

        # Add the classes that are missing
        existing = await attributes.get_classes(road_type.attribute_id)
        class_ids = {}
        for road_class in existing:
            class_ids[road_class.name] = road_class.class_id
        for name, category in ROAD_CLASSES.items():
            if name not in class_ids:
                new_class = await attributes.add_class(road_type.attribute_id, name, category)
                class_ids[name] = new_class.class_id
        print(f"Road type classes: {class_ids}")

        await label_repo.get_or_create_rule(road_type.attribute_id, LABEL_RULE_VERSION, LABEL_RULE_DESCRIPTION)

        # Find the label of every segment
        highway_tags = await SegmentRepository(session).get_highway_tags()
        labels = {}
        for segment_id, highway in highway_tags.items():
            road_type_name = road_type_from_tag(highway)
            if road_type_name is not None:
                labels[segment_id] = class_ids[road_type_name]

        print(f"Saving {len(labels)} labels ({len(highway_tags) - len(labels)} of {len(highway_tags)} segments have no label)")
        await label_repo.save_labels(road_type.attribute_id, LABEL_RULE_VERSION, labels)

    print("Done")


async def main():
    try:
        await create_road_type_labels()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
