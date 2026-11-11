from sqlalchemy import delete, insert, select

from src.db.models import (
    Attribute,
    AttributeClass,
    LabelRule,
    Partition,
    Segment,
    SegmentLabel,
    Split,
    SplitAssignment,
)
from src.db.repositories.base import BaseRepository


class SegmentRepository(BaseRepository):
    # The road segments (read only). get_all() gives every segment, which is what we need to build the graph.
    model = Segment

    async def get_highway_tags(self):
        """Returns a dict {segment_id: highway tag} for all segments.
        Faster than get_all() because it does not load the geometry."""
        result = await self.session.execute(select(Segment.segment_id, Segment.highway))
        tags = {}
        for segment_id, highway in result.all():
            tags[segment_id] = highway
        return tags


class AttributeRepository(BaseRepository):
    model = Attribute

    async def get_by_name(self, name):
        """Find an attribute by its name, e.g. "road_type". Returns None if it does not exist."""
        result = await self.session.execute(select(Attribute).where(Attribute.name == name))
        return result.scalars().first()

    async def get_classes(self, attribute_id):
        """All the classes of an attribute, sorted by class_id."""
        result = await self.session.execute(
            select(AttributeClass)
            .where(AttributeClass.attribute_id == attribute_id)
            .order_by(AttributeClass.class_id)
        )
        return list(result.scalars().all())

    async def add_class(self, attribute_id, name, category=None):
        """Adds a new class to an attribute, e.g. add_class(road_type_id, "primary", "main_roads")."""
        new_class = AttributeClass(attribute_id=attribute_id, name=name, category=category)
        self.session.add(new_class)
        await self.session.flush()
        return new_class


class LabelRepository(BaseRepository):
    model = SegmentLabel

    async def get_or_create_rule(self, attribute_id, version, description=None):
        """Returns the label rule with this version, and creates it if it does not exist yet."""
        rule = await self.session.get(LabelRule, (attribute_id, version))
        if rule is None:
            rule = LabelRule(attribute_id=attribute_id, version=version, description=description)
            self.session.add(rule)
            await self.session.flush()
        return rule

    async def save_labels(self, attribute_id, label_rule_version, labels):
        """Saves the labels from a dict {segment_id: class_id}.
        Old labels for the same attribute and version are deleted first, so it is safe to run again."""
        await self.session.execute(
            delete(SegmentLabel).where(
                SegmentLabel.attribute_id == attribute_id,
                SegmentLabel.label_rule_version == label_rule_version,
            )
        )
        rows = []
        for segment_id, class_id in labels.items():
            rows.append({
                "segment_id": int(segment_id),
                "attribute_id": attribute_id,
                "label_rule_version": label_rule_version,
                "class_id": int(class_id),
            })
        if rows:
            await self.session.execute(insert(SegmentLabel), rows)

    async def get_labels(self, attribute_id, label_rule_version):
        """Returns a dict {segment_id: class_id} with the true label of every labelled segment."""
        result = await self.session.execute(
            select(SegmentLabel.segment_id, SegmentLabel.class_id).where(
                SegmentLabel.attribute_id == attribute_id,
                SegmentLabel.label_rule_version == label_rule_version,
            )
        )
        labels = {}
        for segment_id, class_id in result.all():
            labels[segment_id] = class_id
        return labels


class SplitRepository(BaseRepository):
    model = Split

    async def create_split(self, method, assignments, attribute_id=None):
        """Saves a new split. assignments is a dict {segment_id: "train" / "val" / "test"}."""
        split = await self.create(method=method, attribute_id=attribute_id)
        rows = []
        for segment_id, partition in assignments.items():
            rows.append({"split_id": split.split_id, "segment_id": int(segment_id), "partition": Partition(partition)})
        if rows:
            # one insert for all rows (sqlalchemy sends them in batches) instead of one insert per row
            await self.session.execute(insert(SplitAssignment), rows)
        return split

    async def get_partitions(self, split_id):
        """Returns {"train": [segment ids], "val": [...], "test": [...]} for a split."""
        result = await self.session.execute(
            select(SplitAssignment.segment_id, SplitAssignment.partition)
            .where(SplitAssignment.split_id == split_id)
            .order_by(SplitAssignment.segment_id)
        )
        partitions = {"train": [], "val": [], "test": []}
        for segment_id, partition in result.all():
            partitions[partition.value].append(segment_id)
        return partitions
