import json

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import aliased

from src.db.models import (
    Attribute,
    AttributeClass,
    LabelRule,
    OsmEdge,
    Partition,
    Prediction,
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

    async def get_map_features(self, attribute_id, label_rule_version, run_id=None, bbox=None, only_missing=False,
                               limit=20000):
        """Segments for the map as GeoJSON features. Every feature has:
          - label: the true class name, or None if the segment has no label (a missing value)
          - predicted: the class the run predicted (only if run_id is given)
        bbox is (min_lon, min_lat, max_lon, max_lat). only_missing=True gives only the segments without a label."""
        # attribute_class is used twice (true class and predicted class), so we need two aliases
        label_class = aliased(AttributeClass)
        predicted_class = aliased(AttributeClass)

        query = (
            select(OsmEdge.fid, OsmEdge.name, func.ST_AsGeoJSON(OsmEdge.geometry), label_class.name)
            .outerjoin(SegmentLabel, (SegmentLabel.segment_id == OsmEdge.fid)
                       & (SegmentLabel.attribute_id == attribute_id)
                       & (SegmentLabel.label_rule_version == label_rule_version))
            .outerjoin(label_class, label_class.class_id == SegmentLabel.class_id)
            .where(OsmEdge.geometry.is_not(None))
            .order_by(OsmEdge.fid)
            .limit(limit)
        )
        if run_id is not None:
            query = (
                query.add_columns(predicted_class.name)
                .outerjoin(Prediction, (Prediction.segment_id == OsmEdge.fid) & (Prediction.run_id == run_id))
                .outerjoin(predicted_class, predicted_class.class_id == Prediction.predicted_class_id)
            )
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            # uses the spatial index on osm_edges, so it is fast
            query = query.where(func.ST_Intersects(OsmEdge.geometry, func.ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)))
        if only_missing:
            query = query.where(SegmentLabel.segment_id.is_(None))

        result = await self.session.execute(query)
        features = []
        for row in result.all():
            properties = {"segment_id": row[0], "name": row[1], "label": row[3]}
            if run_id is not None:
                properties["predicted"] = row[4]
            features.append({"type": "Feature", "geometry": json.loads(row[2]), "properties": properties})
        return features


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
