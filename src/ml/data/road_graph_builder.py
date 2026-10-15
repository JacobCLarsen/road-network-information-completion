# src/ml/data/road_graph_builder.py

import math
import torch

# The road types we want to predict. A segment's label is the position of its road type in this list.
ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "motorway_link", "trunk_link", "primary_link", "secondary_link",
                "tertiary_link", "living_street", "service", "pedestrian", "track", "bus_guideway",
                "escape", "raceway", "road", "busway"]

# Values of the oneway tag that mean the road is one-way
ONEWAY_VALUES = ["yes", "true", "1", "-1"]

EARTH_RADIUS_M = 6371000.0

# Smallest length a segment can have (one millimeter). Without it, a segment of length 0 would give
# log(0) and a division by zero when computing the features.
MIN_LENGTH_M = 0.001


class Segment:
    """
    A road between two junctions.

    edge_ids: The IDs of the edge records in the map that this segment comes from. A one-way road has
              one ID, and a two-way road has two (one for each direction). The traffic data uses these IDs.
    u, v:     The IDs of the start and end junction.
    points:   The shape of the road as a list of (longitude, latitude) points, from u to v.
    highway:  The road type, e.g. "residential".
    oneway:   The oneway tag, e.g. "True" or "False".
    """

    def __init__(self, edge_ids, u, v, points, highway=None, oneway=None):
        self.edge_ids = edge_ids
        self.u = u
        self.v = v
        self.points = points
        self.highway = highway
        self.oneway = oneway


def haversine(point_a, point_b):
    """
    Returns the distance in meters between two (longitude, latitude) points, measured along the surface
    of the Earth. Calculated using the haversine formula.
    """
    lon1 = math.radians(point_a[0])
    lat1 = math.radians(point_a[1])
    lon2 = math.radians(point_b[0])
    lat2 = math.radians(point_b[1])

    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def compute_features(segment):
    """
    Returns the features of a segment: [log(length), curvature, one_way].
    """
    # Length along the road: add up the distances between consecutive points
    length = 0.0
    for k in range(len(segment.points) - 1):
        length = length + haversine(segment.points[k], segment.points[k + 1])

    if length < MIN_LENGTH_M:
        length = MIN_LENGTH_M

    # Curvature: 0 for a straight road, close to 1 for a road that winds a lot (exactly 1 for a loop)
    first_point = segment.points[0]
    last_point = segment.points[len(segment.points) - 1]
    straight_distance = haversine(first_point, last_point)
    curvature = 1 - straight_distance / length

    # Rounding errors can push curvature slightly outside [0, 1] (e.g. -2e-16 for a straight road), so curvature is clamped.
    curvature = min(max(curvature, 0.0), 1.0)

    # One-way: 1 if the road is one-way, otherwise 0
    one_way = 0
    if str(segment.oneway).lower() in ONEWAY_VALUES:
        one_way = 1

    return [math.log(length), curvature, one_way]


def add_to_junction(segments_at_junction, junction, segment_index):
    """Adds a segment to the list of segments at a junction, creating the list if it does not exist yet."""
    if junction not in segments_at_junction:
        segments_at_junction[junction] = []
    segments_at_junction[junction].append(segment_index)


class RoadGraphBuilder:
    def __init__(self, segments, classes=ROAD_CLASSES):
        """
        Builds the line graph of the road network: every road segment becomes a vertex, and two segments
        are connected by an edge if they meet at a junction.

        segments: List of Segment objects, already split at junctions.
        classes:  List of road types used as labels.

        After building, the results are stored in:
            x:               Features of each segment (tensor, one row per segment)
            y:               Label of each segment, or -1 if its road type is not in classes
            mask:            True if the segment has a label, otherwise False
            edge_index:      The connected pairs of segments (tensor with 2 rows)
            seg_key:         (edge ID, start junction, end junction) for each segment
            geometry:        The list of points of each segment
            edge_to_segment: Dictionary from edge ID in the map to segment index
        """
        num_segments = len(segments)

        # Step 1: set labels and compute features
        x = []
        y = []
        mask = []
        self.seg_key = []
        self.geometry = []
        self.edge_to_segment = {}

        # For each junction, the list of segments that start or end there
        segments_at_junction = {}

        for i in range(num_segments):
            segment = segments[i]

            self.seg_key.append((segment.edge_ids[0], segment.u, segment.v))
            self.geometry.append(segment.points)
            for edge_id in segment.edge_ids:
                # Check if the edge ID already exists in the list.
                assert edge_id not in self.edge_to_segment, f"Edge ID {edge_id} is used by more than one segment"
                self.edge_to_segment[edge_id] = i

            # If the segment has a valid label, use it, otherwise it is not used in training
            if segment.highway in classes:
                y.append(classes.index(segment.highway))
                mask.append(True)
            else:
                y.append(-1)
                mask.append(False)

            # Add the features
            x.append(compute_features(segment))

            # Remember which junctions the segment touches
            add_to_junction(segments_at_junction, segment.u, i)
            if segment.v != segment.u:
                add_to_junction(segments_at_junction, segment.v, i)

        # Step 2: line graph transformation.
        # Connect every pair of segments that touch the same junction. A set is used so that two segments
        # sharing both junctions are only connected once.
        connections = set()
        for junction in segments_at_junction:
            segments_here = segments_at_junction[junction]
            for a in segments_here:
                for b in segments_here:
                    if a != b:
                        connections.add((a, b))

        # Step 3: add the number of connections (the degree) as a feature
        degree = [0] * num_segments
        for (a, b) in connections:
            degree[a] = degree[a] + 1

        for i in range(num_segments):
            x[i].append(degree[i])

        # Convert to tensors for PyTorch
        self.x = torch.tensor(x, dtype=torch.float).reshape(num_segments, 4)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)

        # edge_index has two rows: the first row holds the start of each connection, the second row the end
        sources = []
        targets = []
        for (a, b) in sorted(connections):
            sources.append(a)
            targets.append(b)
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)

    def get_static_components(self):
        """
        Returns all parts of the graph in a dictionary.
        """
        return {
            'x': self.x,
            'y': self.y,
            'mask': self.mask,
            'edge_index': self.edge_index,
            'seg_key': self.seg_key,
            'geometry': self.geometry,
            'edge_to_segment': self.edge_to_segment,
        }

