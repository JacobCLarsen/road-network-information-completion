# src/ml/data/road_graph_builder.py

import math
import torch

# The road types we want to predict. A segment's label is the position of its road type in this list.
# The labels themselves come from the database (made by src/db/create_road_type_labels.py), where link
# roads are already merged into their parent type, e.g. "primary_link" -> "primary".
ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "living_street"]

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
    label:    The road type from the database, e.g. "residential", or None if the segment has no label.
    oneway:   True if the road is one-way (from the database), or a oneway tag like "yes".

    The traffic from the taxi trips (both directions added together):
    traversal_count:  How many times a taxi drove on the segment.
    avg_speed:        The average speed in m/s, or None if there is no traffic.
    avg_acceleration: The average acceleration in m/s^2 when entering the segment, or None if we don't know it.
    speed_std:        The standard deviation of the speed in m/s (how much it changes), or None if there is no traffic.
    slow_share:       The share of traversals slower than 2 m/s (0 to 1), or None if there is no traffic.
    """

    def __init__(self, edge_ids, u, v, points, label=None, oneway=None,
                 traversal_count=0, avg_speed=None, avg_acceleration=None, speed_std=None, slow_share=None):
        self.edge_ids = edge_ids
        self.u = u
        self.v = v
        self.points = points
        self.label = label
        self.oneway = oneway
        self.traversal_count = traversal_count
        self.avg_speed = avg_speed
        self.avg_acceleration = avg_acceleration
        self.speed_std = speed_std
        self.slow_share = slow_share


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


def compute_raw_features(segment):
    """
    Returns the raw (unscaled) features of a segment: [log(length), curvature, one_way].
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

    # Rounding errors can push curvature slightly outside [0, 1], so curvature is clamped
    curvature = min(max(curvature, 0.0), 1.0)

    # One-way: 1 if the road is one-way, otherwise 0
    one_way = 0
    if str(segment.oneway).lower() in ONEWAY_VALUES:
        one_way = 1

    return [math.log(length), curvature, one_way]


def mean_and_std(values):
    """
    Returns the mean (mu) and standard deviation (sigma) of a list of numbers.
    """
    n = len(values)

    # An empty list can happen for the traffic features if no training segment has traffic.
    # Then mu = 0 and sigma = 1, so the values are not changed.
    if n == 0:
        return 0.0, 1.0

    total = 0.0
    for v in values:
        total = total + v
    mean = total / n

    squared_diff_sum = 0.0
    for v in values:
        squared_diff_sum = squared_diff_sum + (v - mean) ** 2
    std = math.sqrt(squared_diff_sum / n)

    # If every value is the same, sigma is 0; use 1 to avoid dividing by zero
    if std == 0.0:
        std = 1.0

    return mean, std


def fit_standardizer(train_segments, train_degrees):
    """
    Computes mu and sigma for log(length), curvature and degree over the training segments.
    one_way is not standardized, so it gets no mu or sigma.
    train_degrees holds the degree of each training segment, in the same order as train_segments.
    Returns (length_mean, length_std, curv_mean, curv_std, degree_mean, degree_std).
    """
    log_lengths = []
    curvatures = []
    for segment in train_segments:
        log_length, curvature, one_way = compute_raw_features(segment)
        log_lengths.append(log_length)
        curvatures.append(curvature)

    length_mean, length_std = mean_and_std(log_lengths)
    curv_mean, curv_std = mean_and_std(curvatures)
    degree_mean, degree_std = mean_and_std(train_degrees)

    return length_mean, length_std, curv_mean, curv_std, degree_mean, degree_std


def standardize(x, mean, std):
    """
    Applies z = (x - mu) / sigma to one feature value.
    """
    return (x - mean) / std


# z-score scaling is used to standardize input features.
def compute_features(segment, degree, length_mean, length_std, curv_mean, curv_std, degree_mean, degree_std):
    """
    Returns [z_log_length, z_curvature, one_way, z_degree]. one_way stays 0 or 1.
    """
    log_length, curvature, one_way = compute_raw_features(segment)
    return [
        standardize(log_length, length_mean, length_std),
        standardize(curvature, curv_mean, curv_std),
        # Oneway is not standardized, as it is binary
        one_way,
        standardize(degree, degree_mean, degree_std),
    ]


def known_traffic_values(segment):
    """
    Returns the traffic values that can be missing: [speed, acceleration, speed_std, slow_share].
    A value is None if we don't know it.
    """
    return [segment.avg_speed, segment.avg_acceleration, segment.speed_std, segment.slow_share]


def compute_raw_traffic_features(segment):
    """
    Returns the raw (unscaled) traffic features of a segment:
    [log(1 + traversals), has_traffic, speed, acceleration, speed_std, slow_share].
    log is used for the count because a few roads have a lot more traffic than the rest.
    If a value is unknown it is 0 here, and has_traffic tells the model that it is missing.
    """
    log_count = math.log(1 + segment.traversal_count)

    has_traffic = 0
    if segment.traversal_count > 0:
        has_traffic = 1

    features = [log_count, has_traffic]
    for value in known_traffic_values(segment):
        if value is None:
            features.append(0.0)
        else:
            features.append(value)
    return features


def fit_traffic_standardizer(train_segments):
    """
    Computes mu and sigma for the traffic features over the training segments.
    The count uses all training segments, but the other values only use the segments where we know them,
    otherwise all the missing ones (0) would pull the mean down.
    Returns a list of (mean, std) pairs: [count, speed, acceleration, speed_std, slow_share].
    has_traffic is not standardized, so it gets no mean or std.
    """
    log_counts = []
    # one list per value in known_traffic_values
    values = [[], [], [], []]
    for segment in train_segments:
        log_counts.append(math.log(1 + segment.traversal_count))
        known = known_traffic_values(segment)
        for k in range(len(known)):
            if known[k] is not None:
                values[k].append(known[k])

    stats = [mean_and_std(log_counts)]
    for value_list in values:
        stats.append(mean_and_std(value_list))
    return stats


def compute_traffic_features(segment, stats):
    """
    Returns [z_log_count, has_traffic, z_speed, z_acceleration, z_speed_std, z_slow_share]. has_traffic stays 0 or 1.
    stats is the list from fit_traffic_standardizer.
    A missing value gets z = 0, which is the same as the mean of the training segments.
    """
    raw = compute_raw_traffic_features(segment)
    count_mean, count_std = stats[0]
    has_traffic = raw[1]

    features = [standardize(raw[0], count_mean, count_std), has_traffic]
    known = known_traffic_values(segment)
    for k in range(len(known)):
        if known[k] is None:
            features.append(0.0)
        else:
            mean, std = stats[k + 1]
            features.append(standardize(known[k], mean, std))
    return features


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
            x:               Raw features of each segment (tensor, one row per segment), until
                             standardize_features is called. The 10 columns are:
                             log_length, curvature, one_way, degree,
                             log_traversals, has_traffic, avg_speed, avg_acceleration, speed_std, slow_share
                             add_neighbour_label_features adds one more column per road type after these.
            y:               Label of each segment, or -1 if its road type is not in classes
            mask:            True if the segment has a label, otherwise False
            edge_index:      The connected pairs of segments (tensor with 2 rows)
            seg_key:         (edge ID, start junction, end junction) for each segment
            geometry:        The list of points of each segment
            edge_to_segment: Dictionary from edge ID in the map to segment index
        """
        num_segments = len(segments)
        self.num_classes = len(classes)

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

            # If the segment has a label we use it, otherwise it gets -1 and is not used in training
            if segment.label in classes:
                y.append(classes.index(segment.label))
                mask.append(True)
            else:
                y.append(-1)
                mask.append(False)

            # Add the features
            x.append(compute_raw_features(segment))

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

        # Step 4: add the traffic features after the others
        for i in range(num_segments):
            x[i] = x[i] + compute_raw_traffic_features(segments[i])

        self.degree = degree

        # Convert to tensors for PyTorch
        self.x = torch.tensor(x, dtype=torch.float).reshape(num_segments, 10)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)

        # edge_index has two rows: the first row holds the start of each connection, the second row the end
        sources = []
        targets = []
        for (a, b) in sorted(connections):
            sources.append(a)
            targets.append(b)
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)

    def standardize_features(self, segments, train_mask):
        """
        Replaces the raw features in x with standardized ones. Called after the split, because mu and sigma come
        from the training segments only, so nothing is learned from the validation and test segments. They are then
        applied to all segments, because the model uses every segment's features as neighbours.

        x is updated in place, so a dictionary from get_static_components sees the new values.

        segments:   The same list of Segment objects the graph was built from.
        train_mask: True for the training segments.
        """
        train_indices = train_mask.nonzero().view(-1).tolist()
        train_segments = [segments[i] for i in train_indices]
        train_degrees = [self.degree[i] for i in train_indices]
        stats = fit_standardizer(train_segments, train_degrees)
        traffic_stats = fit_traffic_standardizer(train_segments)

        x = []
        for i in range(len(segments)):
            row = compute_features(segments[i], self.degree[i], *stats)
            row = row + compute_traffic_features(segments[i], traffic_stats)
            x.append(row)

        self.x.copy_(torch.tensor(x, dtype=torch.float))

    def add_neighbour_label_features(self, train_mask):
        """
        Adds one column per road type to x: the share of the segment's neighbours that are training segments
        of that road type. E.g. if a segment has 4 neighbours and 2 of them are residential training segments,
        the residential column is 0.5.

        Only the labels of the TRAINING segments are used, so the validation and test labels are never seen.
        A segment's own label is never used either, because a segment is not its own neighbour.
        The shares are already between 0 and 1, so they are not standardized.

        Call this after standardize_features, and only once. x gets more columns, so it is a new tensor:
        a dictionary from get_static_components must get the new x (graph["x"] = builder.x).

        train_mask: True for the training segments.
        """
        num_segments = self.x.shape[0]
        sources = self.edge_index[0].tolist()
        targets = self.edge_index[1].tolist()
        is_train = train_mask.tolist()
        labels = self.y.tolist()

        # counts[i][c] = how many neighbours of segment i are training segments of road type c
        counts = []
        for i in range(num_segments):
            counts.append([0.0] * self.num_classes)

        # every connection (a, b) means a is a neighbour of b
        for k in range(len(sources)):
            a = sources[k]
            b = targets[k]
            if is_train[a]:
                counts[b][labels[a]] = counts[b][labels[a]] + 1

        # turn the counts into shares. A segment without neighbours keeps all zeros
        for i in range(num_segments):
            if self.degree[i] > 0:
                for c in range(self.num_classes):
                    counts[i][c] = counts[i][c] / self.degree[i]

        self.x = torch.cat([self.x, torch.tensor(counts, dtype=torch.float)], dim=1)

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

