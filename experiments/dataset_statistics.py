# experiments/dataset_statistics.py
#
# Computes every dataset figure used in the report (Section 4.3.1 Feature Selection and the class
# distribution in Section 4.3.3) directly from the Beijing data, so the numbers can be checked and
# reproduced.
#
# Run from the repository root:
#     python -m experiments.dataset_statistics --data-dir data/bj_data
#     python -m experiments.dataset_statistics --data-dir data/bj_data --traj-file traj_train_100.csv

#       hvis bj_data bare er i dit downloads folder så kør python -m experiments.dataset_statistics --data-dir ..\bj_data fra repo rooten
#
# The data directory must contain map/edges.* and the trajectory CSV files.

import argparse
import ast
import csv
import json
import math
import os
import statistics

import shapefile

from src.ml.data.file_loader import ShapefileMapLoader
from src.ml.data.road_graph_builder import compute_features

# The main road types, from the top of the road hierarchy to the bottom. Link types are counted under
# their parent type, e.g. motorway_link counts as motorway.
MAIN_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
              "residential", "living_street"]

# The trajectory CSVs contain very long fields (the matched path of a whole trip)
csv.field_size_limit(2 ** 31 - 1)


def parent_type(highway):
    """Returns the road type with the _link suffix removed, e.g. "motorway_link" -> "motorway"."""
    return str(highway).replace("_link", "")


def is_filled(value):
    """True if a tag has a value. Missing tags are stored as empty strings in the shapefile."""
    return str(value).strip() not in ("", "None", "nan")


def percent(part, total):
    if total == 0:
        return 0.0
    return 100.0 * part / total


def skewness(values):
    """Sample skewness: 0 for a symmetric distribution, large and positive for a long right tail."""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return sum(((v - mean) / std) ** 3 for v in values) / len(values)


def print_header(title):
    print()
    print(title)
    print("-" * len(title))


def load_edge_records(map_dir):
    """Returns a dictionary from edge ID (fid) to the tags of that edge record."""
    reader = shapefile.Reader(os.path.join(map_dir, "edges"), encoding="utf-8")
    records = {}
    for record in reader.records():
        tags = record.as_dict()
        records[int(tags["fid"])] = tags
    return records


def segment_has_tag(segment, records, tag):
    """A segment has a tag if at least one of its edge records (one per direction) has it."""
    for edge_id in segment.edge_ids:
        if is_filled(records[int(edge_id)][tag]):
            return True
    return False


def map_statistics(records, segments):
    print_header("Road network")
    print(f"Edge records in edges.shp:              {len(records):,}")
    print(f"Road segments (two directions merged):  {len(segments):,}")

    # Class distribution over edge records, as in Figure 4 (Section 4.3.3)
    print_header("Road types over edge records (Figure 4)")
    counts = {}
    for tags in records.values():
        counts[tags["highway"]] = counts.get(tags["highway"], 0) + 1
    for highway, count in sorted(counts.items(), key=lambda item: -item[1]):
        if count >= 100:
            print(f"  {highway:32s} {count:8,}  {percent(count, len(records)):5.1f}%")
    print(f"  ({sum(1 for c in counts.values() if c < 100)} rarer values with fewer than 100 records not shown)")

    service_records = counts.get("service", 0)
    print(f"Records with highway = service:         {service_records}")

    # Group the segments by their parent road type. Segments whose highway tag holds several road
    # types do not match any of MAIN_TYPES and are left out of the per-type tables.
    by_type = {road_type: [] for road_type in MAIN_TYPES}
    for segment in segments:
        road_type = parent_type(segment.highway)
        if road_type in by_type:
            by_type[road_type].append(segment)

    # One-way, lanes and speed limit per road type, counted over segments
    print_header("Tags per road type (segments, link types merged into parent type)")
    print(f"  {'road type':15s} {'segments':>9s} {'one-way':>8s} {'lanes':>7s} {'maxspeed':>9s}")
    for road_type in MAIN_TYPES:
        segs = by_type[road_type]
        one_way = sum(compute_features(s)[2] for s in segs)
        lanes = sum(segment_has_tag(s, records, "lanes") for s in segs)
        maxspeed = sum(segment_has_tag(s, records, "maxspeed") for s in segs)
        print(f"  {road_type:15s} {len(segs):9,} {percent(one_way, len(segs)):7.1f}% "
              f"{percent(lanes, len(segs)):6.1f}% {percent(maxspeed, len(segs)):8.1f}%")

    # Missing tags over all segments
    print_header("Missing tags (all segments)")
    for tag in ["oneway", "lanes", "maxspeed", "ref", "name", "bridge", "tunnel"]:
        filled = sum(segment_has_tag(s, records, tag) for s in segments)
        print(f"  {tag:10s} missing for {percent(len(segments) - filled, len(segments)):5.1f}% of segments")

    # Road types of segments with a 120 km/h speed limit
    print_header("Road types of segments with maxspeed = 120")
    types_120 = {}
    for segment in segments:
        for edge_id in segment.edge_ids:
            if str(records[int(edge_id)]["maxspeed"]).strip() == "120":
                road_type = parent_type(segment.highway)
                types_120[road_type] = types_120.get(road_type, 0) + 1
                break
    for road_type, count in sorted(types_120.items(), key=lambda item: -item[1]):
        print(f"  {road_type:15s} {count:6,}")

    # Road types of segments whose route number (ref) starts with G
    print_header("Road types of segments whose ref starts with G")
    types_g = {}
    for segment in segments:
        refs = [str(records[int(e)]["ref"]).strip() for e in segment.edge_ids]
        if any(ref.startswith("G") for ref in refs):
            road_type = parent_type(segment.highway)
            types_g[road_type] = types_g.get(road_type, 0) + 1
    total_g = sum(types_g.values())
    for road_type, count in sorted(types_g.items(), key=lambda item: -item[1]):
        print(f"  {road_type:25s} {count:6,}  {percent(count, total_g):5.1f}%")

    # Why log(length) is used: skewness of length before and after the logarithm. Standardization
    # does not change skewness, since it only shifts and scales the values.
    print_header("Length distribution (segments)")
    lengths = []
    for segment in segments:
        log_length = compute_features(segment)[0]
        lengths.append(math.exp(log_length))
    log_lengths = [math.log(length) for length in lengths]
    quartiles = statistics.quantiles(lengths, n=100)
    print(f"  median length:          {statistics.median(lengths):,.1f} m")
    print(f"  1st / 99th percentile:  {quartiles[0]:,.1f} m / {quartiles[98]:,.1f} m")
    print(f"  max length:             {max(lengths):,.1f} m")
    print(f"  skewness of length:     {skewness(lengths):.2f}")
    print(f"  skewness of log(length): {skewness(log_lengths):.2f}")
    mean = statistics.fmean(lengths)
    std = statistics.pstdev(lengths)
    print(f"  standardized length ranges from {(min(lengths) - mean) / std:.1f} to {(max(lengths) - mean) / std:.1f}")

    return by_type


def trajectory_statistics(traj_path, segments, by_type, records):
    print_header(f"Trajectories ({os.path.basename(traj_path)})")

    # Column 3 holds the raw GPS points as (?, unix time, longitude, latitude).
    # Column 4 holds the matched path as [edge ID, entry time, speed in m/s, travel time in s].
    speeds = {}
    num_trajectories = 0
    gps_points = []
    unit_checks = []
    with open(traj_path, newline="") as file:
        for row in csv.reader(file):
            num_trajectories += 1
            if num_trajectories <= 20000:
                gps_points.append(len(ast.literal_eval(row[2])))
            for edge_id, entry_time, speed, travel_time in json.loads(row[3]):
                speeds.setdefault(int(edge_id), []).append(speed)
                # Check the unit of speed: speed * travel time should equal the length of the edge
                if len(unit_checks) < 100000 and speed > 0:
                    unit_checks.append(speed * travel_time / float(records[int(edge_id)]["length"]))

    print(f"  trajectories:                             {num_trajectories:,}")
    print(f"  GPS points per trajectory (first 20,000): median {statistics.median(gps_points):g}, "
          f"max {max(gps_points)}")
    print(f"  speed * travel time / edge length:        median {statistics.median(unit_checks):.3f} "
          f"(1.000 means speed is in m/s)")

    # For each segment, pool the speeds of both directions
    segment_speeds = {}
    for segment in segments:
        pooled = []
        for edge_id in segment.edge_ids:
            pooled.extend(speeds.get(int(edge_id), []))
        segment_speeds[id(segment)] = pooled

    covered = sum(1 for s in segments if segment_speeds[id(s)])
    edges_covered = sum(1 for edge_id in records if edge_id in speeds)
    print(f"  segments traversed at least once:         {percent(covered, len(segments)):.1f}%")
    print(f"  edge records traversed at least once:     {percent(edges_covered, len(records)):.1f}%")

    print_header("Traffic per road type (segments)")
    print(f"  {'road type':15s} {'traversed':>10s} {'median speed':>13s}")
    for road_type in MAIN_TYPES:
        segs = by_type[road_type]
        traversed = [s for s in segs if segment_speeds[id(s)]]
        # Median over segments of each segment's median speed, converted from m/s to km/h
        medians = [statistics.median(segment_speeds[id(s)]) * 3.6 for s in traversed]
        median_speed = statistics.median(medians) if medians else float("nan")
        print(f"  {road_type:15s} {percent(len(traversed), len(segs)):9.1f}% {median_speed:9.1f} km/h")


def main():
    parser = argparse.ArgumentParser(description="Prints the dataset figures used in the report.")
    parser.add_argument("--data-dir", default="data/bj_data",
                        help="Folder containing map/ and the trajectory CSV files")
    parser.add_argument("--traj-file", default="traj_train_20.csv",
                        help="Trajectory file inside the data folder (default: traj_train_20.csv)")
    parser.add_argument("--skip-trajectories", action="store_true",
                        help="Only compute the road network figures")
    args = parser.parse_args()

    map_dir = os.path.join(args.data_dir, "map")
    if not os.path.exists(os.path.join(map_dir, "edges.shp")):
        parser.error(f"could not find {os.path.join(map_dir, 'edges.shp')}. "
                     f"Use --data-dir to point at the folder that contains map/ and the trajectory files.")
    records = load_edge_records(map_dir)
    segments = ShapefileMapLoader(map_dir).get_segments()

    by_type = map_statistics(records, segments)
    if not args.skip_trajectories:
        trajectory_statistics(os.path.join(args.data_dir, args.traj_file), segments, by_type, records)


if __name__ == "__main__":
    main()
