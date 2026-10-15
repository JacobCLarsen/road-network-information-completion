# src/ml/data/file_loader.py

import os
import pandas as pd
import shapefile
from src.ml.data.road_graph_builder import Segment


class FileLoader:
    def __init__(self, edge_connections_path: str, meta_data_path: str):
        """
        Initializes the FileLoader with the paths to the road network files.
        """
        self.edge_connections_path = edge_connections_path
        self.meta_data_path = meta_data_path

    def get_adjacency(self) -> pd.DataFrame:
        """
        Loads the edge connectivity data (adjacency list).
        """
        df = pd.read_csv(self.edge_connections_path)
        return df

    def get_meta_data(self) -> pd.DataFrame:
        """
        Loads the OSM metadata, transposes it (rows are edges), and ensures the Edge ID index is an integer.
        """
        df = pd.read_json(self.meta_data_path).T
        df.index = df.index.astype(int)
        return df


class ShapefileMapLoader:
    def __init__(self, map_dir):
        """
        Loads a road network stored as a shapefile (the file edges.shp in map_dir), such as the DRPK maps.
        """
        self.edges_path = os.path.join(map_dir, "edges")

    def get_segments(self):
        """
        Reads the edge records and turns them into a list of Segment objects.

        The records are already split at junctions, but a two-way road is stored twice: once from u to v
        and once from v to u. The two records are merged into one segment that keeps both edge IDs.
        """
        reader = shapefile.Reader(self.edges_path, encoding="utf-8")

        # Read all records. Each record has the tags of the road and its shape as a list of points.
        records = []
        for shape_record in reader.iterShapeRecords():
            tags = shape_record.record.as_dict()
            points = []
            for point in shape_record.shape.points:
                points.append((point[0], point[1]))
            records.append((tags, points))

        # Find the edge ID of every two-way record, looked up by (u, v, key).
        # key separates records that have the same u and v but are different roads.
        two_way_ids = {}
        for (tags, points) in records:
            if tags["oneway"] != "True":
                two_way_ids[(tags["u"], tags["v"], tags["key"])] = tags["fid"]

        segments = []
        for (tags, points) in records:
            u = tags["u"]
            v = tags["v"]
            key = tags["key"]

            # Look for the record that goes the opposite way. Loops (u == v) have no opposite record.
            reverse_id = None
            if tags["oneway"] != "True" and u != v:
                reverse_id = two_way_ids.get((v, u, key))

            if reverse_id is None:
                edge_ids = [tags["fid"]]
            else:
                # Both directions exist. Only keep the record where u < v, so the road is added once.
                if u > v:
                    continue
                edge_ids = [tags["fid"], reverse_id]

            segments.append(Segment(edge_ids, u, v, points, tags["highway"], tags["oneway"]))

        return segments
