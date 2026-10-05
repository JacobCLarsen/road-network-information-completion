# src/ml/data/file_loader.py

import pandas as pd


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
