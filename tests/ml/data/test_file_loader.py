import pytest
import pandas as pd
import json
import os
from src.ml.data.file_loader import FileLoader


@pytest.fixture
def mock_files(tmp_path):
    """Fixture to create temporary files for testing."""
    d = tmp_path / "data"
    d.mkdir()

    # 1. Create a sample adjacency CSV
    adj_path = d / "adj.csv"
    adj_df = pd.DataFrame({"from": [1], "to": [2]})
    adj_df.to_csv(adj_path, index=False)

    # 2. Create a sample metadata JSON
    meta_path = d / "meta.json"
    meta_content = {
        "1": {"length": 100, "highway": "primary"},
        "2": {"length": 50, "highway": "secondary"}
    }
    with open(meta_path, 'w') as f:
        json.dump(meta_content, f)

    return str(adj_path), str(meta_path)


def test_get_meta_data_structure(mock_files):
    """Test that metadata is transposed and index is integer."""
    adj_path, meta_path = mock_files
    loader = FileLoader(adj_path, meta_path)

    meta_df = loader.get_meta_data()

    # Check transposition (original keys "1", "2" should be the index now)
    assert 1 in meta_df.index
    assert 2 in meta_df.index
    assert meta_df.index.dtype == "int64"
    assert "length" in meta_df.columns


def test_get_adjacency(mock_files):
    """Test basic loading of connectivity data."""
    adj_path, meta_path = mock_files
    loader = FileLoader(adj_path, meta_path)

    adj_df = loader.get_adjacency()
    assert not adj_df.empty
    assert "from" in adj_df.columns