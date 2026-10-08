import pytest
import pandas as pd
import numpy as np
from pixel_counter.io_utils import load_cytocypher_excel

def test_load_cytocypher_excel_missing_y(tmp_path):
    df = pd.DataFrame({"Sampling Frequency": [250.0], "Begin (seconds)": [0.0]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="No 'y ' columns"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_duplicate_y(tmp_path):
    df = pd.DataFrame({"y 1": [1], "y 1.0": [2], "y 2": [3]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Duplicate y-offset"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_non_numeric_y(tmp_path):
    df = pd.DataFrame({"y a": [1]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Non-numeric y-offset"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_conflicting_fs(tmp_path):
    df = pd.DataFrame({"y 1": [1, 2], "Sampling Frequency": [250.0, 500.0]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Conflicting Sampling Frequency"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_invalid_fs(tmp_path):
    df = pd.DataFrame({"y 1": [1], "Sampling Frequency": [0.0]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Invalid Sampling Frequency"):
        load_cytocypher_excel(filepath)
        
    df = pd.DataFrame({"y 1": [1], "Sampling Frequency": [-10.0]})
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Invalid Sampling Frequency"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_conflicting_sample_id(tmp_path):
    df = pd.DataFrame({"y 1": [1, 2], "Sample ID": ["A", "B"]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    with pytest.raises(ValueError, match="Conflicting Sample IDs"):
        load_cytocypher_excel(filepath)

def test_load_cytocypher_excel_y_sorting(tmp_path):
    df = pd.DataFrame({"y 10.5": [1], "y -1": [2], "y 0": [3]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, index=False)
    _, y_cols, _, _ = load_cytocypher_excel(filepath)
    assert y_cols == ["y -1", "y 0", "y 10.5"]
