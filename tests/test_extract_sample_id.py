import pytest
import pandas as pd
import numpy as np
from pixel_counter.io_utils import extract_sample_id_from_segment_sheet

def test_extract_sample_id_from_segment_sheet(tmp_path):
    df = pd.DataFrame({"Sample ID": ["A", "A"]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, sheet_name="Sheet1", index=False)
    assert extract_sample_id_from_segment_sheet(filepath, "Sheet1") == "A"

def test_extract_sample_id_conflicting(tmp_path):
    df = pd.DataFrame({"Sample ID": ["A", "B"]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, sheet_name="Sheet1", index=False)
    with pytest.raises(ValueError, match="Conflicting Sample IDs found"):
        extract_sample_id_from_segment_sheet(filepath, "Sheet1")

def test_extract_sample_id_numeric(tmp_path):
    df = pd.DataFrame({"Sample ID": [42.0, 42.0]})
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, sheet_name="Sheet1", index=False)
    assert extract_sample_id_from_segment_sheet(filepath, "Sheet1") == 42
