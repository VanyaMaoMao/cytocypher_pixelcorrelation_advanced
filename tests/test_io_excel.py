import pytest
import pandas as pd
import numpy as np
from pixel_counter.io_utils import load_cytocypher_excel

def test_load_cytocypher_excel(tmp_path):
    df = pd.DataFrame({
        "Sampling Frequency": [250.0, 250.0, np.nan],
        "Begin (seconds)": [0.0, 0.004, 0.008],
        "y -1.0": [1, 2, 3],
        "y 0.0": [4, 5, 6],
        "y 1.0": [7, 8, 9],
        "Sample ID": ["A", "A", np.nan],
        "Transientnumber": [1, 1, 2]
    })
    filepath = tmp_path / "test.xlsx"
    df.to_excel(filepath, sheet_name="Sheet1", index=False)
    
    out_df, y_cols, fs, t0 = load_cytocypher_excel(filepath, sheet_name="Sheet1")
    assert fs == 250.0
    assert t0 == 0.0
    assert set(y_cols) == {"y -1.0", "y 0.0", "y 1.0"}
    assert list(out_df["_tn"]) == [1, 1, 2]

