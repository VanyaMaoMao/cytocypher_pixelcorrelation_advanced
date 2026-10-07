import pandas as pd
import numpy as np
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.config import BeatCounterConfig

def test_build_concatenated_signal_fallback_without_begin():
    # Test data where y-columns exist but there is no 'Begin' column.
    # This triggers the fallback path that attempts to parse `re.match`
    df = pd.DataFrame({
        "y 0": [1.0, 2.0],
        "y -10": [2.0, 3.0],
    })
    
    config = BeatCounterConfig()
    
    try:
        stitched, seg_meta, meta = build_concatenated_signal(
            df,
            y_cols=["y 0", "y -10"],
            fs=250.0,
            config=config,
            force_invert=False
        )
    except NameError as e:
        import pytest
        pytest.fail(f"build_concatenated_signal failed with NameError: {e}")
