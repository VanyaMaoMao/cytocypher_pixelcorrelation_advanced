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

def test_nan_preserves_position():
    import numpy as np
    import pandas as pd
    from pixel_counter.preprocessing import build_concatenated_signal
    from pixel_counter.config import BeatCounterConfig
    
    df = pd.DataFrame({
        "Begin": [0.0],
        "End": [5.0],
        "y 0": [0.0],
        "y -10": [1.0],
        "y -20": [np.nan],
        "y -30": [2.0],
        "y -40": [0.0],
    })
    y_cols = ["y 0", "y -10", "y -20", "y -30", "y -40"]
    config = BeatCounterConfig()
    
    stitched, seg_meta, meta = build_concatenated_signal(
        df,
        y_cols=y_cols,
        fs=1.0,
        config=config,
        force_invert=False
    )
    
    assert stitched.shape == (5,)
    assert np.isnan(stitched[2])
    assert stitched[1] == 1.0
    assert stitched[3] == 2.0
