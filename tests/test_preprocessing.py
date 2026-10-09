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

def test_build_transient_id_vector_sentinel():
    import numpy as np
    from pixel_counter.preprocessing import build_transient_id_vector
    
    # We have a signal of length 100
    # Two valid segments: 0-30 and 40-70. Rest are gaps.
    seg_meta = [
        (1, 0, 30, 0.0, 30),
        (2, 40, 70, 0.0, 30)
    ]
    
    trans_id = build_transient_id_vector(100, seg_meta)
    
    assert trans_id.shape == (100,)
    assert np.all(trans_id[0:30] == 1)
    assert np.all(trans_id[30:40] == -1)
    assert np.all(trans_id[40:70] == 2)
    assert np.all(trans_id[70:100] == -1)


def test_build_concatenated_signal_overlap_and_gap():
    import numpy as np
    import pandas as pd
    from pixel_counter.preprocessing import build_concatenated_signal
    from pixel_counter.config import BeatCounterConfig

    # Construct a dataframe with realistic overlap and gap
    # Row 0: 0.0s to 3.0s
    # Row 1: 2.0s to 5.0s  <- overlaps with Row 0 by 1.0s
    # Row 2: 6.0s to 9.0s  <- gap of 1.0s from end of Row 1
    # Sample rate 10Hz to make it easy to reason about
    fs = 10.0
    
    # We need 3 seconds of data at 10Hz = 30 samples per row.
    y_cols = [f"y {i}" for i in range(30)]
    
    # recreate dataframe to have actual 30 y columns
    data = {"Begin": [0.0, 2.0, 6.0], "End": [3.0, 5.0, 9.0]}
    # Make sure overlaps do not conflict numerically
    for i in range(30):
        # Row 0: starts at 0.0, so at index 20 it is at y 20
        # Row 1: starts at 2.0, so at index 20 it is at y 0
        # We need Row 0's y 20 to match Row 1's y 0.
        # Let's just use the absolute time index for the values to guarantee no conflict.
        # t0 = 0, y = 0.01 * idx
        # t1 = 20, y = 0.01 * (idx + 20)
        # t2 = 60, y = 0.01 * (idx + 60)
        data[f"y {i}"] = [0.01 * i, 0.01 * (i + 20), 0.01 * (i + 60)]
    
    df = pd.DataFrame(data)
    config = BeatCounterConfig()
    
    stitched, seg_meta, meta = build_concatenated_signal(
        df,
        y_cols=y_cols,
        fs=fs,
        config=config,
        force_invert=False
    )
    
    # global start: 0
    # global end: 9.0 * 10 = 90
    # Total length should be 90.
    assert stitched.shape == (90,)
    
    # Verify the gap between 5.0s (idx 50) and 6.0s (idx 60) is NaN
    assert np.all(np.isnan(stitched[50:60]))
    
    # Verify traces are valid outside gap
    assert np.all(~np.isnan(stitched[0:50]))
    assert np.all(~np.isnan(stitched[60:90]))

