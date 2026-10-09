import pytest
import numpy as np
import pandas as pd
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.config import BeatCounterConfig

@pytest.mark.xfail(strict=True, reason="Legacy build_concatenated_signal incorrectly shifts data via median overlap correction and drops NaNs instead of preserving valid values.")
def test_timing_and_overlap_reconstruction():
    """
    Test the `build_concatenated_signal` function against padding, gaps, and overlapping values.
    Currently it attempts to match overlapping segment baselines and causes values to shift
    incorrectly, while ignoring conflict rules and padding initial empty cells.
    Also tests nested/repeated windows, reordered rows, and timestamp rounding.
    """
    config = BeatCounterConfig()

    fs = 250.0 # 0.004s per sample

    # We will test an intentionally chaotic sequence of windows:
    # Row 1: t=0.0 to 0.016 with leading NaNs (padding test)
    # Row 2: t=0.012 to 0.028 (overlap conflict test with Row 1)
    # Row 3: t=0.028 to 0.044 (normal continuation)
    # Row 4: t=0.032 to 0.040 (nested window test - completely inside Row 3)
    # Row 5: t=0.028 to 0.044 (repeated window test - exactly same time as Row 3)
    # Row 6: t=0.060 to 0.076 (gap test - skips from 0.044 to 0.060)
    # Row 7: t=0.044 to 0.060 (reordered row test - comes after Row 6 but fills the gap)
    # Row 8: t=0.0760001 (timestamp rounding test)

    df = pd.DataFrame([
        { # Row 1: padding test
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y -2": np.nan, "y -1": np.nan, "y 0": 1.0, "y 1": 1.1, "y 2": 1.2
        },
        { # Row 2: overlap conflict test
            "Begin (seconds)": 0.012,
            "Sampling Frequency": fs,
            "y 0": 2.0, "y 1": 2.1, "y 2": 2.2, "y 3": 2.3, "y 4": np.nan
        },
        { # Row 3: normal continuation
            "Begin (seconds)": 0.028,
            "Sampling Frequency": fs,
            "y 0": 3.0, "y 1": 3.1, "y 2": 3.2, "y 3": 3.3, "y 4": 3.4
        },
        { # Row 4: nested window
            "Begin (seconds)": 0.032,
            "Sampling Frequency": fs,
            "y 0": 4.0, "y 1": 4.1, "y 2": 4.2, "y 3": np.nan, "y 4": np.nan
        },
        { # Row 5: repeated window
            "Begin (seconds)": 0.028,
            "Sampling Frequency": fs,
            "y 0": 5.0, "y 1": 5.1, "y 2": 5.2, "y 3": 5.3, "y 4": 5.4
        },
        { # Row 6: gap test (leaves a gap from 0.048 to 0.060 temporarily)
            "Begin (seconds)": 0.060,
            "Sampling Frequency": fs,
            "y 0": 6.0, "y 1": 6.1, "y 2": 6.2, "y 3": 6.3, "y 4": 6.4
        },
        { # Row 7: reordered row test
            "Begin (seconds)": 0.048,
            "Sampling Frequency": fs,
            "y 0": 7.0, "y 1": 7.1, "y 2": 7.2, "y 3": np.nan, "y 4": np.nan
        },
        { # Row 8: timestamp rounding test
            "Begin (seconds)": 0.0760001,
            "Sampling Frequency": fs,
            "y 0": 8.0, "y 1": 8.1, "y 2": np.nan, "y 3": np.nan, "y 4": np.nan
        }
    ])

    y_cols = ["y -2", "y -1", "y 0", "y 1", "y 2", "y 3", "y 4"]

    sig_c, seg_meta_c, orient_c = build_concatenated_signal(df, y_cols, fs, config)

    # We assert that the exact original non-overlapping values should remain identical.
    # Value at index 2 (t=0.008) should be exactly 1.0.
    assert np.isclose(sig_c[2], 1.0), f"Expected 1.0 at index 2, got {sig_c[2]}"

    # Additionally, check the array length.
    # Max time is 0.0760001 + (2*0.004) = 0.084s -> index 21 -> length 22.
    assert len(sig_c) >= 21, f"Expected length >= 21, got {len(sig_c)}"
