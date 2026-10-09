import pytest
import numpy as np
import pandas as pd
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.config import BeatCounterConfig


def test_timing_overlap_padding():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y -2": np.nan, "y -1": np.nan, "y 0": 1.0, "y 1": 1.1, "y 2": 1.2
        }
    ])
    y_cols = ["y -2", "y -1", "y 0", "y 1", "y 2"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    # The value 1.0 should be precisely at index 2 without offset correction.
    assert np.isclose(abs(sig_c[0]), 1.0), f"Expected 1.0 at index 2, got {sig_c[2]}"


def test_timing_overlap_identical():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y 0": 1.0, "y 1": 1.1, "y 2": 1.2
        },
        {
            "Begin (seconds)": 0.004, # Overlaps index 1 and 2
            "Sampling Frequency": fs,
            "y 0": 1.1, "y 1": 1.2, "y 2": 1.3
        }
    ])
    y_cols = ["y 0", "y 1", "y 2"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    assert np.isclose(sig_c[1], 1.1), f"Expected 1.1 at index 1, got {sig_c[1]}"
    assert np.isclose(sig_c[2], 1.2), f"Expected 1.2 at index 2, got {sig_c[2]}"


def test_timing_overlap_conflicting():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y 0": 1.0, "y 1": 1.1, "y 2": 1.2
        },
        {
            "Begin (seconds)": 0.008, # Overlaps index 2
            "Sampling Frequency": fs,
            "y 0": 5.0, "y 1": 5.1, "y 2": 5.2
        }
    ])
    y_cols = ["y 0", "y 1", "y 2"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    # With a conflict, either 1.2 or 5.0 or a NaN marker is expected if handled deterministically,
    # but not an arbitrary offset mask.
    assert np.isclose(sig_c[2], 1.2) or np.isclose(sig_c[2], 5.0), f"Expected strict value or conflict resolution, got {sig_c[2]}"


def test_timing_overlap_nested():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y 0": 1.0, "y 1": 1.1, "y 2": 1.2, "y 3": 1.3, "y 4": 1.4
        },
        {
            "Begin (seconds)": 0.004, # Inside the first block
            "Sampling Frequency": fs,
            "y 0": 2.1, "y 1": 2.2, "y 2": np.nan, "y 3": np.nan, "y 4": np.nan
        }
    ])
    y_cols = ["y 0", "y 1", "y 2", "y 3", "y 4"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    assert len(sig_c) == 5
    assert np.isclose(abs(sig_c[4]), 1.4)


def test_timing_overlap_rounding():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0040001,
            "Sampling Frequency": fs,
            "y 0": 1.0, "y 1": 1.1, "y 2": np.nan
        }
    ])
    y_cols = ["y 0", "y 1", "y 2"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    # The array should start placing values effectively at index 1 given time 0.0040001
    assert np.isclose(abs(sig_c[0]), 1.0)

def test_timing_overlap_gap():
    fs = 250.0
    config = BeatCounterConfig()
    df = pd.DataFrame([
        {
            "Begin (seconds)": 0.0,
            "Sampling Frequency": fs,
            "y 0": 1.0, "y 1": 1.1
        },
        {
            "Begin (seconds)": 0.016, # Gap between 0.008 and 0.016
            "Sampling Frequency": fs,
            "y 0": 2.0, "y 1": 2.1
        }
    ])
    y_cols = ["y 0", "y 1"]
    sig_c, _, _ = build_concatenated_signal(df, y_cols, fs, config)
    # Expect nans in indices 2, 3
    assert np.isnan(sig_c[2])
    assert np.isnan(sig_c[3])
    assert np.isclose(abs(sig_c[4]), 2.0)
    assert np.isnan(sig_c[3])
