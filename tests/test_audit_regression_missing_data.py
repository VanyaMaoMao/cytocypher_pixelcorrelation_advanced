import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig

def generate_signal_with_optional_nan(include_nan=False):
    fs = 250.0
    t = np.arange(2500) / fs
    centers = 0.5 + np.arange(10)
    signal = sum(0.2 * np.exp(-0.5 * ((t - c) / 0.035) ** 2) for c in centers)
    rng = np.random.default_rng(12)
    signal += rng.normal(0, 0.0005, len(t))

    if include_nan:
        signal[750:775] = np.nan

    rows = []
    samples_per_row = 250
    for i in range(0, len(t), samples_per_row):
        row = {"Begin (seconds)": t[i], "Sampling Frequency": fs}
        for j in range(samples_per_row):
            row[f"y {j}"] = signal[i+j]
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.mark.xfail(strict=True, reason="NaN downstream handling produces hundreds of artifacts. Fix in step 12-13.")
def test_missing_data_robustness():
    """
    Test that a small gap away from peaks does not create hundreds of false artifacts.
    """
    config = BeatCounterConfig()

    # 1. Intact signal
    df_clean = generate_signal_with_optional_nan(include_nan=False)
    bpm_clean, count_clean, events_clean, meta_clean = count_main_beats_from_excel(
        df_clean, sheet_name="Clean", config=config, show_plot=False
    )

    # Verify clean baseline is as expected
    assert count_clean == 10
    assert meta_clean["quality_status"] == "PASS"
    assert meta_clean.get("vertical_artifact_count", 0) == 0

    # 2. Signal with NaN
    df_nan = generate_signal_with_optional_nan(include_nan=True)
    bpm_nan, count_nan, events_nan, meta_nan = count_main_beats_from_excel(
        df_nan, sheet_name="NaN", config=config, show_plot=False
    )

    # We expect 10 peaks and very few artifacts, preserving robust gap handling.
    # Currently it yields 0 peaks and ~587 artifacts due to NaN propagation.
    assert count_nan == 10, f"Expected 10 peaks, got {count_nan}"
    assert meta_nan["quality_status"] == "PASS", f"Expected PASS, got {meta_nan['quality_status']}"
    assert meta_nan.get("vertical_artifact_count", 0) < 5, f"Too many artifacts: {meta_nan.get('vertical_artifact_count')}"
