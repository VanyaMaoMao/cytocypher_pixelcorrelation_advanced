import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig

@pytest.mark.xfail(strict=True, reason="Result BPM uses len/fs while Reporting BPM uses user-defined duration. Rejected uses 0 BPM instead of NaN.")
def test_result_and_bpm_consistency():
    """
    Test BPM denominator consistency and REJECT status representation.
    """
    config = BeatCounterConfig()
    fs = 250.0
    # Simulate a signal that is 4 seconds long (1000 samples)
    t = np.arange(1000) / fs
    # 5 peaks
    centers = 0.5 + 0.5 * np.arange(5)
    signal = sum(0.2 * np.exp(-0.5 * ((t - c) / 0.035) ** 2) for c in centers)
    rng = np.random.default_rng(42)
    signal += rng.normal(0, 0.0005, len(t))

    # Pack it into a DataFrame where 'Begin (seconds)' dictates a short observed duration
    # But let's say the user 'duration' is 10s.
    df = pd.DataFrame({
        "Begin (seconds)": t,
        "Sampling Frequency": [fs] * len(t),
        "y 0": signal
    })

    # We run analysis.
    bpm_file, count, events, meta = count_main_beats_from_excel(
        df, sheet_name="Test Segment", config=config, show_plot=False
    )

    # 5 peaks over 4 seconds = 1.25 Hz = 75 BPM.
    # The detector computes BPM as 5 / (4.0s) * 60 = 75 BPM.
    # We assert that the returned BPM is indeed 75 BPM.
    assert np.isclose(bpm_file, 75.0), f"Expected 75 BPM based on array length, got {bpm_file}"

    # In run_pixel_analysis, it overrides recording_s=10.0 and computes BPM = 5 / 10.0 * 60 = 30 BPM.
    # The discrepancy is noted in the issue. Here we will also simulate a REJECT scenario.

    # Simulate REJECT by having too few rows or something else.
    # Or just a noisy array.
    signal_noise = rng.normal(0, 0.1, len(t))
    df_reject = pd.DataFrame({
        "Begin (seconds)": t,
        "Sampling Frequency": [fs] * len(t),
        "y 0": signal_noise
    })
    bpm_reject, count_reject, events_reject, meta_reject = count_main_beats_from_excel(
        df_reject, sheet_name="Test Reject", config=config, show_plot=False
    )

    # Ensure it rejected
    assert meta_reject["quality_status"] != "PASS"

    # BPM for a rejected file must be NaN, not 0!
    # Currently it returns 0.0.
    assert np.isnan(bpm_reject), f"Expected NaN BPM for rejected segment, got {bpm_reject}"
