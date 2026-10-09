import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig

def test_result_and_bpm_consistency():
    config = BeatCounterConfig()
    fs = 250.0
    t = np.arange(1000) / fs
    centers = 0.5 + 0.5 * np.arange(5)
    signal = sum(0.2 * np.exp(-0.5 * ((t - c) / 0.035) ** 2) for c in centers)
    rng = np.random.default_rng(42)
    signal += rng.normal(0, 0.0005, len(t))

    row_data = []
    for i in range(10):
        row = {
            "Begin (seconds)": i * 100 / fs,
            "End": (i + 1) * 100 / fs,
            "Sampling Frequency": fs
        }
        for j in range(100):
            row[f"y {j}"] = signal[i * 100 + j]
        row_data.append(row)
    df = pd.DataFrame(row_data)

    import pixel_counter.analysis as analysis
    def mock_load1(*args, **kwargs): return df, [f"y {i}" for i in range(100)], fs, 0.0
    analysis.load_cytocypher_excel = mock_load1

    bpm_file, count, events, meta = count_main_beats_from_excel(
        "dummy.xlsx", sheet_name="Test Segment", config=config, show_plot=False
    )

    assert np.isclose(bpm_file, 75.0), f"Expected 75 BPM based on array length, got {bpm_file}"

    signal_noise = rng.normal(0, 0.1, len(t))
    row_data_reject = []
    for i in range(10):
        row = {
            "Begin (seconds)": i * 100 / fs,
            "End": (i + 1) * 100 / fs,
            "Sampling Frequency": fs
        }
        for j in range(100):
            row[f"y {j}"] = signal_noise[i * 100 + j]
        row_data_reject.append(row)
    df_reject = pd.DataFrame(row_data_reject)

    def mock_load2(*args, **kwargs): return df_reject, [f"y {i}" for i in range(100)], fs, 0.0
    analysis.load_cytocypher_excel = mock_load2

    bpm_reject, count_reject, events_reject, meta_reject = count_main_beats_from_excel(
        "dummy.xlsx", sheet_name="Test Reject", config=config, show_plot=False
    )

    assert meta_reject["qc_pass"] == False
    assert np.isnan(bpm_reject), f"Expected NaN BPM for rejected segment, got {bpm_reject}"
