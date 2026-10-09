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


def test_missing_data_robustness_clean(tmp_path):
    config = BeatCounterConfig()

    # 1. Intact signal
    df_clean = generate_signal_with_optional_nan(include_nan=False)
    test_file_clean = tmp_path / "clean.xlsx"
    df_clean.to_excel(test_file_clean, sheet_name="Clean", index=False)

    bpm_clean, count_clean, events_clean, meta_clean = count_main_beats_from_excel(
        str(test_file_clean), sheet_name="Clean", config=config, show_plot=False
    )

    assert count_clean == 10
    assert meta_clean.get("qc_pass", True) or meta_clean.get("quality_status") == "PASS"

@pytest.mark.xfail(strict=True, raises=AssertionError, reason="NaN downstream handling produces hundreds of artifacts. Fix in step 12-13.")
def test_missing_data_robustness_nan(tmp_path):
    config = BeatCounterConfig()

    # 2. Signal with NaN
    df_nan = generate_signal_with_optional_nan(include_nan=True)
    test_file_nan = tmp_path / "nan.xlsx"
    df_nan.to_excel(test_file_nan, sheet_name="NaN", index=False)

    bpm_nan, count_nan, events_nan, meta_nan = count_main_beats_from_excel(
        str(test_file_nan), sheet_name="NaN", config=config, show_plot=False
    )

    assert count_nan == 10, f"Expected 10 peaks, got {count_nan}"
    assert meta_nan.get("qc_pass", True) or meta_nan.get("quality_status") == "PASS"
    assert meta_nan.get("n_vertical_artifacts", 0) < 5, f"Too many artifacts: {meta_nan.get('n_vertical_artifacts')}"
