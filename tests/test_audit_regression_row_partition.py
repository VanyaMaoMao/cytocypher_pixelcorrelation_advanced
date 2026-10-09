import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig

def generate_partitioned_df(samples_per_row):
    fs = 250.0
    t = np.arange(2500) / fs
    centers = 0.125 + 0.25 * np.arange(40)
    signal = sum(0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2) for c in centers)
    rng = np.random.default_rng(42)
    signal += rng.normal(0, 0.0005, len(t))

    rows = []
    for i in range(0, len(t), samples_per_row):
        row = {"Begin (seconds)": t[i], "Sampling Frequency": fs}
        for j in range(samples_per_row):
            row[f"y {j}"] = signal[i+j]
        rows.append(row)

    return pd.DataFrame(rows)

@pytest.mark.parametrize("samples_per_row", [125, 250, 500, 2500])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Legacy detector depends on row partition. Fix in steps 14-18.")
def test_row_partition_invariance(samples_per_row, tmp_path):
    """
    Test that the same signal produces exactly 40 peaks regardless of how it is partitioned into rows.
    Legacy results are: 125 -> 36, 250 -> 24, 500 -> 21, 2500 -> 0.
    """
    config = BeatCounterConfig()
    df = generate_partitioned_df(samples_per_row)

    test_file = tmp_path / "test.xlsx"
    df.to_excel(test_file, sheet_name="Test Segment", index=False)

    bpm, count, events, meta = count_main_beats_from_excel(
        str(test_file),
        sheet_name="Test Segment",
        config=config,
        show_plot=False
    )

    assert count == 40, f"Expected exactly 40 peaks for partition size {samples_per_row}, got {count}"

    # Check that events timestamps are correct
    centers = 0.125 + 0.25 * np.arange(40)
    acc = events[events["decision_status"] == "accepted"]
    timestamps = acc["Time_s"].values if "Time_s" in acc.columns else acc["time_s"].values

    # Allow 2 samples (8ms) tolerance due to peak extraction interpolation/finding
    assert len(timestamps) == 40, "Length of accepted timestamps must equal 40"
    assert np.allclose(timestamps, centers, atol=0.008), "Timestamps deviate too far from ground truth centers"
