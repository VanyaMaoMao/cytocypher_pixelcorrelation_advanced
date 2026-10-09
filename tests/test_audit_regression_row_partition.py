import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig
import warnings

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

@pytest.mark.xfail(strict=True, reason="Legacy detector depends on row partition. Fix in steps 14-18.")
def test_row_partition_invariance():
    """
    Test that the same signal produces 40 peaks regardless of how it is partitioned into rows.
    This demonstrates the legacy detector defect where identical signals yield 36, 24, 21, or 0 beats
    based on the row length.
    """
    config = BeatCounterConfig()

    # In a perfect world all counts should be 40
    # Current behavior is: 125 -> 36, 250 -> 24, 500 -> 21, 2500 -> 0
    counts = []
    for samples_per_row in [125, 250, 500, 2500]:
        df = generate_partitioned_df(samples_per_row)

        # Suppress plotting and file output for tests
        bpm, count, events, meta = count_main_beats_from_excel(
            df,
            sheet_name="Test Segment",
            config=config,
            show_plot=False
        )
        counts.append(count)

    # We expect all counts to equal 40.
    # Currently this will fail, which is why we xfail(strict=True).
    assert counts == [40, 40, 40, 40], f"Counts were {counts}, expected [40, 40, 40, 40]"
