import pytest
import numpy as np
from pixel_counter.analysis import detect_events_v2
from pixel_counter.config import BeatCounterConfig

def test_detect_events_v2_synthetic_frequencies():
    """Test synthetic signals containing known 1, 2, and 4 Hz contractions."""
    fs = 250.0
    t = np.arange(10 * fs) / fs
    config = BeatCounterConfig(prom0=0.01)

    for freq in [1.0, 2.0, 4.0]:
        signal = np.zeros_like(t)
        centers = np.arange(1, 10, 1.0 / freq)
        for c in centers:
            signal += 0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2)

        res = detect_events_v2(signal, fs, config)
        assert res["candidate_index"].size == len(centers), f"Failed for {freq} Hz"
        # Check timestamps within 8ms tolerance
        np.testing.assert_allclose(res["time_s"], centers, atol=0.008)
        assert np.all(res["amplitude"] > 0.1)
        assert np.all(res["prominence"] > 0.05)


def test_detect_events_v2_missing_data_gaps():
    """Verify that missing-data gaps do not generate artificial contraction candidates."""
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Insert a genuine peak
    c1, c2 = 1.0, 3.0
    signal += 0.2 * np.exp(-0.5 * ((t - c1) / 0.020) ** 2)
    signal += 0.2 * np.exp(-0.5 * ((t - c2) / 0.020) ** 2)

    # Insert missing data gap between them
    gap_start, gap_end = int(1.5 * fs), int(2.5 * fs)
    signal[gap_start:gap_end] = np.nan

    config = BeatCounterConfig(prom0=0.01)
    res = detect_events_v2(signal, fs, config)

    assert res["candidate_index"].size == 2
    np.testing.assert_allclose(res["time_s"], [c1, c2], atol=0.008)
    assert not np.any((res["time_s"] > 1.5) & (res["time_s"] < 2.5))


def test_detect_events_v2_polarity():
    """Test positive and negative signals with explicitly specified polarity."""
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Insert a negative peak
    c = 2.0
    signal -= 0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2)

    config = BeatCounterConfig(prom0=0.01)

    # With positive polarity, shouldn't find it
    res_pos = detect_events_v2(signal, fs, config, polarity=1)
    assert res_pos["candidate_index"].size == 0

    # With negative polarity, should find it
    res_neg = detect_events_v2(signal, fs, config, polarity=-1)
    assert res_neg["candidate_index"].size == 1
    assert abs(res_neg["time_s"][0] - c) < 0.008
    assert res_neg["polarity"][0] == -1


def test_detect_events_v2_row_boundary():
    """
    Detect a peak located exactly at an Excel row boundary.
    In v2, the global valid block should detect it seamlessly.
    """
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Peak exactly at a boundary (e.g., at sample 250)
    c = 1.0 # Sample 250
    signal += 0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2)

    config = BeatCounterConfig(prom0=0.01)
    res = detect_events_v2(signal, fs, config)

    assert res["candidate_index"].size == 1
    assert res["candidate_index"][0] == 250
    assert abs(res["time_s"][0] - 1.0) < 1e-5


def test_detect_events_v2_short_block():
    """Verify correct handling of short valid blocks."""
    fs = 250.0
    t = np.arange(10) / fs # extremely short
    signal = np.zeros_like(t)

    config = BeatCounterConfig(prom0=0.01)
    # Should not crash, should return empty
    res = detect_events_v2(signal, fs, config)
    assert res["candidate_index"].size == 0
