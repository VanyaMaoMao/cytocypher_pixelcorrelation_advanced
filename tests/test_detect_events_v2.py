import pytest
import numpy as np
from pixel_counter.analysis import detect_events_v2, deduplicate_events_v2
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

@pytest.mark.parametrize("samples_per_row", [125, 250, 500])
def test_detect_events_v2_row_partition_invariance(samples_per_row):
    """
    Directly verify that evaluating data derived from different Excel row partitions
    yields identically invariant peak detection in the v2 pipeline.
    """
    fs = 250.0
    t = np.arange(2500) / fs
    centers = 0.125 + 0.25 * np.arange(40)

    # Generate the base signal
    base_signal = np.zeros_like(t)
    for c in centers:
        base_signal += 0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2)

    # Simulate extraction of the signal from varying row partitions
    stitched_signal = np.full(2500, np.nan)
    for i in range(0, 2500, samples_per_row):
        end = min(i + samples_per_row, 2500)
        stitched_signal[i:end] = base_signal[i:end]

    config = BeatCounterConfig(prom0=0.01)

    # Evaluate via v2
    res = detect_events_v2(stitched_signal, fs, config)

    assert res["candidate_index"].size == 40, f"Expected exactly 40 peaks for partition size {samples_per_row}, got {res['candidate_index'].size}"
    np.testing.assert_allclose(res["time_s"], centers, atol=0.008)

def test_dedup_deep_valley_separation():
    """Verify that two close peaks separated by a deep valley are preserved as separate genuine contractions."""
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Insert two close peaks (e.g. 0.08s apart)
    c1, c2 = 1.0, 1.08
    signal += 0.2 * np.exp(-0.5 * ((t - c1) / 0.01) ** 2)
    signal += 0.2 * np.exp(-0.5 * ((t - c2) / 0.01) ** 2)

    # Force a deep valley between them
    idx_c1 = int(c1 * fs)
    idx_c2 = int(c2 * fs)
    valley_idx = (idx_c1 + idx_c2) // 2
    signal[valley_idx] = 0.0

    config = BeatCounterConfig(prom0=0.01, min_peak_distance_s=0.0)
    res = detect_events_v2(signal, fs, config)
    res = deduplicate_events_v2(signal, fs, res, config)
    assert res["candidate_index"].size == 2
    np.testing.assert_allclose(res["time_s"], [c1, c2], atol=0.008)


def test_dedup_shoulder_morphology():
    """Verify that a shoulder on a single contraction is merged correctly."""
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Main peak
    c1 = 1.0
    signal += 0.5 * np.exp(-0.5 * ((t - c1) / 0.03) ** 2)

    # Shoulder peak (very close, shallow valley)
    c2 = 1.05
    signal += 0.45 * np.exp(-0.5 * ((t - c2) / 0.03) ** 2)

    config = BeatCounterConfig(prom0=0.01, min_peak_distance_s=0.0)
    res = detect_events_v2(signal, fs, config)
    res = deduplicate_events_v2(signal, fs, res, config)
    # Should be deduplicated to a single peak
    assert res["candidate_index"].size == 1
    np.testing.assert_allclose(res["time_s"][0], 1.0, atol=0.02)


def test_dedup_abc_chain():
    """Verify that a chain of nearby candidates (A-B-C) is correctly deduplicated."""
    fs = 250.0
    t = np.arange(1000) / fs
    signal = np.zeros_like(t)

    # Three peaks forming a chain
    c1, c2, c3 = 1.0, 1.04, 1.08
    signal += 0.3 * np.exp(-0.5 * ((t - c1) / 0.015) ** 2)
    signal += 0.4 * np.exp(-0.5 * ((t - c2) / 0.015) ** 2)
    signal += 0.3 * np.exp(-0.5 * ((t - c3) / 0.015) ** 2)

    config = BeatCounterConfig(prom0=0.01, min_peak_distance_s=0.0)
    res = detect_events_v2(signal, fs, config)
    res = deduplicate_events_v2(signal, fs, res, config)
    # They should all merge into the central dominant peak (c2)
    assert res["candidate_index"].size == 1
    np.testing.assert_allclose(res["time_s"][0], c2, atol=0.01)


def test_dedup_fast_regular_rhythm():
    """Verify that fast regular rhythms (~10 Hz) are preserved."""
    fs = 250.0
    t = np.arange(2000) / fs
    signal = np.zeros_like(t)

    # 10 Hz contractions (every 0.1s)
    centers = np.arange(1.0, 4.0, 0.1)
    for c in centers:
        signal += 0.2 * np.exp(-0.5 * ((t - c) / 0.015) ** 2)

    # Ensure valley is deep enough
    for i in range(len(centers) - 1):
        valley_idx = int((centers[i] + centers[i+1]) / 2.0 * fs)
        signal[valley_idx] = 0.01

    config = BeatCounterConfig(prom0=0.01, min_peak_distance_s=0.0)
    res = detect_events_v2(signal, fs, config)
    res = deduplicate_events_v2(signal, fs, res, config)
    assert res["candidate_index"].size == len(centers)
    np.testing.assert_allclose(res["time_s"], centers, atol=0.008)


def test_dedup_flag_for_review():
    """Verify that virtually indistinguishable candidates are flagged for REVIEW."""
    fs = 250.0
    signal = np.zeros(1000)

    peaks = np.array([250, 252])
    signal[250] = 0.2
    signal[251] = 0.199
    signal[252] = 0.198

    from pixel_counter.analysis import deduplicate_events_v2
    cands = {
        "candidate_index": np.array([250, 252]),
        "time_s": np.array([1.0, 1.008]),
        "amplitude": np.array([0.2, 0.198]),
        "prominence": np.array([0.2, 0.198]),
        "width_s": np.array([0.02, 0.02]),
        "polarity": np.array([1, 1])
    }

    config = BeatCounterConfig()
    res = deduplicate_events_v2(signal, fs, cands, config)

    # Should deduplicate to 1, but be flagged for review
    assert res["candidate_index"].size == 1
    assert res["status"][0] == "REVIEW"
