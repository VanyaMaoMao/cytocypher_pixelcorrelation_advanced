import numpy as np
from pixel_counter.config import BeatCounterConfig
from pixel_counter.analysis import auto_select_polarity_v2, detect_events_v2

def test_polarity_invariance():
    """Test that inverting a signal yields equivalent physiological events with reversed polarity."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Create a true physiological signal (spends most time at baseline 0, spikes up to 1)
    centers = [1.0, 2.0, 3.0]
    for c in centers:
        sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.05) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")

    # Original signal
    res1 = auto_select_polarity_v2(sig, fs, config)
    assert res1["best_polarity"] == 1
    pks1 = res1["best_result"]["candidate_index"]

    # Inverted signal
    res2 = auto_select_polarity_v2(-sig, fs, config)
    assert res2["best_polarity"] == -1
    pks2 = res2["best_result"]["candidate_index"]

    # The timestamps should be exactly the same
    np.testing.assert_array_equal(pks1, pks2)


def test_baseline_shift_invariance():
    """Test that shifting the baseline does not affect polarity selection."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    centers = [1.0, 2.0, 3.0]
    for c in centers:
        sig -= 1.0 * np.exp(-0.5 * ((t - c) / 0.05) ** 2)  # Negative peaks

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")

    res1 = auto_select_polarity_v2(sig, fs, config)
    assert res1["best_polarity"] == -1
    pks1 = res1["best_result"]["candidate_index"]

    # Shift baseline by +500 and scale amplitude
    res2 = auto_select_polarity_v2(sig * 10 + 500, fs, config)
    assert res2["best_polarity"] == -1
    pks2 = res2["best_result"]["candidate_index"]

    np.testing.assert_array_equal(pks1, pks2)


def test_recording_start_phase_invariance():
    """Test that changing the phase at which the recording starts doesn't flip polarity."""
    fs = 250.0
    # Long recording
    t = np.arange(2000) / fs
    sig = np.zeros_like(t)

    # Peaks every 0.5s starting at 0.5
    centers = np.arange(0.5, 8.0, 0.5)
    for c in centers:
        sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")

    # Base: starts at baseline
    res1 = auto_select_polarity_v2(sig[:1000], fs, config)
    assert res1["best_polarity"] == 1

    # Shifted: starts right in the middle of a peak (idx 125 is 0.5s)
    shifted_sig = sig[125:1125]
    res2 = auto_select_polarity_v2(shifted_sig, fs, config)
    assert res2["best_polarity"] == 1

def test_asymmetric_waveform():
    """Test that asymmetric contraction/relaxation curves still identify the correct polarity."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Create asymmetric peaks: fast rise, slow fall
    centers = [1.0, 2.0, 3.0]
    for c in centers:
        # Before center: narrow (0.02)
        idx_before = t < c
        sig[idx_before] += 1.0 * np.exp(-0.5 * ((t[idx_before] - c) / 0.02) ** 2)
        # After center: wide (0.1)
        idx_after = t >= c
        sig[idx_after] += 1.0 * np.exp(-0.5 * ((t[idx_after] - c) / 0.1) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")

    res1 = auto_select_polarity_v2(sig, fs, config)
    assert res1["best_polarity"] == 1

    # And inverted
    res2 = auto_select_polarity_v2(-sig, fs, config)
    assert res2["best_polarity"] == -1
    np.testing.assert_array_equal(res1["best_result"]["candidate_index"], res2["best_result"]["candidate_index"])
