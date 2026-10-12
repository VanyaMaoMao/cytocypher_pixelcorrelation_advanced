import pytest
import numpy as np
from pixel_counter.config import BeatCounterConfig
from pixel_counter.analysis import detect_events_v2, auto_select_polarity_v2, analyze_v2_pipeline
from pixel_counter.qc_v2 import evaluate_qc_v2

def create_synthetic_signal(fs, duration, centers, amp=1.0, width=0.02, noise_std=0.0):
    t = np.arange(int(duration * fs)) / fs
    sig = np.zeros_like(t)
    for c in centers:
        sig += amp * np.exp(-0.5 * ((t - c) / width) ** 2)
    if noise_std > 0:
        np.random.seed(42)
        sig += np.random.normal(0, noise_std, size=sig.size)
    return sig

def test_qc_v2_clean_regular_contractions():
    """Clean regular contractions -> PASS."""
    fs = 250.0
    centers = np.arange(1.0, 5.0, 1.0)
    sig = create_synthetic_signal(fs, 6.0, centers)
    config = BeatCounterConfig()

    res = analyze_v2_pipeline(sig, fs, config)
    assert res["metadata"]["qc_status"] == "PASS"

def test_qc_v2_clean_irregular_contractions():
    """Clean but irregular contractions -> PASS (not rejected solely because of irregularity)."""
    fs = 250.0
    centers = [1.0, 1.2, 2.5, 3.8, 3.9, 5.0]
    sig = create_synthetic_signal(fs, 6.0, centers)

    config = BeatCounterConfig(v2_polarity_mode="positive")

    res = analyze_v2_pipeline(sig, fs, config)
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["metadata"]["qc_reason"] == "clean_signal"
    assert res["metadata"]["ibi_cv"] > 0.5  # Highly irregular



def test_qc_v2_genuine_pauses():
    """Genuine pauses -> preserved, not treated as automatic technical failure."""
    fs = 250.0
    centers = [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]
    sig = create_synthetic_signal(fs, 12.0, centers)
    config = BeatCounterConfig()

    res = analyze_v2_pipeline(sig, fs, config)
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["metadata"]["max_pause_s"] >= 4.0

def test_qc_v2_invalid_input():
    """Invalid input or sampling frequency -> ERROR."""
    fs = 250.0
    sig = create_synthetic_signal(fs, 6.0, [1.0, 2.0])
    config = BeatCounterConfig()

    res = analyze_v2_pipeline(sig, -1.0, config)
    assert res["metadata"]["qc_status"] == "ERROR"
    assert res["metadata"]["qc_reason"] == "invalid_sampling_frequency"

    res_empty = analyze_v2_pipeline(np.array([]), fs, config)
    assert res_empty["metadata"]["qc_status"] == "ERROR"
    assert res_empty["metadata"]["qc_reason"] == "empty_signal"

def test_qc_v2_technical_corruption():
    """Genuine technical corruption (clipping, missing data) -> appropriate REJECT or ERROR."""
    fs = 250.0
    sig = create_synthetic_signal(fs, 6.0, [1.0, 2.0])

    # Missing data
    sig_nan = sig.copy()
    sig_nan[:] = np.nan
    res = analyze_v2_pipeline(sig_nan, fs, BeatCounterConfig())
    assert res["metadata"]["qc_status"] == "REJECT"
    assert res["metadata"]["qc_reason"] == "no_valid_data"

    # Clipping
    sig_clip = sig.copy()
    sig_clip[100:300] = 1.0 # Long flat top
    res_clip = analyze_v2_pipeline(sig_clip, fs, BeatCounterConfig())
    assert res_clip["metadata"]["qc_status"] == "REVIEW"
    assert res_clip["metadata"]["qc_reason"] == "signal_clipping"


def test_qc_v2_row_partition_invariance():
    """Repartitioning identical signals into different Excel rows does not change QC decisions."""
    fs = 250.0
    t = np.arange(2500) / fs
    centers = 0.125 + 0.25 * np.arange(40)
    base_signal = np.zeros_like(t)
    for c in centers:
        base_signal += 0.2 * np.exp(-0.5 * ((t - c) / 0.020) ** 2)

    config = BeatCounterConfig(v2_polarity_mode="positive")
    res_base = analyze_v2_pipeline(base_signal, fs, config)

    assert res_base["metadata"]["qc_status"] == "PASS"




def test_qc_v2_ambiguous_polarity():
    """Ambiguous polarity -> REVIEW."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)  # Pure sine wave, perfectly symmetric

    config = BeatCounterConfig(prom0=0.01)
    res = analyze_v2_pipeline(sig, fs, config)

    assert res["metadata"]["qc_status"] == "REVIEW"
    assert res["metadata"]["qc_reason"] == "ambiguous_polarity"

def test_qc_v2_unresolved_neighboring_peaks():
    """Unresolved neighboring peaks -> REVIEW."""
    fs = 250.0
    sig = np.zeros(1000)

    config = BeatCounterConfig()

    # Directly test the QC evaluation by providing a mock candidate dictionary
    cands = {
        "candidate_index": np.array([250, 252]),
        "time_s": np.array([1.0, 1.008]),
        "amplitude": np.array([0.2, 0.198]),
        "prominence": np.array([0.2, 0.198]),
        "width_s": np.array([0.02, 0.02]),
        "polarity": np.array([1, 1]),
        "status": np.array(["PASS", "REVIEW"], dtype=object)
    }

    meta = {"candidate_selection_reason": "auto_selected_positive"}

    from pixel_counter.qc_v2 import evaluate_qc_v2
    status, reason, qc_meta = evaluate_qc_v2(sig, fs, cands, meta, config)

    assert status == "REVIEW"
    assert reason == "unresolved_neighboring_peaks"





def test_qc_v2_insufficient_reliable_seeds():
    """Insufficient reliable seeds -> appropriate uncertainty (REVIEW)."""
    fs = 250.0
    # Many peaks but very low SNR so they barely cross threshold, making them non-seeds.
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    np.random.seed(42)
    sig += np.random.normal(0, 0.05, sig.size) # High noise

    # Weak peaks
    for c in np.arange(1.0, 3.5, 0.5):
        sig += 0.1 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.05, v2_polarity_mode="positive", v2_two_pass_enabled=True)
    res = analyze_v2_pipeline(sig, fs, config)

    # Assert unconditionally
    assert res["candidates"]["candidate_index"].size >= 3
    assert res["metadata"]["qc_status"] == "REVIEW"
    assert res["metadata"]["qc_reason"] == "insufficient_reliable_seeds"


def test_qc_v2_threshold_sensitivity():
    """Threshold sensitivity -> reproducible diagnostics with candidate counts and matched timestamps."""
    fs = 250.0
    centers = np.arange(1.0, 5.0, 1.0)
    sig = create_synthetic_signal(fs, 6.0, centers)
    config = BeatCounterConfig(v2_polarity_mode="positive")

    res = analyze_v2_pipeline(sig, fs, config)
    meta = res["metadata"]

    assert "stability_count_base" in meta
    assert "stability_count_high" in meta
    assert "stability_count_low" in meta
    assert "stability_match_high" in meta
    assert "stability_match_low" in meta

    assert meta["stability_count_base"] == len(centers)
    assert meta["stability_count_high"] == len(centers)
    assert meta["stability_count_low"] == len(centers)
    assert meta["stability_match_high"] == len(centers)
    assert meta["stability_match_low"] == len(centers)



def test_qc_v2_timestamp_instability(monkeypatch):
    """Timestamp instability -> REVIEW (even if event counts remain identical)."""
    fs = 250.0
    sig = np.zeros(1000)

    config = BeatCounterConfig()

    cands = {
        "candidate_index": np.array([250, 500, 750]),
        "time_s": np.array([1.0, 2.0, 3.0]),
        "amplitude": np.array([0.2, 0.2, 0.2]),
        "prominence": np.array([0.2, 0.2, 0.2]),
        "width_s": np.array([0.02, 0.02, 0.02]),
        "polarity": np.array([1, 1, 1]),
        "status": np.array(["PASS", "PASS", "PASS"], dtype=object)
    }

    meta = {"candidate_selection_reason": "auto_selected_positive", "best_polarity": 1}

    def mock_detect(sig, fs, config, polarity=1):
        # Return identical counts but shifted timestamps
        return {
            "candidate_index": np.array([260, 510, 760]),
            "time_s": np.array([1.04, 2.04, 3.04]), # shifted by 40ms (tol is 10ms)
            "status": np.array(["PASS", "PASS", "PASS"], dtype=object)
        }

    monkeypatch.setattr("pixel_counter.analysis.detect_events_v2", mock_detect)

    from pixel_counter.qc_v2 import evaluate_qc_v2
    status, reason, qc_meta = evaluate_qc_v2(sig, fs, cands, meta, config)

    assert qc_meta["stability_count_high"] == 3
    assert qc_meta["stability_match_high"] == 0 # no timestamps match
    assert status == "REVIEW"
    assert reason == "sensitivity_instability"

def test_qc_v2_competing_matches(monkeypatch):
    """Competing matches for the same event should only count once in sensitivity diagnostic."""
    fs = 250.0
    sig = np.zeros(1000)
    config = BeatCounterConfig()

    # Baseline has 3 peaks very close together
    cands = {
        "candidate_index": np.array([250, 252, 254]),
        "time_s": np.array([1.0, 1.008, 1.016]),
        "amplitude": np.array([0.2, 0.2, 0.2]),
        "prominence": np.array([0.2, 0.2, 0.2]),
        "width_s": np.array([0.02, 0.02, 0.02]),
        "polarity": np.array([1, 1, 1]),
        "status": np.array(["PASS", "PASS", "PASS"], dtype=object)
    }

    meta = {"candidate_selection_reason": "auto_selected_positive", "best_polarity": 1}

    def mock_detect(sig, fs, config, polarity=1):
        # Comparison has only 1 peak that is within tolerance (0.01) of multiple baseline peaks.
        # It is exactly at 1.008, which is within 0.01 of 1.0 and 1.016.
        # But it can only match ONE of them.
        return {
            "candidate_index": np.array([252]),
            "time_s": np.array([1.008]),
            "status": np.array(["PASS"], dtype=object)
        }

    monkeypatch.setattr("pixel_counter.analysis.detect_events_v2", mock_detect)

    from pixel_counter.qc_v2 import evaluate_qc_v2
    status, reason, qc_meta = evaluate_qc_v2(sig, fs, cands, meta, config)

    # Because only 1 peak is returned, matches should be exactly 1, not 2 or 3.
    assert qc_meta["stability_match_high"] == 1
    assert qc_meta["stability_match_low"] == 1
