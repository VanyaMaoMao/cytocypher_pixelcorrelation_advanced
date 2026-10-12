import pytest
import numpy as np
from pixel_counter.analysis import auto_select_polarity_v2
from pixel_counter.config import BeatCounterConfig

def test_polarity_override_positive():
    """Manual override to positive polarity."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Negative peak (should not be found if forced positive)
    c = 2.0
    sig -= 0.5 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="positive")
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["best_polarity"] == 1
    assert result["metadata"]["candidate_selected"] == "positive"
    assert result["metadata"]["candidate_selection_reason"] == "manual_override"
    assert result["best_result"]["candidate_index"].size == 0

def test_polarity_override_negative():
    """Manual override to negative polarity."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Negative peak
    c = 2.0
    sig -= 0.5 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="negative")
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["best_polarity"] == -1
    assert result["metadata"]["candidate_selected"] == "negative"
    assert result["metadata"]["candidate_selection_reason"] == "manual_override"
    assert result["best_result"]["candidate_index"].size == 1

def test_asymmetric_auto_selection():
    """Asymmetric signal where one polarity is clearly better."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Positive peaks: clear, large, consistent width
    for c in [1.0, 2.0, 3.0]:
        sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    # Introduce small negative bumps
    for c in [1.5, 2.5]:
        sig -= 0.1 * np.exp(-0.5 * ((t - c) / 0.05) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["best_polarity"] == 1
    assert not result["metadata"]["polarity_ambiguity"]
    assert result["metadata"]["score_margin"] > 0.5  # Large margin

def test_inverted_signal_preserves_count():
    """Test positive and inverted copies of asymmetric contraction signals produce identical candidates."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig_pos = np.zeros_like(t)

    centers = [1.0, 2.0, 3.0]
    for c in centers:
        sig_pos += 1.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    sig_neg = -sig_pos

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")

    res_pos = auto_select_polarity_v2(sig_pos, fs, config)
    res_neg = auto_select_polarity_v2(sig_neg, fs, config)

    assert res_pos["best_polarity"] == 1
    assert res_neg["best_polarity"] == -1

    # Counts should be identical
    assert res_pos["best_result"]["candidate_index"].size == 3
    assert res_neg["best_result"]["candidate_index"].size == 3

    # Timestamps should be identical
    np.testing.assert_allclose(res_pos["best_result"]["time_s"], res_neg["best_result"]["time_s"])

def test_ambiguous_symmetric_signal():
    """Ambiguous symmetric/sinusoidal signals should trigger REVIEW."""
    fs = 250.0
    t = np.arange(1000) / fs

    # Pure sine wave, perfectly symmetric
    sig = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["metadata"]["polarity_ambiguity"] is True
    assert result["metadata"]["score_margin"] < 0.15

def test_clean_irregular_contractions():
    """Irregularly spaced but high-quality contractions should not lose out."""
    fs = 250.0
    t = np.arange(2000) / fs
    sig = np.zeros_like(t)

    # Irregular spacing
    centers = [1.0, 1.2, 1.8, 2.5, 3.5, 3.6, 5.0]
    for c in centers:
        sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    # Introduce small negative bumps to test against
    for c in [2.0, 4.0]:
        sig -= 0.2 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["best_polarity"] == 1
    assert result["best_result"]["candidate_index"].size == len(centers)
    assert not result["metadata"]["polarity_ambiguity"]

def test_incorrect_polarity_more_candidates():
    """Cases where the incorrect polarity produces more candidates but should score lower."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Real positive contractions (few but strong and consistent)
    centers = [1.0, 2.0]
    for c in centers:
        sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)

    # Fake negative artifacts (many but weak/inconsistent)
    np.random.seed(42)
    fake_centers = np.arange(0.5, 3.5, 0.2)
    for c in fake_centers:
        sig -= 0.1 * np.random.uniform(0.5, 1.5) * np.exp(-0.5 * ((t - c) / np.random.uniform(0.01, 0.05)) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")
    result = auto_select_polarity_v2(sig, fs, config)

    # Even if there are more negative peaks, the positive ones have better morphology
    assert result["best_polarity"] == 1

    # The negative score should be lower than positive score
    assert result["metadata"]["candidate_rankings"]["positive"] > result["metadata"]["candidate_rankings"]["negative"]

def test_both_results_preserved_in_auto():
    """Verify both result_positive and result_negative are returned."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    sig += 1.0 * np.exp(-0.5 * ((t - 2.0) / 0.02) ** 2)

    config = BeatCounterConfig(prom0=0.01, v2_polarity_mode="auto")
    result = auto_select_polarity_v2(sig, fs, config)

    assert "result_positive" in result
    assert "result_negative" in result
    assert result["result_positive"] is not None
    assert result["result_negative"] is not None

def test_both_polarities_failed():
    """Verify failure of both polarities triggers REVIEW with correct reason."""
    fs = 250.0
    t = np.arange(1000) / fs
    sig = np.zeros_like(t)

    # Pure noise (no distinct peaks)
    np.random.seed(42)
    sig += np.random.normal(0, 0.01, sig.shape)

    config = BeatCounterConfig(prom0=10.0, v2_polarity_mode="auto") # High threshold to force failure
    result = auto_select_polarity_v2(sig, fs, config)

    assert result["metadata"]["candidate_selected"] == "none"
    assert result["metadata"]["candidate_selection_reason"] == "both_failed"
    assert result["metadata"]["qc_recommendation"] == "REVIEW"
    assert result["metadata"]["qc_reason"] == "both_polarities_failed"
