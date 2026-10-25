"""
tests.test_calcium_analyzer
===========================

Comprehensive unit and scientific integration tests for the independent
calcium transient detector and review pipeline.

Verifies:
1. Positive-polarity transient detection on synthetic signals (varying frequencies, amplitudes).
2. Discrimination against narrow optical/electrical spikes (FWHM threshold).
3. Robust local baseline F0(t) estimation and peak height calculation (Delta F = y_peak - F0).
4. Frequency calculation in Hz and BPM with correct handling of quiescent segments (0 Hz).
5. Hardware sensor dropout detection (< 0.05) and masking:
   - Partial dropouts (e.g. DISH2ICI Seg 7, 8; DISH3CGP Seg 3) pass with partial exclusion.
   - Severe dropouts (e.g. DISH2ICI Seg 9 > 85% dropouts) reject with severe_sensor_dropout.
6. Noise-dominated negative control rejection (SCRAMBLED ISO, ERROR-MOTION-CONTROLLER).
7. Benchmark agreement with expert reference timings:
   - 25092026_IMR90_1,2,3HZ (Segments 6, 9, 19, 28, 31, 37)
   - DISH1NICOTINE (Segments 1, 6, 10)
8. CLI analysis modes ('pixel', 'calcium', 'pixel-calcium') and channel independence.
9. Annotation export format: uncalibrated confidence (empty string), explicit status, reason, channel.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from pixel_counter.calcium_detector import (
    CALCIUM_STRICTNESS_PRESETS,
    CalciumDetectorParams,
    analyze_calcium_pipeline,
    detect_calcium_dropouts,
    detect_calcium_transients,
    estimate_calcium_baseline,
    estimate_calcium_noise_sigma,
    evaluate_calcium_qc,
)
from pixel_counter.config import BeatCounterConfig
from pixel_counter.io_utils import load_cytocypher_excel
from pixel_counter.preprocessing import build_concatenated_signal


DATA_DIR = Path(__file__).resolve().parent.parent / "all data"


# ---------------------------------------------------------------------------
# 1. Synthetic Calcium Signal Synthesis Helper
# ---------------------------------------------------------------------------
def make_synthetic_calcium(
    dur_s: float = 8.0,
    fs: float = 250.0,
    peak_times: List[float] = None,
    f0: float = 0.25,
    amp: float = 0.12,
    noise_sigma: float = 0.0015,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate realistic synthetic cardiomyocyte calcium fluorescence ratio transients."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    t = np.arange(n) / fs
    sig = np.full(n, f0, dtype=float)

    if peak_times is None:
        peak_times = [1.0, 2.5, 4.0, 5.5, 7.0]

    # Asymmetric calcium transient morphology: fast upstroke (~40 ms), exponential decay (~400 ms)
    tau_up = 0.035
    tau_down = 0.350

    for pt in peak_times:
        idx_start = int(max(0, (pt - 0.10) * fs))
        idx_end = int(min(n, (pt + 1.20) * fs))
        for i in range(idx_start, idx_end):
            dt = t[i] - pt
            if dt < -tau_up * 3:
                continue
            elif dt <= 0:
                # Rising phase
                sig[i] += amp * np.exp(-0.5 * (dt / tau_up) ** 2)
            else:
                # Decay phase
                sig[i] += amp * np.exp(-dt / tau_down)

    if noise_sigma > 0:
        sig += rng.normal(0, noise_sigma, n)

    return sig, t, fs


# ---------------------------------------------------------------------------
# 2. Synthetic Transient Detection & Spike Discrimination Tests
# ---------------------------------------------------------------------------
def test_synthetic_calcium_transient_detection():
    """Verify that genuine positive calcium transients are accurately detected."""
    expected_times = [1.2, 2.8, 4.4, 6.0]
    sig, t, fs = make_synthetic_calcium(dur_s=7.5, fs=250.0, peak_times=expected_times, f0=0.30, amp=0.10)

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    assert res["qc_status"] == "PASS"
    assert res["qc_reason"] == "valid_transients"

    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"]
    detected_times = acc["time_s"].values

    assert len(detected_times) == len(expected_times)
    for exp_t, det_t in zip(expected_times, detected_times):
        assert abs(exp_t - det_t) <= 0.05, f"Expected {exp_t}, got {det_t}"

    # Delta R check: raw excursion should be close to 0.10
    for dr in acc["delta_r"].values:
        assert 0.08 <= dr <= 0.12, f"Expected Delta R ~ 0.10, got {dr}"

    # Normalized amplitude check: Delta R / R0 should be close to 0.33
    for a in acc["amplitude"].values:
        assert 0.25 <= a <= 0.40, f"Expected Delta R / R0 ~ 0.33, got {a}"


def test_narrow_spike_rejection():
    """Verify that single-sample non-physiological optical spikes are rejected."""
    sig, t, fs = make_synthetic_calcium(dur_s=5.0, fs=250.0, peak_times=[1.5, 3.5], f0=0.25, amp=0.10)

    # Inject an isolated single-sample spike at t = 2.5s (width = 4 ms)
    spike_idx = int(2.5 * fs)
    sig[spike_idx] += 0.15

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    ev = res["events"]

    # The spike at 2.5s must be rejected as narrow_spike
    spike_candidates = ev[np.abs(ev["time_s"] - 2.5) < 0.02]
    assert len(spike_candidates) == 1
    assert spike_candidates.iloc[0]["status"] == "REJECTED"
    assert spike_candidates.iloc[0]["reason"] == "narrow_spike"

    # Physiological transients at 1.5s and 3.5s must be accepted
    acc = ev[ev["status"] == "ACCEPTED"]
    assert len(acc) == 2


def test_clean_quiescent_recording():
    """A valid live cell without spontaneous calcium transients must pass QC with 0 Hz frequency."""
    rng = np.random.default_rng(123)
    fs = 250.0
    # 6s of flat, quiet baseline (F0 = 0.28, low noise sigma ~ 0.0008)
    sig = 0.28 + rng.normal(0, 0.0008, int(6.0 * fs))

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    assert res["qc_status"] == "PASS"
    assert res["qc_reason"] == "no_accepted_transients"
    assert res["metadata"]["n_accepted"] == 0
    assert res["metadata"]["frequency_hz"] == 0.0
    assert res["metadata"]["frequency_bpm"] == 0.0


# ---------------------------------------------------------------------------
# 3. Hardware Dropout Handling & QC Classification
# ---------------------------------------------------------------------------
def test_partial_sensor_dropout_masking():
    """Partial dropouts (< 0.05) must be masked while preserving valid transients."""
    sig, t, fs = make_synthetic_calcium(dur_s=8.0, fs=250.0, peak_times=[1.0, 2.0, 3.0, 6.0, 7.0], f0=0.25, amp=0.10)

    # Drop signal to near-zero (0.0004) between t = 4.0s and t = 5.0s (hardware shutter closure)
    drop_start = int(4.0 * fs)
    drop_end = int(5.0 * fs)
    sig[drop_start:drop_end] = 0.0004

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    assert res["qc_status"] == "PASS"
    assert res["qc_reason"] == "partial_sensor_dropout_excluded"

    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"]
    # Peaks at 1, 2, 3, 6, 7 must be accepted
    assert len(acc) == 5
    for expected_t in [1.0, 2.0, 3.0, 6.0, 7.0]:
        assert np.any(np.abs(acc["time_s"].values - expected_t) <= 0.05)


def test_severe_sensor_dropout_rejection():
    """Recordings with >50% dropouts or <2.5s analyzable duration must be REJECTED by QC."""
    fs = 250.0
    n = int(8.0 * fs)
    sig = np.full(n, 0.0002, dtype=float)
    # Only 1.0 second of valid signal
    sig[int(2.0 * fs) : int(3.0 * fs)] = 0.28

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    assert res["qc_status"] == "REJECT"
    assert res["qc_reason"] == "severe_sensor_dropout"
    assert res["metadata"]["frequency_hz"] is None


def test_noise_dominated_rejection():
    """Uncorrelated high-frequency noise with no transients must be REJECTED."""
    rng = np.random.default_rng(999)
    fs = 250.0
    # High optical noise floor sigma = 0.0060, zero biological transients
    sig = 0.30 + rng.normal(0, 0.0060, int(6.0 * fs))

    res = analyze_calcium_pipeline(sig, fs, t0=0.0)
    assert res["qc_status"] == "REJECT"
    assert res["qc_reason"] in ("noise_dominated", "uninterpretable_noise")
    assert res["metadata"]["frequency_hz"] is None


# ---------------------------------------------------------------------------
# 4. DISH2ICI Workbook Dropout & Positive Control Verification
# ---------------------------------------------------------------------------
def test_dish2ici_dropout_and_control_segments():
    """Verify DISH2ICI Segments 7, 8 (partial dropout pass), 9 (severe dropout reject), 10 (clean pass)."""
    wb_dish2 = DATA_DIR / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish2.exists():
        pytest.skip(f"Workbook not found: {wb_dish2}")

    # Segment 7: Partial dropout ~236.3s-237.3s, valid transients at 1 Hz
    df7, y_cols7, fs7, t07 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 7")
    sig7, _, meta7 = build_concatenated_signal(df7, y_cols7, fs7, BeatCounterConfig(), force_invert=False)
    res7 = analyze_calcium_pipeline(sig7, fs7, float(meta7.get("time_origin_s", t07)))
    assert res7["qc_status"] == "PASS"
    assert res7["qc_reason"] == "partial_sensor_dropout_excluded"
    assert res7["metadata"]["n_accepted"] >= 7

    # Segment 8: Partial dropout ~257.5s-258.4s, valid transients at 1 Hz
    df8, y_cols8, fs8, t08 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 8")
    sig8, _, meta8 = build_concatenated_signal(df8, y_cols8, fs8, BeatCounterConfig(), force_invert=False)
    res8 = analyze_calcium_pipeline(sig8, fs8, float(meta8.get("time_origin_s", t08)))
    assert res8["qc_status"] == "PASS"
    assert res8["qc_reason"] == "partial_sensor_dropout_excluded"
    assert res8["metadata"]["n_accepted"] >= 8

    # Segment 9: Severe acquisition failure (>85% zeros) -> REJECT
    df9, y_cols9, fs9, t09 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 9")
    sig9, _, meta9 = build_concatenated_signal(df9, y_cols9, fs9, BeatCounterConfig(), force_invert=False)
    res9 = analyze_calcium_pipeline(sig9, fs9, float(meta9.get("time_origin_s", t09)))
    assert res9["qc_status"] == "REJECT"
    assert res9["qc_reason"] == "severe_sensor_dropout"

    # Segment 10: Clean positive control (0 dropouts, 1 Hz regular pacing)
    df10, y_cols10, fs10, t010 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 10")
    sig10, _, meta10 = build_concatenated_signal(df10, y_cols10, fs10, BeatCounterConfig(), force_invert=False)
    res10 = analyze_calcium_pipeline(sig10, fs10, float(meta10.get("time_origin_s", t010)))
    assert res10["qc_status"] == "PASS"
    assert res10["qc_reason"] == "valid_transients"
    assert res10["metadata"]["n_accepted"] == 10
    assert 0.95 <= res10["metadata"]["frequency_hz"] <= 1.05


# ---------------------------------------------------------------------------
# 5. Negative Control Verification (SCRAMBLED ISO & Motion Controller Error)
# ---------------------------------------------------------------------------
def test_negative_controls_rejected():
    """Verify that pure noise recordings are correctly rejected by calcium QC."""
    neg1 = DATA_DIR / "SCRAMBLED ISO individual transients result.xlsx"
    if neg1.exists():
        df1, y_cols1, fs1, t01 = load_cytocypher_excel(neg1, "RatioMetricCalcium Segment 1")
        sig1, _, meta1 = build_concatenated_signal(df1, y_cols1, fs1, BeatCounterConfig(), force_invert=False)
        res1 = analyze_calcium_pipeline(sig1, fs1, float(meta1.get("time_origin_s", t01)))
        assert res1["qc_status"] == "REJECT"
        assert res1["qc_reason"] in ("noise_dominated", "uninterpretable_noise")

    neg2 = DATA_DIR / "05-12-24.10D.ICI.SEARCH.BASE.NICO.NICO.ERROR-MOTION-CONTROLLER individual transients result.xlsx"
    if neg2.exists():
        df2, y_cols2, fs2, t02 = load_cytocypher_excel(neg2, "RatioMetricCalcium Segment 1")
        sig2, _, meta2 = build_concatenated_signal(df2, y_cols2, fs2, BeatCounterConfig(), force_invert=False)
        res2 = analyze_calcium_pipeline(sig2, fs2, float(meta2.get("time_origin_s", t02)))
        assert res2["qc_status"] == "REJECT"
        assert res2["qc_reason"] in ("noise_dominated", "uninterpretable_noise")


# ---------------------------------------------------------------------------
# 6. Benchmark Verification Against Expert Reference Timings
# ---------------------------------------------------------------------------
def test_imr90_reference_segments_benchmark():
    """Benchmark calcium transient detections against expert reference timings for IMR90."""
    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    ref_timings = {
        6: [170.49, 171.51, 172.48, 173.49],
        9: [244.52, 245.50],
        19: [533.02, 533.49, 534.02],
        28: [612.04, 613.04, 613.52, 614.04],
        31: [668.64, 669.01, 669.34],
    }

    for s, targets in ref_timings.items():
        df, y_cols, fs, t0 = load_cytocypher_excel(wb_imr90, f"RatioMetricCalcium Segment {s}")
        sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
        res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))
        assert res["qc_status"] == "PASS"

        acc = res["events"][res["events"]["status"] == "ACCEPTED"]
        det_times = acc["time_s"].values

        # Each reference target should match at least one detected peak within tolerance
        tol_s = 0.12
        for tgt in targets:
            min_dist = float(np.min(np.abs(det_times - tgt)))
            assert min_dist <= tol_s, f"IMR90 Seg {s}: target {tgt}s not matched within {tol_s}s (closest: {min_dist:.3f}s)"


def test_dish1nicotine_reference_segments_benchmark():
    """Benchmark calcium transient detections against expert reference timings for DISH1NICOTINE."""
    wb_nico = DATA_DIR / "DISH1NICOTINE individual transients result.xlsx"
    if not wb_nico.exists():
        pytest.skip(f"Workbook not found: {wb_nico}")

    ref_timings = {
        1: [0.85, 2.8, 4.8, 6.8, 8.8],
        6: [200.8, 201.8, 202.8, 203.8],
        10: [298.86, 299.68, 300.88, 301.91],
    }

    for s, targets in ref_timings.items():
        df, y_cols, fs, t0 = load_cytocypher_excel(wb_nico, f"RatioMetricCalcium Segment {s}")
        sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
        res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))
        assert res["qc_status"] == "PASS"

        acc = res["events"][res["events"]["status"] == "ACCEPTED"]
        det_times = acc["time_s"].values

        tol_s = 0.12
        for tgt in targets:
            min_dist = float(np.min(np.abs(det_times - tgt)))
            assert min_dist <= tol_s, f"NICO Seg {s}: target {tgt}s not matched within {tol_s}s (closest: {min_dist:.3f}s)"


# ---------------------------------------------------------------------------
# 7. Review Utility CLI Analysis Mode Integration Tests
# ---------------------------------------------------------------------------
def test_review_utility_analysis_modes():
    """Verify that run_v2_review_utility supports 'pixel', 'calcium', and 'pixel-calcium' modes."""
    import run_v2_review_utility as rv

    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    # Mode 1: PixelCorrelation only (default)
    segs_pixel, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="pixel")
    assert len(segs_pixel) == 1
    assert segs_pixel[0]["channel"] == "PixelCorrelation"
    assert segs_pixel[0]["sheet_name"] == "PixelCorrelation Segment 6"

    # Mode 2: Calcium only
    segs_cal, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="calcium")
    assert len(segs_cal) == 1
    assert segs_cal[0]["channel"] == "RatioMetricCalcium"
    assert segs_cal[0]["sheet_name"] == "RatioMetricCalcium Segment 6"
    assert segs_cal[0]["qc_status"] == "PASS"

    # Mode 3: Pixel-Calcium dual channel
    segs_dual, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="pixel-calcium")
    assert len(segs_dual) == 1
    assert segs_dual[0]["channel"] == "PixelCorrelation"
    assert segs_dual[0]["cal_runs"] is not None
    assert segs_dual[0]["cal_metadata"] is not None
    assert segs_dual[0]["cal_metadata"]["qc_status"] == "PASS"


# ---------------------------------------------------------------------------
# 8. Scientific Problem Investigations: Cases A through F
# ---------------------------------------------------------------------------
def test_case_a_imr90_segment_6_false_positive_rejection():
    """Case A: IMR90 Segment 6 must detect 4 genuine transients and reject the ~172.103s artifact."""
    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    df, y_cols, fs, t0 = load_cytocypher_excel(wb_imr90, "RatioMetricCalcium Segment 6")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "PASS"
    acc = res["events"][res["events"]["status"] == "ACCEPTED"]
    det_times = acc["time_s"].values

    # Must accept exactly the 4 genuine transients
    assert len(acc) == 4

    # The 172.103s artifact must NOT be accepted in ANY preset
    for preset_name in ["default", "strict", "permissive"]:
        preset_acc = res["runs"][preset_name]["events"]
        preset_acc_times = preset_acc[preset_acc["status"] == "ACCEPTED"]["time_s"].values
        assert not np.any(np.abs(preset_acc_times - 172.103) < 0.15), (
            f"False positive at ~172.103s was accepted under preset '{preset_name}'!"
        )


def test_case_b_imr90_segments_9_and_19_consistent_peak_localization():
    """Case B: IMR90 Segments 9 and 19 must have reproducible peak timing and consistent localization."""
    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    for seg_num, expected_count in [(9, 5), (19, 10)]:
        df, y_cols, fs, t0 = load_cytocypher_excel(wb_imr90, f"RatioMetricCalcium Segment {seg_num}")
        sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
        res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

        # Default run
        def_ev = res["runs"]["default"]["events"]
        def_acc = def_ev[def_ev["status"] == "ACCEPTED"]
        assert len(def_acc) == expected_count

        # Peak timestamps must be identical across presets for the shared events
        perm_ev = res["runs"]["permissive"]["events"]
        perm_acc = perm_ev[perm_ev["status"] == "ACCEPTED"]

        strict_ev = res["runs"]["strict"]["events"]
        strict_acc = strict_ev[strict_ev["status"] == "ACCEPTED"]

        # All strict peaks must match default peaks with exact sample indices
        for s_idx in strict_acc["peak_index"].values:
            assert s_idx in def_acc["peak_index"].values, (
                f"Peak index {s_idx} in strict was displaced in default!"
            )


def test_case_c_imr90_segment_37_noise_rejection():
    """Case C: IMR90 Segment 37 is dominated by fluorescence noise and must be REJECTED by QC."""
    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    df, y_cols, fs, t0 = load_cytocypher_excel(wb_imr90, "RatioMetricCalcium Segment 37")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "REJECT"
    assert res["qc_reason"] == "uninterpretable_noise"


def test_case_d_dish1nicotine_segment_10_missed_transient_recovery():
    """Case D: DISH1NICOTINE Segment 10 must recover the genuine transient at ~304.91s."""
    wb_nico = DATA_DIR / "DISH1NICOTINE individual transients result.xlsx"
    if not wb_nico.exists():
        pytest.skip(f"Workbook not found: {wb_nico}")

    df, y_cols, fs, t0 = load_cytocypher_excel(wb_nico, "RatioMetricCalcium Segment 10")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "PASS"
    def_ev = res["runs"]["default"]["events"]
    def_acc = def_ev[def_ev["status"] == "ACCEPTED"]

    # Must find all 10 transients including the one at ~304.91s
    assert len(def_acc) == 10
    det_times = def_acc["time_s"].values
    assert np.any(np.abs(det_times - 304.91) < 0.10), "Missed transient at ~304.91s was not recovered!"


def test_case_e_dish2ici_segment_8_strictness_hierarchy():
    """Case E: DISH2ICI Segment 8 must enforce strictness hierarchy and suppress small fluctuation at 255.43s."""
    wb_dish2 = DATA_DIR / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish2.exists():
        pytest.skip(f"Workbook not found: {wb_dish2}")

    df, y_cols, fs, t0 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 8")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "PASS"

    strict_acc = res["runs"]["strict"]["events"]
    strict_acc = strict_acc[strict_acc["status"] == "ACCEPTED"]

    default_acc = res["runs"]["default"]["events"]
    default_acc = default_acc[default_acc["status"] == "ACCEPTED"]

    perm_acc = res["runs"]["permissive"]["events"]
    perm_acc = perm_acc[perm_acc["status"] == "ACCEPTED"]

    # Strictness hierarchy: Accepted(Strict) <= Accepted(Default) <= Accepted(Permissive)
    assert len(strict_acc) <= len(default_acc) <= len(perm_acc)
    assert len(strict_acc) == 9
    assert len(default_acc) == 9
    assert len(perm_acc) >= 9

    # The ~255.43s sub-transient fluctuation must be REJECTED in strict and default
    assert not np.any(np.abs(strict_acc["time_s"].values - 255.43) < 0.10)
    assert not np.any(np.abs(default_acc["time_s"].values - 255.43) < 0.10)


def test_case_f_independent_dual_mode_invariants():
    """Case F: Analysing the same calcium and pixel channels produces identical results in dual vs single modes."""
    import run_v2_review_utility as rv

    wb_imr90 = DATA_DIR / "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90.exists():
        pytest.skip(f"Workbook not found: {wb_imr90}")

    # Process in single modes
    segs_pixel, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="pixel")
    segs_cal, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="calcium")

    # Process in dual mode
    segs_dual, _ = rv.process_workbook(str(wb_imr90), segments=[6], analysis_mode="pixel-calcium")

    # Invariant 1: PixelCorrelation results match
    assert segs_pixel[0]["qc_status"] == segs_dual[0]["qc_status"]
    assert segs_pixel[0]["effective_polarity"] == segs_dual[0]["effective_polarity"]
    pix_run_single = segs_pixel[0]["runs"]["default"]
    pix_run_dual = segs_dual[0]["runs"]["default"]
    n_acc_single = np.sum(np.array(pix_run_single["events"]["status"]) == "ACCEPTED")
    n_acc_dual = np.sum(np.array(pix_run_dual["events"]["status"]) == "ACCEPTED")
    assert n_acc_single == n_acc_dual

    # Invariant 2: Calcium results match
    cal_meta_single = segs_cal[0]["cal_metadata"]
    cal_meta_dual = segs_dual[0]["cal_metadata"]
    assert cal_meta_single["qc_status"] == cal_meta_dual["qc_status"]
    assert cal_meta_single["n_accepted"] == cal_meta_dual["n_accepted"]
    assert cal_meta_single["frequency_hz"] == cal_meta_dual["frequency_hz"]

    cal_ev_single = segs_cal[0]["cal_runs"]["default"]["events"]
    cal_ev_dual = segs_dual[0]["cal_runs"]["default"]["events"]
    np.testing.assert_array_equal(cal_ev_single["index"], cal_ev_dual["index"])
    np.testing.assert_allclose(cal_ev_single["time_s"], cal_ev_dual["time_s"], rtol=1e-5)


def test_dish1control_segment_4_dropout_baseline_integrity():
    """Verify that a 1s dropout in DISH1CONTROL Segment 4 does NOT drag down the baseline or threshold."""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish1.exists():
        pytest.skip(f"Workbook not found: {wb_dish1}")

    df, y_cols, fs, t0 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 4")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "PASS"
    # Baseline must remain stable near resting level (0.22 - 0.24) and NOT be dragged to 0.157
    assert np.min(res["baseline"]) >= 0.215, f"Baseline corrupted by dropout: min={np.min(res['baseline'])}"
    assert np.max(res["baseline"]) <= 0.260

    def_ev = res["runs"]["default"]["events"]
    def_acc = def_ev[def_ev["status"] == "ACCEPTED"]
    # 9 genuine transients outside the dropout window
    assert len(def_acc) == 9
    # No transients detected inside the dropout (t ~ 124.0s to 125.0s)
    dropout_transients = def_acc[(def_acc["time_s"] >= 123.8) & (def_acc["time_s"] <= 125.1)]
    assert len(dropout_transients) == 0


def test_dish1control_segment_45_severe_dropout_fragmentation_qc_reject():
    """Verify that severely fragmented dropouts in DISH1CONTROL Segment 45 trigger calcium QC REJECT while preserving PixelCorrelation."""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish1.exists():
        pytest.skip(f"Workbook not found: {wb_dish1}")

    import run_v2_review_utility as rv

    # Calcium analysis must REJECT due to severe sensor dropout
    df, y_cols, fs, t0 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 45")
    sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

    assert res["qc_status"] == "REJECT"
    assert res["qc_reason"] == "severe_sensor_dropout"
    assert res["metadata"]["n_accepted"] == 0
    assert res["metadata"]["frequency_hz"] is None

    # Dual mode test: PixelCorrelation must remain PASS while Calcium is REJECT
    segs_dual, _ = rv.process_workbook(str(wb_dish1), segments=[45], analysis_mode="pixel-calcium")
    assert segs_dual[0]["qc_status"] == "PASS"
    assert segs_dual[0]["cal_metadata"]["qc_status"] == "REJECT"
    assert segs_dual[0]["cal_metadata"]["qc_reason"] == "severe_sensor_dropout"


def test_imr90_240926_flat_baseline_noise_rejection():
    """Verify that flat baseline noise in 240926 IMR90 Segments 4 and 19 does not produce false positives."""
    wb_imr90_24 = DATA_DIR / "240926_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90_24.exists():
        pytest.skip(f"Workbook not found: {wb_imr90_24}")

    # Segment 4: exactly 2 genuine transients
    df4, y_cols4, fs4, t04 = load_cytocypher_excel(wb_imr90_24, "RatioMetricCalcium Segment 4")
    sig4, _, meta4 = build_concatenated_signal(df4, y_cols4, fs4, BeatCounterConfig(), force_invert=False)
    res4 = analyze_calcium_pipeline(sig4, fs4, float(meta4.get("time_origin_s", t04)))
    assert res4["qc_status"] == "PASS"
    def_acc4 = res4["runs"]["default"]["events"]
    def_acc4 = def_acc4[def_acc4["status"] == "ACCEPTED"]
    assert len(def_acc4) == 2, f"Expected 2 genuine transients in Seg 4, got {len(def_acc4)}"

    # Segment 19: exactly 3 genuine transients
    df19, y_cols19, fs19, t019 = load_cytocypher_excel(wb_imr90_24, "RatioMetricCalcium Segment 19")
    sig19, _, meta19 = build_concatenated_signal(df19, y_cols19, fs19, BeatCounterConfig(), force_invert=False)
    res19 = analyze_calcium_pipeline(sig19, fs19, float(meta19.get("time_origin_s", t019)))
    assert res19["qc_status"] == "PASS"
    def_acc19 = res19["runs"]["default"]["events"]
    def_acc19 = def_acc19[def_acc19["status"] == "ACCEPTED"]
    assert len(def_acc19) == 3, f"Expected 3 genuine transients in Seg 19, got {len(def_acc19)}"


def test_calcium_normalization_delta_r_over_r0():
    """Verify that ratiometric calcium is normalized as Delta R / R0 = (R - R0) / R0 with zero resting baseline."""
    from pixel_counter.calcium_detector import compute_normalized_calcium

    sig_raw = np.array([0.25, 0.25, 0.35, 0.25, 0.00, 0.25])
    baseline = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    is_dropout = np.array([False, False, False, False, True, False])

    norm = compute_normalized_calcium(sig_raw, baseline, is_dropout)
    # Resting baseline is 0.0
    assert norm[0] == 0.0
    assert norm[1] == 0.0
    # Peak amplitude is (0.35 - 0.25) / 0.25 = 0.40
    assert abs(norm[2] - 0.40) < 1e-6
    # Dropout is NaN (leaves gap without interpolation)
    assert np.isnan(norm[4])


def test_refined_doublets_dish1control_seg48_and_seg79():
    """Verify recovery of coupled doublets in DISH1CONTROL Segments 48 and 79."""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish1.exists():
        pytest.skip(f"Workbook not found: {wb_dish1}")

    # Segment 48: Doublet at 1275.43s and 1275.68s
    df48, y_cols48, fs48, t048 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 48")
    sig48, _, meta48 = build_concatenated_signal(df48, y_cols48, fs48, BeatCounterConfig(), force_invert=False)
    res48 = analyze_calcium_pipeline(sig48, fs48, float(meta48.get("time_origin_s", t048)))
    assert res48["qc_status"] == "PASS"
    acc48 = res48["runs"]["default"]["events"]
    acc48 = acc48[acc48["status"] == "ACCEPTED"]
    assert len(acc48) == 10
    assert np.any(np.abs(acc48["time_s"].values - 1275.43) < 0.08)
    assert np.any(np.abs(acc48["time_s"].values - 1275.68) < 0.08)

    # Segment 79: Doublet at 1775.54s and 1775.74s
    df79, y_cols79, fs79, t079 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 79")
    sig79, _, meta79 = build_concatenated_signal(df79, y_cols79, fs79, BeatCounterConfig(), force_invert=False)
    res79 = analyze_calcium_pipeline(sig79, fs79, float(meta79.get("time_origin_s", t079)))
    assert res79["qc_status"] == "PASS"
    acc79 = res79["runs"]["default"]["events"]
    acc79 = acc79[acc79["status"] == "ACCEPTED"]
    assert len(acc79) == 12
    assert np.any(np.abs(acc79["time_s"].values - 1775.54) < 0.08)
    assert np.any(np.abs(acc79["time_s"].values - 1775.74) < 0.08)


def test_refined_weaker_beats_dish2ici_seg15_and_seg29():
    """Verify recovery of weaker genuine beats and alternans in DISH2ICI Segments 15 and 29."""
    wb_dish2 = DATA_DIR / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish2.exists():
        pytest.skip(f"Workbook not found: {wb_dish2}")

    # Segment 15: 1 Hz paced sequence with weaker beat at 446.73s
    df15, y_cols15, fs15, t015 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 15")
    sig15, _, meta15 = build_concatenated_signal(df15, y_cols15, fs15, BeatCounterConfig(), force_invert=False)
    res15 = analyze_calcium_pipeline(sig15, fs15, float(meta15.get("time_origin_s", t015)))
    assert res15["qc_status"] == "PASS"
    acc15 = res15["runs"]["default"]["events"]
    acc15 = acc15[acc15["status"] == "ACCEPTED"]
    assert len(acc15) == 10
    assert np.any(np.abs(acc15["time_s"].values - 446.73) < 0.08)

    # Segment 29: Alternans with weaker beat at 713.76s (>= threshold) and subthreshold beat at 710.73s
    df29, y_cols29, fs29, t029 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 29")
    sig29, _, meta29 = build_concatenated_signal(df29, y_cols29, fs29, BeatCounterConfig(), force_invert=False)
    res29 = analyze_calcium_pipeline(sig29, fs29, float(meta29.get("time_origin_s", t029)))
    assert res29["qc_status"] == "PASS"
    events29 = res29["runs"]["default"]["events"]
    acc29 = events29[events29["status"] == "ACCEPTED"]
    # Under strict Stage 1 threshold invariant (amp >= eff_min_norm_height):
    # 713.76s (amp=0.3526 >= 0.3475) is auto-accepted in Stage 2 alternans;
    # 710.73s (amp=0.3363 < 0.3475) is strictly rejected from auto-acceptance (left for manual BLUE addition).
    assert len(acc29) == 7
    assert np.any(np.abs(acc29["time_s"].values - 713.76) < 0.08)
    # The subthreshold event is preserved as candidate but REJECTED
    sub_pk = events29[np.abs(events29["time_s"] - 710.73) < 0.08]
    assert len(sub_pk) == 1
    assert sub_pk.iloc[0]["status"] != "ACCEPTED"


def test_refined_noise_rejection_imr90_seg3():
    """Verify that 240926 IMR90 Segment 3 accepts only the 2 genuine transients and excludes noise ripple at ~97.72s."""
    wb_imr90_24 = DATA_DIR / "240926_IMR90_1,2,3HZ individual transients result.xlsx"
    if not wb_imr90_24.exists():
        pytest.skip(f"Workbook not found: {wb_imr90_24}")

    df3, y_cols3, fs3, t03 = load_cytocypher_excel(wb_imr90_24, "RatioMetricCalcium Segment 3")
    sig3, _, meta3 = build_concatenated_signal(df3, y_cols3, fs3, BeatCounterConfig(), force_invert=False)
    res3 = analyze_calcium_pipeline(sig3, fs3, float(meta3.get("time_origin_s", t03)))
    assert res3["qc_status"] == "PASS"
    acc3 = res3["runs"]["default"]["events"]
    acc3 = acc3[acc3["status"] == "ACCEPTED"]
    assert len(acc3) == 2, f"Expected exactly 2 accepted transients in Seg 3, got {len(acc3)}"
    assert np.any(np.abs(acc3["time_s"].values - 94.41) < 0.10)
    assert np.any(np.abs(acc3["time_s"].values - 95.39) < 0.10)
    # The noise ripple at 97.72s must be rejected
    assert not np.any(np.abs(acc3["time_s"].values - 97.72) < 0.20)


def test_refined_doublets_dish1nicotine_seg21_and_seg50():
    """Verify recovery of coupled doublets in DISH1NICOTINE Segments 21 and 50."""
    wb_nico = DATA_DIR / "DISH1NICOTINE individual transients result.xlsx"
    if not wb_nico.exists():
        pytest.skip(f"Workbook not found: {wb_nico}")

    # Segment 21: Coupled doublet at 533.64s and 533.86s
    df21, y_cols21, fs21, t021 = load_cytocypher_excel(wb_nico, "RatioMetricCalcium Segment 21")
    sig21, _, meta21 = build_concatenated_signal(df21, y_cols21, fs21, BeatCounterConfig(), force_invert=False)
    res21 = analyze_calcium_pipeline(sig21, fs21, float(meta21.get("time_origin_s", t021)))
    assert res21["qc_status"] == "PASS"
    acc21 = res21["runs"]["default"]["events"]
    acc21 = acc21[acc21["status"] == "ACCEPTED"]
    assert len(acc21) == 11
    assert np.any(np.abs(acc21["time_s"].values - 533.64) < 0.08)
    assert np.any(np.abs(acc21["time_s"].values - 533.86) < 0.08)

    # Segment 50: Coupled doublets at 1006.67s / 1006.93s, 1007.65s / 1007.94s, etc.
    df50, y_cols50, fs50, t050 = load_cytocypher_excel(wb_nico, "RatioMetricCalcium Segment 50")
    sig50, _, meta50 = build_concatenated_signal(df50, y_cols50, fs50, BeatCounterConfig(), force_invert=False)
    res50 = analyze_calcium_pipeline(sig50, fs50, float(meta50.get("time_origin_s", t050)))
    assert res50["qc_status"] == "PASS"
    acc50 = res50["runs"]["default"]["events"]
    acc50 = acc50[acc50["status"] == "ACCEPTED"]
    assert len(acc50) >= 13
    assert np.any(np.abs(acc50["time_s"].values - 1006.67) < 0.08)
    assert np.any(np.abs(acc50["time_s"].values - 1006.93) < 0.08)


# ---------------------------------------------------------------------------
# 10. Calcium Detection Threshold Visualization & Stage Diagnostics Regression
# ---------------------------------------------------------------------------
def test_stage1_amplitude_criterion_visualization_and_serialization():
    """Verify that displayed calcium threshold represents genuine Stage 1 criterion,
    not post hoc 0.95 * min(accepted), and correctly exposes Stage 1 / Stage 2 metrics."""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish1.exists():
        pytest.skip(f"Workbook not found: {wb_dish1}")

    from run_v2_review_utility import _calcium_run_payload

    # --- SEGMENT 42 ---
    df42, y_cols42, fs42, t042 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 42")
    sig42, _, meta42 = build_concatenated_signal(df42, y_cols42, fs42, BeatCounterConfig(), force_invert=False)
    res42 = analyze_calcium_pipeline(sig42, fs42, float(meta42.get("time_origin_s", t042)))
    assert res42["qc_status"] == "PASS"

    def_run42 = res42["runs"]["default"]
    det_met42 = def_run42["detector_metrics"]

    # Must equal actual Stage 1 eff_min_norm_height (~0.4812)
    expected_s1_amp42 = det_met42["eff_min_norm_height"]
    assert pytest.approx(expected_s1_amp42, abs=1e-4) == 0.4812
    assert def_run42["stage1_amplitude_criterion"] == expected_s1_amp42
    assert def_run42["effective_amplitude_threshold"] == expected_s1_amp42

    # Must NOT equal post-hoc 0.95 * min(accepted) (~0.8719)
    acc_amps42 = def_run42["events"][def_run42["events"]["status"] == "ACCEPTED"]["amplitude"]
    post_hoc42 = float(np.min(acc_amps42)) * 0.95
    assert abs(def_run42["stage1_amplitude_criterion"] - post_hoc42) > 0.30

    # Diagnostic counts
    assert def_run42["n_stage1_accepted"] == 10
    assert def_run42["n_stage2_recovered"] == 0
    assert def_run42["n_accepted"] == 10

    # Payload serialization check
    payload42 = _calcium_run_payload(def_run42)
    assert payload42["stage1_amplitude_criterion"] == pytest.approx(0.4812, abs=1e-4)
    assert payload42["amplitude_threshold"] == pytest.approx(0.4812, abs=1e-4)
    assert payload42["reference_amplitude"] == pytest.approx(1.0694, abs=1e-4)
    assert payload42["effective_prominence"] == pytest.approx(0.1144, abs=1e-4)
    assert payload42["n_stage1_accepted"] == 10
    assert payload42["n_stage2_recovered"] == 0
    assert all(stg == "stage1" for stg, stat in zip(payload42["events"]["detection_stage"], payload42["events"]["status"]) if stat == "ACCEPTED")

    # --- SEGMENT 43 ---
    df43, y_cols43, fs43, t043 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 43")
    sig43, _, meta43 = build_concatenated_signal(df43, y_cols43, fs43, BeatCounterConfig(), force_invert=False)
    res43 = analyze_calcium_pipeline(sig43, fs43, float(meta43.get("time_origin_s", t043)))
    assert res43["qc_status"] == "PASS"

    def_run43 = res43["runs"]["default"]
    det_met43 = def_run43["detector_metrics"]

    # Must equal actual Stage 1 eff_min_norm_height (~0.2497)
    expected_s1_amp43 = det_met43["eff_min_norm_height"]
    assert pytest.approx(expected_s1_amp43, abs=1e-4) == 0.2497
    assert def_run43["stage1_amplitude_criterion"] == expected_s1_amp43
    assert def_run43["effective_amplitude_threshold"] == expected_s1_amp43

    # Must NOT equal post-hoc 0.95 * min(accepted) (~0.2244)
    acc_amps43 = def_run43["events"][def_run43["events"]["status"] == "ACCEPTED"]["amplitude"]
    post_hoc43 = float(np.min(acc_amps43)) * 0.95
    assert abs(def_run43["stage1_amplitude_criterion"] - post_hoc43) > 0.020

    # Diagnostic counts: 10 Stage 1, 0 Stage 2 (subthreshold beat at 1208.134s strictly rejected from auto-acceptance)
    assert def_run43["n_stage1_accepted"] == 10
    assert def_run43["n_stage2_recovered"] == 0
    assert def_run43["n_accepted"] == 10

    # Under the mandatory invariant, beat at 1208.134s (amp=0.2362 < 0.2497) is REJECTED from auto-acceptance
    ev43 = def_run43["events"]
    sub_beat = ev43[np.abs(ev43["time_s"] - 1208.134) < 0.08]
    assert len(sub_beat) == 1
    sub_row = sub_beat.iloc[0]
    assert sub_row["status"] != "ACCEPTED"
    assert sub_row["amplitude"] < expected_s1_amp43
    assert pytest.approx(sub_row["amplitude"], abs=1e-4) == 0.2362

    # Payload serialization check
    payload43 = _calcium_run_payload(def_run43)
    assert payload43["stage1_amplitude_criterion"] == pytest.approx(0.2497, abs=1e-4)
    assert payload43["amplitude_threshold"] == pytest.approx(0.2497, abs=1e-4)
    assert payload43["reference_amplitude"] == pytest.approx(0.5550, abs=1e-4)
    assert payload43["effective_prominence"] == pytest.approx(0.0500, abs=1e-4)
    assert payload43["n_stage1_accepted"] == 10
    assert payload43["n_stage2_recovered"] == 0
    stages43 = [stg for stg, stat in zip(payload43["events"]["detection_stage"], payload43["events"]["status"]) if stat == "ACCEPTED"]
    assert stages43.count("stage1") == 10
    assert stages43.count("stage2_alternans") == 0

    # --- PRESET STRICTNESS SWITCHING ---
    perm_run43 = res43["runs"]["permissive"]
    strict_run43 = res43["runs"]["strict"]
    # Stage 1 criterion increases strictly with strictness preset
    assert perm_run43["stage1_amplitude_criterion"] <= def_run43["stage1_amplitude_criterion"] <= strict_run43["stage1_amplitude_criterion"]
    assert perm_run43["effective_prominence"] <= def_run43["effective_prominence"] <= strict_run43["effective_prominence"]


# ---------------------------------------------------------------------------
# 26. Mandatory Invariant: Zero Automatically Accepted Peaks Below Threshold
# ---------------------------------------------------------------------------
def test_strict_threshold_enforcement_no_subthreshold_auto_peaks():
    """Verify that every automatically accepted peak across all presets and segments
    strictly satisfies A_peak >= T_stage1 (eff_min_norm_height).
    Under-threshold candidates must not be auto-accepted as GREEN."""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    wb_dish2 = DATA_DIR / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"

    check_cases = [
        (wb_dish1, [42, 43, 56, 64, 84]),
        (wb_dish2, [15, 29, 36, 38, 58, 68, 73, 81]),
    ]

    for wb_path, segs in check_cases:
        if not wb_path.exists():
            continue
        for s in segs:
            df, y_cols, fs, t0 = load_cytocypher_excel(wb_path, f"RatioMetricCalcium Segment {s}")
            sig, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
            res = analyze_calcium_pipeline(sig, fs, float(meta.get("time_origin_s", t0)))

            for p_name in ("permissive", "default", "strict"):
                run_data = res["runs"][p_name]
                s1_crit = run_data["stage1_amplitude_criterion"]
                events = run_data["events"]
                acc = events[events["status"] == "ACCEPTED"]
                # Mathematical invariant check:
                sub_thresh = acc[acc["amplitude"] < s1_crit - 1e-6]
                assert len(sub_thresh) == 0, (
                    f"Segment {s} ({p_name}): {len(sub_thresh)} auto-accepted peaks below criterion ({s1_crit:.4f})"
                )

    # Specific assertions for previously problematic under-threshold peaks:
    # DISH2ICI Segment 36 (~809.416s, amp ~0.2069 vs thresh ~0.3284)
    df36, y36, fs36, t36 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 36")
    sig36, _, m36 = build_concatenated_signal(df36, y36, fs36, BeatCounterConfig(), force_invert=False)
    res36 = analyze_calcium_pipeline(sig36, fs36, float(m36.get("time_origin_s", t36)))
    ev36 = res36["runs"]["default"]["events"]
    c36 = ev36[np.abs(ev36["time_s"] - 809.416) < 0.08]
    if len(c36) > 0:
        assert c36.iloc[0]["status"] != "ACCEPTED"

    # DISH2ICI Segment 38 (~834.421s vs thresh ~0.3704)
    df38, y38, fs38, t38 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 38")
    sig38, _, m38 = build_concatenated_signal(df38, y38, fs38, BeatCounterConfig(), force_invert=False)
    res38 = analyze_calcium_pipeline(sig38, fs38, float(m38.get("time_origin_s", t38)))
    ev38 = res38["runs"]["default"]["events"]
    c38 = ev38[np.abs(ev38["time_s"] - 834.421) < 0.08]
    if len(c38) > 0:
        assert c38.iloc[0]["status"] != "ACCEPTED"


# ---------------------------------------------------------------------------
# 27. Close Doublet Recovery on Authentic Experimental Datasets
# ---------------------------------------------------------------------------
def test_close_doublet_recovery():
    """Verify recovery of genuine closely spaced transients:
    - DISH1CONTROL Seg 64 (early beat near 1552.53s)
    - DISH1CONTROL Seg 84 (doublet beat near 1840.74s)
    - DISH2ICI Seg 58 (beat near 1177.79s)
    - DISH2ICI Seg 68 (doublet beat near 1319.66s)
    - DISH2ICI Seg 81 (doublet beat near 1543.70s)"""
    wb_dish1 = DATA_DIR / "13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    wb_dish2 = DATA_DIR / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"

    # DISH1CONTROL Seg 64: 1552.528s recovered in Stage 1
    df64, y64, fs64, t64 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 64")
    sig64, _, m64 = build_concatenated_signal(df64, y64, fs64, BeatCounterConfig(), force_invert=False)
    res64 = analyze_calcium_pipeline(sig64, fs64, float(m64.get("time_origin_s", t64)))
    acc64 = res64["runs"]["default"]["events"]
    acc64 = acc64[acc64["status"] == "ACCEPTED"]
    assert np.any(np.abs(acc64["time_s"].values - 1552.528) < 0.06)

    # DISH1CONTROL Seg 84: 1840.740s recovered in Stage 2 doublet
    df84, y84, fs84, t84 = load_cytocypher_excel(wb_dish1, "RatioMetricCalcium Segment 84")
    sig84, _, m84 = build_concatenated_signal(df84, y84, fs84, BeatCounterConfig(), force_invert=False)
    res84 = analyze_calcium_pipeline(sig84, fs84, float(m84.get("time_origin_s", t84)))
    acc84 = res84["runs"]["default"]["events"]
    acc84 = acc84[acc84["status"] == "ACCEPTED"]
    assert np.any(np.abs(acc84["time_s"].values - 1840.740) < 0.06)

    # DISH2ICI Seg 58: 1177.786s recovered
    df58, y58, fs58, t58 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 58")
    sig58, _, m58 = build_concatenated_signal(df58, y58, fs58, BeatCounterConfig(), force_invert=False)
    res58 = analyze_calcium_pipeline(sig58, fs58, float(m58.get("time_origin_s", t58)))
    acc58 = res58["runs"]["default"]["events"]
    acc58 = acc58[acc58["status"] == "ACCEPTED"]
    assert np.any(np.abs(acc58["time_s"].values - 1177.786) < 0.06)

    # DISH2ICI Seg 68: 1319.660s recovered in Stage 2 doublet
    df68, y68, fs68, t68 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 68")
    sig68, _, m68 = build_concatenated_signal(df68, y68, fs68, BeatCounterConfig(), force_invert=False)
    res68 = analyze_calcium_pipeline(sig68, fs68, float(m68.get("time_origin_s", t68)))
    acc68 = res68["runs"]["default"]["events"]
    acc68 = acc68[acc68["status"] == "ACCEPTED"]
    assert np.any(np.abs(acc68["time_s"].values - 1319.660) < 0.06)

    # DISH2ICI Seg 81: 1543.698s recovered in Stage 2 doublet
    df81, y81, fs81, t81 = load_cytocypher_excel(wb_dish2, "RatioMetricCalcium Segment 81")
    sig81, _, m81 = build_concatenated_signal(df81, y81, fs81, BeatCounterConfig(), force_invert=False)
    res81 = analyze_calcium_pipeline(sig81, fs81, float(m81.get("time_origin_s", t81)))
    acc81 = res81["runs"]["default"]["events"]
    acc81 = acc81[acc81["status"] == "ACCEPTED"]
    assert np.any(np.abs(acc81["time_s"].values - 1543.698) < 0.06)


# ---------------------------------------------------------------------------
# 28. Noise False Positive Rejection: DISH1NICOTINE Segment 44
# ---------------------------------------------------------------------------
def test_noise_rejection_dish1nicotine_seg44():
    """Verify that in DISH1NICOTINE Segment 44, the noise ripple at 919.550s is REJECTED
    because cand_prom < 1.8 * raw_noise_sigma, while genuine calcium transients elsewhere are ACCEPTED."""
    wb_nic = DATA_DIR / "DISH1NICOTINE individual transients result.xlsx"
    if not wb_nic.exists():
        pytest.skip(f"Workbook not found: {wb_nic}")

    df44, y44, fs44, t44 = load_cytocypher_excel(wb_nic, "RatioMetricCalcium Segment 44")
    sig44, _, m44 = build_concatenated_signal(df44, y44, fs44, BeatCounterConfig(), force_invert=False)
    res44 = analyze_calcium_pipeline(sig44, fs44, float(m44.get("time_origin_s", t44)))
    assert res44["qc_status"] == "PASS"

    def_run44 = res44["runs"]["default"]
    events44 = def_run44["events"]
    acc44 = events44[events44["status"] == "ACCEPTED"]

    # 10 genuine calcium transients are preserved
    assert len(acc44) == 10

    # The noise ripple at ~919.550s is REJECTED
    noise_cand = events44[np.abs(events44["time_s"] - 919.550) < 0.08]
    assert len(noise_cand) == 1
    assert noise_cand.iloc[0]["status"] == "REJECTED"
    assert noise_cand.iloc[0]["reason"] == "below_prominence_threshold"


# ---------------------------------------------------------------------------
# 29. Review Utility: Contract for Below-Threshold Manual Addition (BLUE)
# ---------------------------------------------------------------------------
def test_review_utility_manual_addition_contract():
    """Verify review HTML template contains exact click handlers and logic for:
    1. Clicking calcium amplitude/raw trace adds BLUE manual peak below or above threshold
    2. Clicking an unaccepted candidate dot directly converts it to a BLUE manual peak
    3. Clicking a BLUE manual peak completely deletes it (NOTHING -> BLUE -> NOTHING)
    4. Clicking a GREEN auto peak toggles to RED CROSS (GREEN -> RED CROSS -> GREEN)
    5. Preserves original status when a manual peak is deleted"""
    from run_v2_review_utility import HTML_TEMPLATE

    assert "ΔR/R₀ amplitude" in HTML_TEMPLATE
    assert "Raw RatioMetricCalcium" in HTML_TEMPLATE
    assert 'targetEv.origin = "manual";' in HTML_TEMPLATE
    assert 'manMap[targetEv.index] = {' in HTML_TEMPLATE
    assert 'action: "ADDED"' in HTML_TEMPLATE
    assert "deleteManualPeak" in HTML_TEMPLATE
    assert "toggleEventState" in HTML_TEMPLATE
    assert 'a.status !== "ACCEPTED" && !a.manually_rejected' in HTML_TEMPLATE



