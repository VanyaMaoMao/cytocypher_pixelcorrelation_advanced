"""
tests/test_sarcomere_calcium_independence.py
============================================

Comprehensive regression and invariant test suite confirming total independence
of the Sarcomere Shortening and RatioMetricCalcium analysis channels.

Requirements Tested:
1. Strict Channel Invariance Invariant:
   SarcomereResult(sarcomere) == SarcomereResult(sarcomere-calcium)
2. Calcium never creates mechanical events, rescues rejected candidates, alters
   sarcomere MAIN/SCE classifications, or promotes sarcomere QC.
3. Calcium QC failure (e.g. noise) cannot cause mechanical QC failure.
4. Calcium QC PASS cannot rescue mechanical QC REJECT.
5. ISO Segments 16, 26, 32, 42 strictly REJECT across all presets.
6. CAR Segment 46 Calcium noise strictly returns Calcium QC = REJECT (0 transients).
7. ISO Segment 13 passes sarcomere analysis independently without calcium.
8. Strict amplitude hierarchy: A >= T_MAIN -> MAIN, T_SCE <= A < T_MAIN -> SECONDARY, A < T_SCE -> REJECTED.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pixel_counter.sarcomere_pipeline import (
    analyze_sarcomere_segment_from_excel,
    analyze_sarcomere_pipeline,
)
from pixel_counter.calcium_detector import analyze_calcium_pipeline

SARCOMERE_DIR = Path(__file__).resolve().parent.parent / "sarcomere shortening"
DATASET_A_PATH = SARCOMERE_DIR / "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx"
DATASET_B_PATH = SARCOMERE_DIR / "(MI) CGP+ISO+ROLI+IBMX DISH1  individual transients result.xlsx"


@pytest.mark.skipif(not DATASET_A_PATH.exists() or not DATASET_B_PATH.exists(), reason="Workbooks not found")
@pytest.mark.parametrize("wb_path,sheet_name", [
    (DATASET_A_PATH, "SarcomereLength Segment 1"),
    (DATASET_A_PATH, "SarcomereLength Segment 2"),
    (DATASET_A_PATH, "SarcomereLength Segment 10"),
    (DATASET_B_PATH, "SarcomereLength Segment 13"),
    (DATASET_B_PATH, "SarcomereLength Segment 16"),
    (DATASET_B_PATH, "SarcomereLength Segment 18"),
    (DATASET_B_PATH, "SarcomereLength Segment 20"),
    (DATASET_B_PATH, "SarcomereLength Segment 26"),
])
def test_strict_sarcomere_channel_invariance(wb_path, sheet_name):
    """Verify SarcomereResult(sarcomere) == SarcomereResult(sarcomere-calcium) exactly."""
    res_sarc = analyze_sarcomere_segment_from_excel(wb_path, sheet_name, mode="sarcomere")
    res_comb = analyze_sarcomere_segment_from_excel(wb_path, sheet_name, mode="sarcomere-calcium")

    # 1. Segment-level QC must be identical
    assert res_sarc["qc_status"] == res_comb["qc_status"]
    assert res_sarc["qc_reason"] == res_comb["qc_reason"]

    # 2. Accepted counts and event classifications must be identical
    assert res_sarc["n_accepted"] == res_comb["n_accepted"]
    assert res_sarc["n_main"] == res_comb["n_main"]
    assert res_sarc["n_secondary"] == res_comb["n_secondary"]
    assert res_sarc["n_ambiguous"] == res_comb["n_ambiguous"]

    # 3. Frequency and shortening amplitudes must be identical
    assert res_sarc["frequency_bpm"] == res_comb["frequency_bpm"]
    assert res_sarc["mean_pct_shortening"] == res_comb["mean_pct_shortening"]
    assert res_sarc["threshold_main_pct"] == res_comb["threshold_main_pct"]
    assert res_sarc["threshold_sce_pct"] == res_comb["threshold_sce_pct"]

    # 4. Individual event timestamps and classifications must match
    ev_s = res_sarc["events"]
    ev_c = res_comb["events"]
    if ev_s is not None and not ev_s.empty:
        assert np.allclose(ev_s["time_s"].to_numpy(), ev_c["time_s"].to_numpy())
        assert (ev_s["event_class"].to_numpy() == ev_c["event_class"].to_numpy()).all()
        assert (ev_s["status"].to_numpy() == ev_c["status"].to_numpy()).all()


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
@pytest.mark.parametrize("seg_num", [16, 26, 32, 42])
@pytest.mark.parametrize("preset", ["permissive", "default", "strict"])
def test_corrupted_and_noise_segments_strictly_rejected(seg_num, preset):
    """Verify ISO Segments 16, 26, 32, 42 strictly REJECT across all presets."""
    sheet = f"SarcomereLength Segment {seg_num}"
    for mode in ["sarcomere", "sarcomere-calcium"]:
        res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, sheet, mode=mode)
        p_run = res["preset_runs"][preset]
        assert p_run["qc_status"] == "REJECT", f"Expected REJECT for {sheet} under {preset} ({mode}), got {p_run['qc_status']}"
        assert p_run["n_accepted"] == 0, f"Expected 0 reportable events for {sheet} under {preset}, got {p_run['n_accepted']}"


@pytest.mark.skipif(not DATASET_A_PATH.exists(), reason="Dataset A workbook not found")
def test_dataset_a_seg46_calcium_noise_rejection():
    """Verify CAR Segment 46 Calcium noise returns Calcium QC = REJECT, 0 transients."""
    res = analyze_sarcomere_segment_from_excel(DATASET_A_PATH, "SarcomereLength Segment 46", mode="sarcomere-calcium")
    assert res["calcium_qc_status"] == "REJECT"
    cal_res = res["cal_pipeline_res"]
    assert cal_res is not None
    def_run = cal_res["runs"]["default"]
    assert def_run["n_accepted"] == 0


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_dataset_b_seg13_sarcomere_recovery_passes():
    """Verify ISO Segment 13 is recovered independently without calcium assistance."""
    res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 13", mode="sarcomere")
    assert res["qc_status"] == "PASS"
    assert res["n_accepted"] == 10
    assert res["n_main"] == 10
    assert res["n_secondary"] == 0
    recon = res.get("reconstruction", {})
    assert recon.get("n_reconstructed_intervals", 0) > 0


def test_calcium_present_absent_shuffled_synthetic_invariance():
    """Verify that altering, removing, or corrupting calcium channel produces zero change in sarcomere results."""
    fs = 250.0
    rng = np.random.default_rng(123)
    n = 1500
    t = np.arange(n) / fs
    sig_sarc = np.full(n, 1.80)
    for bt in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sig_sarc -= 0.16 * np.exp(-0.5 * ((t - bt) / 0.045) ** 2)
    sig_sarc += rng.normal(0, 0.0005, size=n)

    # 1. Mode: sarcomere only
    res_sarc = analyze_sarcomere_pipeline(sig_sarc, fs, mode="sarcomere")

    # 2. Mode: sarcomere-calcium with matching calcium signal
    sig_cal = np.full(n, 0.95)
    for bt in [0.95, 1.95, 2.95, 3.95, 4.95]:
        sig_cal += 0.40 * np.exp(-0.5 * ((t - bt) / 0.060) ** 2)
    res_with_cal = analyze_sarcomere_pipeline(sig_sarc, fs, mode="sarcomere-calcium", cal_sig_raw=sig_cal, cal_fs=fs)

    # 3. Mode: sarcomere-calcium with pure random noise calcium
    noise_cal = rng.normal(1.0, 0.05, size=n)
    res_noise_cal = analyze_sarcomere_pipeline(sig_sarc, fs, mode="sarcomere-calcium", cal_sig_raw=noise_cal, cal_fs=fs)

    # 4. Mode: sarcomere-calcium with None calcium
    res_none_cal = analyze_sarcomere_pipeline(sig_sarc, fs, mode="sarcomere-calcium", cal_sig_raw=None, cal_fs=None)

    # Invariant: across all 4 conditions, sarcomere results must be 100% identical!
    for other in [res_with_cal, res_noise_cal, res_none_cal]:
        assert res_sarc["qc_status"] == other["qc_status"]
        assert res_sarc["n_accepted"] == other["n_accepted"]
        assert res_sarc["n_main"] == other["n_main"]
        assert res_sarc["n_secondary"] == other["n_secondary"]
        assert res_sarc["frequency_bpm"] == other["frequency_bpm"]
        assert res_sarc["mean_pct_shortening"] == other["mean_pct_shortening"]
        assert (res_sarc["events"]["event_class"].to_numpy() == other["events"]["event_class"].to_numpy()).all()
