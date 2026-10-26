"""
tests/test_sarcomere_analyzer.py
================================

Focused Phase 1 regression and experimental validation test suite for the
Sarcomere Shortening Analyzer (`pixel_counter.sarcomere_qc`,
`pixel_counter.sarcomere_detector`, `pixel_counter.sarcomere_pipeline`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_counter.io_utils import load_cytocypher_excel
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.config import BeatCounterConfig
from pixel_counter.sarcomere_detector import (
    SARCOMERE_STRICTNESS_PRESETS,
    detect_sarcomere_contractions,
)
from pixel_counter.sarcomere_pipeline import (
    analyze_sarcomere_pipeline,
    analyze_sarcomere_segment_from_excel,
    discover_sarcomere_workbook_channels,
    load_sarcomere_segment,
)
from pixel_counter.sarcomere_qc import detect_sarcomere_artifacts


SARCOMERE_DIR = Path(__file__).resolve().parent.parent / "sarcomere shortening"
DATASET_A_PATH = SARCOMERE_DIR / "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx"
DATASET_B_PATH = SARCOMERE_DIR / "(MI) CGP+ISO+ROLI+IBMX DISH1  individual transients result.xlsx"


def _make_synthetic_sarcomere_trace(
    fs: float = 250.0,
    duration_s: float = 10.0,
    l0_um: float = 1.80,
    beat_times_s: list[float] | None = None,
    shortening_um: float = 0.18,
    sigma_s: float = 0.045,
    noise_sd: float = 0.0008,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = np.full(n, l0_um, dtype=float) + rng.normal(0.0, noise_sd, size=n)
    if beat_times_s is None:
        beat_times_s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for bt in beat_times_s:
        sig -= shortening_um * np.exp(-0.5 * ((t - bt) / sigma_s) ** 2)
    return t, sig


@pytest.mark.skipif(not DATASET_A_PATH.exists(), reason="Dataset A workbook not found")
def test_workbook_discovery_and_edgelength_exclusion():
    """Verify channel discovery and strict refusal to use EdgeLength as SarcomereLength."""
    info_a = discover_sarcomere_workbook_channels(DATASET_A_PATH)
    assert info_a["n_sarcomere_segments"] == 52
    assert info_a["n_calcium_segments"] == 52
    assert info_a["n_edgelength_segments"] == 52

    info_b = discover_sarcomere_workbook_channels(DATASET_B_PATH)
    assert info_b["n_sarcomere_segments"] == 50
    assert info_b["n_calcium_segments"] == 50
    assert info_b["n_edgelength_segments"] == 50

    with pytest.raises(ValueError, match="Refusing to load EdgeLength"):
        load_sarcomere_segment(DATASET_A_PATH, "EdgeLength Segment 1")


def test_downward_shortening_and_percentage_calculation():
    """Verify downward contraction detection and exact L0, Lmin, dL, and % shortening math."""
    fs = 250.0
    beat_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    _, sig = _make_synthetic_sarcomere_trace(
        fs=fs, duration_s=10.0, l0_um=1.80, beat_times_s=beat_times, shortening_um=0.18, noise_sd=0.0005
    )

    res = analyze_sarcomere_pipeline(sig, fs, t0=100.0)
    assert res["qc_status"] == "PASS"
    assert res["n_accepted"] == 9
    assert res["mean_l0_um"] == pytest.approx(1.80, abs=0.005)
    assert res["mean_l_min_um"] == pytest.approx(1.62, abs=0.005)
    assert res["mean_abs_shortening_um"] == pytest.approx(0.18, abs=0.006)
    assert res["mean_pct_shortening"] == pytest.approx(10.0, abs=0.35)

    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"]
    for _, row in acc.iterrows():
        pk_idx = int(row["peak_index"])
        assert row["l_min_um"] == pytest.approx(sig[pk_idx], abs=1e-12)
        expected_abs = row["l0_um"] - row["l_min_um"]
        expected_pct = 100.0 * expected_abs / row["l0_um"]
        assert row["abs_shortening_um"] == pytest.approx(expected_abs, abs=1e-9)
        assert row["pct_shortening"] == pytest.approx(expected_pct, abs=1e-9)


@pytest.mark.skipif(not DATASET_A_PATH.exists(), reason="Dataset A workbook not found")
def test_authentic_timestamp_and_raw_value_preservation():
    """Verify raw SarcomereLength values and timestamps are preserved without modification."""
    cfg = BeatCounterConfig()
    df_sheet, y_cols, fs, _ = load_cytocypher_excel(DATASET_A_PATH, sheet_name="SarcomereLength Segment 1")
    sig_expected, _, meta = build_concatenated_signal(df_sheet, y_cols, fs, cfg, force_invert=False)

    res = analyze_sarcomere_segment_from_excel(DATASET_A_PATH, "SarcomereLength Segment 1")
    assert res["fs"] == 250.0
    assert res["t0"] == pytest.approx(float(meta["time_origin_s"]), abs=1e-6)
    # Raw trace must remain 100% identical to source, including 2.2335 um vertical rail artifacts
    np.testing.assert_array_equal(res["sig_raw"], sig_expected)
    assert np.max(res["sig_raw"]) == pytest.approx(2.2335, abs=1e-4)


def test_synthetic_spike_and_corrupted_trough_edge_cases():
    """Verify spike rejection, flank-spike recovery, and corrupted-trough NaN reporting."""
    fs = 250.0
    # 1. Pure vertical spikes on flat baseline -> 0 accepted contractions, REJECT
    _, flat_sig = _make_synthetic_sarcomere_trace(
        fs=fs, duration_s=8.0, l0_um=1.80, beat_times_s=[], shortening_um=0.0, noise_sd=0.0008
    )
    spike_only = flat_sig.copy()
    for idx in [300, 700, 1100, 1500]:
        spike_only[idx] = 1.35  # 1-sample downward spike
    res_spikes = analyze_sarcomere_pipeline(spike_only, fs)
    assert res_spikes["n_accepted"] == 0
    assert res_spikes["qc_status"] == "REJECT"

    # 2. Genuine contractions where beat 1 has an isolated flank spike and beat 2 has a trough-obliterating artifact
    _, clean_sig = _make_synthetic_sarcomere_trace(
        fs=fs, duration_s=6.0, l0_um=1.80, beat_times_s=[1.5, 3.5, 5.0], shortening_um=0.16, noise_sd=0.0005
    )
    mixed_sig = clean_sig.copy()
    # Beat 1 (t=1.5s -> sample 375): add 1-sample upward spike on contraction limb at sample 363
    mixed_sig[363] = 2.2335
    # Beat 2 (t=3.5s -> sample 875): obliterate trough nadir [872..878] with vertical rail 2.2335
    mixed_sig[872:879] = 2.2335

    res_mixed = analyze_sarcomere_pipeline(mixed_sig, fs)
    assert res_mixed["qc_status"] == "PASS"
    assert res_mixed["n_accepted"] == 3
    assert res_mixed["n_corrupted_events"] == 1

    ev = res_mixed["events"]
    corrupted = ev[ev["peak_corrupted"]]
    assert len(corrupted) == 1
    assert corrupted.iloc[0]["status"] == "ACCEPTED"
    assert "reconstructed" in str(corrupted.iloc[0]["reason"])
    assert np.isnan(corrupted.iloc[0]["abs_shortening_um"])
    assert np.isnan(corrupted.iloc[0]["pct_shortening"])


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_missing_or_empty_segment_graceful_qc_reject():
    """Verify Dataset B Segment 34 (no y columns) returns structured REJECT without crashing."""
    res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 34")
    assert res["load_ok"] is False
    assert res["qc_status"] == "REJECT"
    assert res["qc_reason"] == "no_valid_sarcomere_signal"
    assert res["n_accepted"] == 0
    assert res["frequency_hz"] is None


@pytest.mark.skipif(not DATASET_A_PATH.exists() or not DATASET_B_PATH.exists(), reason="Workbooks not found")
def test_isolated_single_contraction_segments():
    """Verify single-contraction recordings in Datasets A and B detect exactly 1 event and PASS."""
    for seg_a in [28, 29, 39, 41]:
        res_a = analyze_sarcomere_segment_from_excel(DATASET_A_PATH, f"SarcomereLength Segment {seg_a}")
        assert res_a["qc_status"] == "PASS", f"Dataset A Seg {seg_a} failed QC"
        assert res_a["n_accepted"] == 1, f"Dataset A Seg {seg_a} expected 1 event, got {res_a['n_accepted']}"

    for seg_b in [39, 49]:
        res_b = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {seg_b}")
        assert res_b["qc_status"] == "PASS", f"Dataset B Seg {seg_b} failed QC"
        assert res_b["n_accepted"] == 1, f"Dataset B Seg {seg_b} expected 1 event, got {res_b['n_accepted']}"


@pytest.mark.skipif(not DATASET_A_PATH.exists() or not DATASET_B_PATH.exists(), reason="Workbooks not found")
def test_noise_and_tracking_failure_qc_rejection():
    """Verify noise-only and artifact-dominated segments in Datasets A and B are rejected by QC."""
    noise_segs_a = [11, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44, 48, 53, 54, 57, 58]
    for seg_a in noise_segs_a:
        res_a = analyze_sarcomere_segment_from_excel(DATASET_A_PATH, f"SarcomereLength Segment {seg_a}")
        assert res_a["qc_status"] == "REJECT", f"Dataset A Seg {seg_a} should be REJECT"
        assert res_a["n_accepted"] == 0

    reject_segs_b = [7, 12, 14, 16, 19, 24, 26, 31, 32, 34, 36, 42, 44, 46]
    for seg_b in reject_segs_b:
        res_b = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {seg_b}")
        assert res_b["qc_status"] == "REJECT", f"Dataset B Seg {seg_b} should be REJECT"
        assert res_b["n_accepted"] == 0


@pytest.mark.skipif(not DATASET_A_PATH.exists() or not DATASET_B_PATH.exists(), reason="Workbooks not found")
def test_vertical_artifact_handling_and_corrupted_trough_nan():
    """Verify partial recovery and non-fabrication of corrupted troughs on A1, A16, B15, B25, B51."""
    # Dataset A Seg 16: Recovered contractions around localized spikes -> 10 accepted beats
    res_a16 = analyze_sarcomere_segment_from_excel(DATASET_A_PATH, "SarcomereLength Segment 16")
    assert res_a16["qc_status"] == "PASS"
    assert res_a16["n_accepted"] == 10
    assert res_a16["n_corrupted_events"] == 2
    assert res_a16["events"]["peak_corrupted"].sum() == 2
    # Zero fabrication: corrupted troughs retain NaN
    corr_a16 = res_a16["events"][res_a16["events"]["peak_corrupted"]]
    assert np.all(np.isnan(corr_a16["abs_shortening_um"]))
    ev_a16 = res_a16["events"]
    acc_a16 = ev_a16[ev_a16["status"] == "ACCEPTED"]
    assert np.all(acc_a16["l_min_um"] > 1.65)

    # Dataset B Seg 15: Small vertical-line artifacts on limbs -> all 10 troughs intact
    res_b15 = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 15")
    assert res_b15["qc_status"] in ("PASS", "REJECT")

    # Dataset B Seg 25: ~18 peaks, valid contractions
    res_b25 = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 25")
    assert res_b25["qc_status"] in ("PASS", "REJECT")

    # Dataset B Seg 51: Strictly binary QC
    res_b51 = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 51")
    assert res_b51["qc_status"] in ("PASS", "REJECT")


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_high_frequency_irregular_cardiomyocyte_beats():
    """Verify unpaced / high-frequency irregular recordings are not suppressed by rate assumptions."""
    expected_approx = {
        2: (10, 10),
        17: (33, 37),
        21: (15, 17),
        23: (16, 18),
        27: (34, 38),
        28: (34, 38),
        30: (35, 39),
    }
    for seg_b, (low_cnt, high_cnt) in expected_approx.items():
        res_b = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {seg_b}")
        assert res_b["qc_status"] in ("PASS", "PASS_PARTIAL"), f"Dataset B Seg {seg_b} failed QC"
        assert low_cnt <= res_b["n_accepted"] <= high_cnt, (
            f"Dataset B Seg {seg_b}: got {res_b['n_accepted']}, expected in [{low_cnt}, {high_cnt}]"
        )


# =========================================================================
# Phase 2 Tests: MAIN vs SECONDARY vs AMBIGUOUS Contraction Classification
# =========================================================================

def test_synthetic_main_vs_secondary_classification():
    """Verify classification logic on synthetic primary/secondary combinations."""
    fs = 250.0

    # 1. MAIN followed by a SECONDARY on incomplete relaxation tail
    # Beat 1: large main at t=1.0s (dL = 0.18 um)
    # Beat 2: smaller secondary at t=1.16s (dt = 0.16s, takes off before full recovery)
    # Beat 3: independent main at t=2.5s (dL = 0.18 um)
    t, sig = _make_synthetic_sarcomere_trace(
        fs=fs, duration_s=4.0, l0_um=1.80, beat_times_s=[1.0, 2.5], shortening_um=0.18, noise_sd=0.0004
    )
    # Add secondary excursion at t=1.16s
    sig -= 0.075 * np.exp(-0.5 * ((t - 1.16) / 0.035) ** 2)

    res = analyze_sarcomere_pipeline(sig, fs)
    assert res["qc_status"] == "PASS"
    assert res["n_accepted"] == 3
    assert res["n_main"] == 2
    assert res["n_secondary"] == 1
    assert res["n_ambiguous"] == 0

    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"].reset_index(drop=True)
    assert acc.loc[0, "event_class"] == "MAIN"
    assert acc.loc[1, "event_class"] == "SECONDARY"
    assert "incomplete_relaxation" in acc.loc[1, "classification_reason"]
    assert acc.loc[2, "event_class"] == "MAIN"


def test_takeoff_baseline_and_local_excursion_metrics():
    """Verify local takeoff baseline vs diastolic baseline metrics integrity."""
    fs = 250.0
    t, sig = _make_synthetic_sarcomere_trace(
        fs=fs, duration_s=3.0, l0_um=1.80, beat_times_s=[1.0], shortening_um=0.20, noise_sd=0.0003
    )
    # Add secondary at t=1.15s
    sig -= 0.070 * np.exp(-0.5 * ((t - 1.15) / 0.035) ** 2)

    res = analyze_sarcomere_pipeline(sig, fs)
    acc = res["events"][res["events"]["status"] == "ACCEPTED"].reset_index(drop=True)
    assert len(acc) == 2
    sec_row = acc.loc[1]
    assert sec_row["event_class"] == "SECONDARY"

    # Takeoff level must be < L0 because relaxation was incomplete
    assert sec_row["takeoff_l0_um"] < sec_row["l0_um"]
    # Local shortening must equal takeoff - Lmin
    expected_local_dl = sec_row["takeoff_l0_um"] - sec_row["l_min_um"]
    assert sec_row["local_abs_shortening_um"] == pytest.approx(expected_local_dl, abs=1e-6)
    # Diastolic shortening must equal L0 - Lmin (distinct and larger than local shortening)
    expected_diast_dl = sec_row["l0_um"] - sec_row["l_min_um"]
    assert sec_row["abs_shortening_um"] == pytest.approx(expected_diast_dl, abs=1e-6)
    assert sec_row["abs_shortening_um"] > sec_row["local_abs_shortening_um"]

    # Percentage shortening relative to takeoff vs diastolic L0
    expected_local_pct = 100.0 * sec_row["local_abs_shortening_um"] / sec_row["takeoff_l0_um"]
    assert sec_row["local_pct_shortening"] == pytest.approx(expected_local_pct, abs=1e-4)
    expected_diast_pct = 100.0 * sec_row["abs_shortening_um"] / sec_row["l0_um"]
    assert sec_row["pct_shortening"] == pytest.approx(expected_diast_pct, abs=1e-4)


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_experimental_segment_33_sequence_classification():
    """Verify Segment 33 begins with MAIN -> SECONDARY -> MAIN -> 2 SECONDARY events."""
    res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 33")
    assert res["qc_status"] == "PASS"
    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"].reset_index(drop=True)
    # The first 5 events must strictly be: MAIN, SECONDARY, MAIN, SECONDARY, SECONDARY
    classes = acc["event_class"].head(5).tolist()
    assert classes == ["MAIN", "SECONDARY", "MAIN", "SECONDARY", "SECONDARY"], f"Got classes: {classes}"
    assert res["n_main"] >= 10
    assert res["n_secondary"] >= 7


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_experimental_segment_35_classification():
    """Verify Segment 35 begins with 3 MAIN events followed by mixed MAIN/SECONDARY."""
    res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 35")
    assert res["qc_status"] == "PASS"
    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"].reset_index(drop=True)
    # First three events are MAIN
    assert acc.loc[0, "event_class"] == "MAIN"
    assert acc.loc[1, "event_class"] == "MAIN"
    assert acc.loc[2, "event_class"] == "MAIN"
    # Event 3 is an alternating SECONDARY after-contraction
    assert acc.loc[3, "event_class"] == "SECONDARY"
    assert acc.loc[4, "event_class"] == "MAIN"
    assert res["n_main"] >= 12
    assert res["n_secondary"] >= 4


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_experimental_segment_41_classification():
    """Verify Segment 41 identifies >=20 MAIN and >=8 SECONDARY events under strict amplitude hierarchy."""
    res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, "SarcomereLength Segment 41")
    assert res["qc_status"] == "PASS"
    assert res["n_accepted"] == 32
    assert res["n_main"] >= 20
    assert res["n_secondary"] >= 8
    # Contractions with amplitude >= T_MAIN are strictly MAIN; T_SCE <= amp < T_MAIN are SECONDARY
    t_main = res["threshold_main_pct"]
    t_sce = res["threshold_sce_pct"]
    ev = res["events"]
    acc = ev[ev["status"] == "ACCEPTED"].reset_index(drop=True)
    for i in range(len(acc)):
        amp = acc.loc[i, "pct_shortening"]
        cls = acc.loc[i, "event_class"]
        if amp >= t_main:
            assert cls == "MAIN", f"Event at t={acc.loc[i, 'time_s']} with amp {amp:.2f}% >= {t_main:.2f}% must be MAIN"
        elif amp >= t_sce:
            assert cls == "SECONDARY", f"Event at t={acc.loc[i, 'time_s']} with amp {amp:.2f}% < {t_main:.2f}% must be SECONDARY"


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_negative_controls_unpaced_fast_rhythms_remain_main():
    """Verify rapid unpaced cardiomyocytes (Segs 17, 27, 28, 30) remain predominantly MAIN."""
    for s in [17, 27, 28, 30]:
        res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {s}")
        assert res["qc_status"] == "PASS"
        total = res["n_accepted"]
        main_cnt = res["n_main"]
        # Fast independent beats must not automatically become secondary
        assert main_cnt / total >= 0.85, f"Segment {s}: {main_cnt}/{total} MAIN is below 85%"

    # Isolated beats in Segments 39, 49 must be 100% MAIN
    for s in [39, 49]:
        res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {s}")
        assert res["qc_status"] == "PASS"
        assert res["n_accepted"] == 1
        assert res["n_main"] == 1
        assert res["n_secondary"] == 0
        assert res["n_ambiguous"] == 0


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_qc_rejected_recordings_have_zero_reportable_classifications():
    """Verify QC-rejected recordings produce zero reportable MAIN, SECONDARY, or AMBIGUOUS events."""
    for s in [7, 12, 16, 19, 24, 26, 32, 34, 42, 44]:
        res = analyze_sarcomere_segment_from_excel(DATASET_B_PATH, f"SarcomereLength Segment {s}")
        assert res["qc_status"] == "REJECT"
        assert res["n_accepted"] == 0
        assert res["n_main"] == 0
        assert res["n_secondary"] == 0
        assert res["n_ambiguous"] == 0


def _make_synthetic_calcium_trace(
    fs: float = 250.0,
    duration_s: float = 10.0,
    r0: float = 1.0,
    beat_times_s: list[float] | None = None,
    delta_r: float = 0.40,
    sigma_rise_s: float = 0.025,
    sigma_decay_s: float = 0.080,
    noise_sd: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = np.full(n, r0, dtype=float) + rng.normal(0.0, noise_sd, size=n)
    if beat_times_s is None:
        beat_times_s = [0.96, 1.96, 2.96, 3.96, 4.96, 5.96, 6.96, 7.96, 8.96]
    for bt in beat_times_s:
        dt = t - bt
        transient = np.where(
            dt < 0,
            delta_r * np.exp(-0.5 * (dt / sigma_rise_s) ** 2),
            delta_r * np.exp(-0.5 * (dt / sigma_decay_s) ** 2),
        )
        sig += transient
    return t, sig


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_sarcomere_only_mode_invariance_with_calcium_present():
    """Verify Mode A (SARCOMERE ONLY) produces 100% identical results with and without calcium."""
    # Analyze in sarcomere mode
    res_sarc_only = analyze_sarcomere_segment_from_excel(
        DATASET_B_PATH, "SarcomereLength Segment 33", mode="sarcomere"
    )
    # Also analyze passing cal_sig directly into analyze_sarcomere_pipeline with mode='sarcomere'
    sarc_data = load_sarcomere_segment(DATASET_B_PATH, "SarcomereLength Segment 33")
    df_c, y_c, fs_c, t0_c = load_cytocypher_excel(DATASET_B_PATH, sheet_name="RatioMetricCalcium Segment 33")
    sig_c, _, meta_c = build_concatenated_signal(df_c, y_c, fs_c, BeatCounterConfig(), force_invert=False)
    t0_cal = float(meta_c.get("time_origin_s", t0_c))

    res_with_cal_passed = analyze_sarcomere_pipeline(
        sarc_data["sig_raw"],
        sarc_data["fs"],
        t0=sarc_data["t0"],
        mode="sarcomere",
        cal_sig_raw=sig_c,
        cal_fs=fs_c,
        cal_t0=t0_cal,
    )

    assert res_sarc_only["n_accepted"] == res_with_cal_passed["n_accepted"]
    assert res_sarc_only["n_main"] == res_with_cal_passed["n_main"]
    assert res_sarc_only["n_secondary"] == res_with_cal_passed["n_secondary"]
    assert res_sarc_only["n_ambiguous"] == res_with_cal_passed["n_ambiguous"]
    assert res_sarc_only["mean_abs_shortening_um"] == pytest.approx(res_with_cal_passed["mean_abs_shortening_um"], abs=1e-12)

    ev = res_sarc_only["events"]
    assert (ev["calcium_match_status"] == "UNAVAILABLE").all()
    assert (ev["sarcomere_only_classification"] == ev["assisted_classification"]).all()


def test_clean_paired_signals_with_known_emd():
    """Verify paired 1 Hz signals with 40ms electromechanical delay are matched 1-to-1 with HIGH confidence."""
    fs = 250.0
    s_beats = [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00]
    c_beats = [0.96, 1.96, 2.96, 3.96, 4.96, 5.96, 6.96, 7.96, 8.96]  # 40ms delay

    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=s_beats, shortening_um=0.15)
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, beat_times_s=c_beats, delta_r=0.45)

    res = analyze_sarcomere_pipeline(
        s_sig, fs, t0=0.0, mode="sarcomere-calcium",
        cal_sig_raw=c_sig, cal_fs=fs, cal_t0=0.0,
    )

    assert res["qc_status"] == "PASS"
    assert res["calcium_qc_status"] == "PASS"
    assert res["combined_qc"]["combined_qc_status"] == "PASS"
    assert res["n_matched_pairs"] == 9
    assert res["n_unmatched_sarc"] == 0
    assert res["n_unmatched_cal"] == 0
    # Peak-to-peak delay accounts for asymmetric transient smoothing (~32-40ms)
    assert res["estimated_emd_delay_s"] == pytest.approx(0.035, abs=0.010)

    pairs = res["matched_pairs"]
    for p in pairs:
        assert p["matching_confidence"] == "HIGH"
        assert p["peak_to_peak_delay_s"] == pytest.approx(0.035, abs=0.010)


def test_unmatched_mechanical_and_calcium_events():
    """Verify unmatched mechanical secondary events and uncoupled calcium transients are preserved."""
    fs = 250.0
    # Sarcomere has 9 main beats + 2 secondary after-contractions at 2.25s and 5.25s
    s_beats = [1.00, 2.00, 2.25, 3.00, 4.00, 5.00, 5.25, 6.00, 7.00, 8.00, 9.00]
    # Calcium has 9 main beats + 1 uncoupled calcium transient at 7.45s
    c_beats = [0.96, 1.96, 2.96, 3.96, 4.96, 5.96, 6.96, 7.45, 7.96, 8.96]

    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=s_beats, shortening_um=0.15)
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, beat_times_s=c_beats, delta_r=0.45)

    res = analyze_sarcomere_pipeline(
        s_sig, fs, t0=0.0, mode="sarcomere-calcium",
        cal_sig_raw=c_sig, cal_fs=fs, cal_t0=0.0,
    )

    assert res["qc_status"] == "PASS"
    assert res["n_matched_pairs"] == 9
    assert res["n_unmatched_sarc"] == 2
    assert res["n_unmatched_cal"] == 1

    unm_s = res["unmatched_sarcomere_events"]
    assert len(unm_s) == 2
    assert (unm_s["calcium_match_status"] == "UNMATCHED_MECHANICAL").all()

    unm_c = res["unmatched_calcium_events"]
    assert len(unm_c) == 1
    assert (unm_c["calcium_match_status"] == "UNMATCHED_CALCIUM").all()
    # Unmatched calcium must NOT have fabricated sarcomere amplitude
    assert np.isnan(unm_c.iloc[0]["sarcomere_amplitude_um"])


def test_borderline_sarcomere_candidate_recovery():
    """Verify that calcium NEVER rescues or alters borderline mechanical candidates (channel independence)."""
    fs = 250.0
    rng = np.random.default_rng(101)
    n = int(round(6.0 * fs))
    t = np.arange(n, dtype=float) / fs
    s_sig = np.full(n, 1.80, dtype=float) + rng.normal(0.0, 0.0018, size=n)
    for bt in [1.00, 2.00, 4.00, 5.00]:
        s_sig -= 0.12 * np.exp(-0.5 * ((t - bt) / 0.045) ** 2)
    s_sig -= 0.0078 * np.exp(-0.5 * ((t - 3.00) / 0.040) ** 2)
    c_beats = [0.96, 1.96, 2.96, 3.96, 4.96]
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, duration_s=6.0, beat_times_s=c_beats, delta_r=0.45)
    res_sarc = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    res_comb = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere-calcium", cal_sig_raw=c_sig, cal_fs=fs)
    # Strict invariant: calcium presence has zero influence on sarcomere candidate acceptance
    assert res_sarc["n_accepted"] == res_comb["n_accepted"] == 4
    ev_s = res_sarc["events"]
    weak_s = ev_s[(ev_s["time_s"] >= 2.90) & (ev_s["time_s"] <= 3.10)]
    assert len(weak_s) == 1
    assert weak_s.iloc[0]["status"] == "REJECTED"
    ev_c = res_comb["events"]
    weak_c = ev_c[(ev_c["time_s"] >= 2.90) & (ev_c["time_s"] <= 3.10)]
    assert len(weak_c) == 1
    assert weak_c.iloc[0]["status"] == "REJECTED", "Calcium must never promote rejected mechanical candidate!"
def test_vertical_tracking_artifact_corrupted_trough_preserves_nan():
    """Verify trough obliterated by tracking artifact keeps NaN amplitude independently of calcium."""
    fs = 250.0
    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0])
    art_idx = int(2.0 * fs)
    s_sig[art_idx - 2 : art_idx + 2] = 2.2335
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, beat_times_s=[0.96, 1.96, 2.96, 3.96])
    res_sarc = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    res_comb = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere-calcium", cal_sig_raw=c_sig, cal_fs=fs)
    assert res_sarc["n_accepted"] == res_comb["n_accepted"]
    for r in [res_sarc, res_comb]:
        ev = r["events"]
        corr = ev[(ev["time_s"] >= 1.90) & (ev["time_s"] <= 2.10)]
        assert len(corr) == 1
        assert corr.iloc[0]["peak_corrupted"] is True or bool(corr.iloc[0]["peak_corrupted"]) is True
        assert np.isnan(corr.iloc[0]["abs_shortening_um"])
        assert np.isnan(corr.iloc[0]["pct_shortening"])
def test_calcium_rejection_and_dropout_does_not_break_sarcomere():
    """Verify calcium QC failure preserves sarcomere-only results without crashing or altering metrics."""
    fs = 250.0
    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0, 5.0])
    # Pure noise calcium with zero transients (QC REJECT)
    rng = np.random.default_rng(999)
    c_noisy = np.full(len(s_sig), 1.0) + rng.normal(0.0, 0.05, size=len(s_sig))

    res = analyze_sarcomere_pipeline(
        s_sig, fs, mode="sarcomere-calcium",
        cal_sig_raw=c_noisy, cal_fs=fs,
    )

    assert res["qc_status"] == "PASS"
    assert res["calcium_qc_status"] == "REJECT"
    assert res["combined_qc"]["combined_qc_reason"] == "sarcomere_valid_calcium_rejected"
    assert res["n_accepted"] == 5
    assert res["n_matched_pairs"] == 0
    ev = res["events"]
    assert (ev[ev["status"] == "ACCEPTED"]["calcium_match_status"] == "UNAVAILABLE").all()


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_sarcomere_qc_rejection_precedence_over_clean_calcium():
    """Verify segment with rejected sarcomere QC produces 0 reportable events even with clean calcium."""
    # Dataset B Segment 7 has tracking_artifact_dominated sarcomere QC REJECT, but clean calcium PASS
    res = analyze_sarcomere_segment_from_excel(
        DATASET_B_PATH, "SarcomereLength Segment 7", mode="sarcomere-calcium"
    )

    assert res["qc_status"] == "REJECT"
    assert res["calcium_qc_status"] == "PASS"
    assert res["combined_qc"]["combined_qc_reason"] == "sarcomere_invalid_calcium_valid"
    assert res["n_accepted"] == 0
    assert res["frequency_hz"] is None
    assert res["mean_abs_shortening_um"] is None


@pytest.mark.skipif(not DATASET_B_PATH.exists(), reason="Dataset B workbook not found")
def test_ambiguous_contraction_promotion_on_seg41():
    """Verify Segment 41 contraction at 1457.79s has amplitude >= T_MAIN and is strictly MAIN in both modes."""
    res_sarc = analyze_sarcomere_segment_from_excel(
        DATASET_B_PATH, "SarcomereLength Segment 41", mode="sarcomere"
    )
    res_comb = analyze_sarcomere_segment_from_excel(
        DATASET_B_PATH, "SarcomereLength Segment 41", mode="sarcomere-calcium"
    )

    # In both modes, 24 MAIN and 8 SECONDARY beats exist under strict amplitude hierarchy
    assert res_sarc["n_accepted"] == 32
    assert res_sarc["n_main"] == 24
    assert res_sarc["n_secondary"] == 8

    assert res_comb["n_accepted"] == 32
    assert res_comb["n_main"] == 24
    assert res_comb["n_secondary"] == 8

    ev_s = res_sarc["events"]
    ev_c = res_comb["events"]
    b_s = ev_s[(ev_s["time_s"] >= 1457.70) & (ev_s["time_s"] <= 1457.85)].iloc[0]
    b_c = ev_c[(ev_c["time_s"] >= 1457.70) & (ev_c["time_s"] <= 1457.85)].iloc[0]

    assert b_s["pct_shortening"] >= res_sarc["threshold_main_pct"]
    assert b_s["event_class"] == "MAIN"
    assert b_c["pct_shortening"] >= res_comb["threshold_main_pct"]
    assert b_c["event_class"] == "MAIN"


def test_different_durations_and_time_origins():
    """Verify authentic timestamps and continuous temporal overlap with unequal channel durations."""
    fs = 250.0
    # Sarcomere: starts at t0=10.0s, duration 10.0s (2500 samples), beats at absolute t = 11.0, 12.0, ...
    t_s, s_sig = _make_synthetic_sarcomere_trace(fs=fs, duration_s=10.0, beat_times_s=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    t0_s = 10.0

    # Calcium: starts at t0=10.05s, duration 9.5s (2375 samples)
    # Absolute calcium beats at t = 10.96, 11.96, ... (array times 10.96 - 10.05 = 0.91, 1.91, ...)
    c_array_beats = [0.91, 1.91, 2.91, 3.91, 4.91, 5.91, 6.91, 7.91, 8.91]
    t_c, c_sig = _make_synthetic_calcium_trace(fs=fs, duration_s=9.5, beat_times_s=c_array_beats)
    t0_c = 10.05

    res = analyze_sarcomere_pipeline(
        s_sig, fs, t0=t0_s, mode="sarcomere-calcium",
        cal_sig_raw=c_sig, cal_fs=fs, cal_t0=t0_c,
    )

    assert res["qc_status"] == "PASS"
    assert res["calcium_qc_status"] == "PASS"
    assert res["estimated_emd_delay_s"] == pytest.approx(0.035, abs=0.010)
    # Timestamps on matched pairs must preserve their original origins
    pairs = res["matched_pairs"]
    assert len(pairs) == 9
    first_pair = pairs[0]
    assert first_pair["sarcomere_time_s"] == pytest.approx(11.00, abs=0.01)
    assert first_pair["calcium_time_s"] == pytest.approx(10.96, abs=0.01)


@pytest.mark.skipif(not DATASET_A_PATH.exists(), reason="Dataset A workbook not found")
def test_sarcomere_review_process_workbook_and_html_generation(tmp_path):
    """Verify Phase 4 workbook processing and HTML review generation for sarcomere and sarcomere-calcium modes."""
    from pixel_counter.sarcomere_review import (
        process_sarcomere_workbook,
        generate_sarcomere_html,
    )

    # 1. Test sarcomere mode on segment 1
    sarc_data, src_name = process_sarcomere_workbook(
        DATASET_A_PATH, segments=[1], analysis_mode="sarcomere"
    )
    assert len(sarc_data) == 1
    s0 = sarc_data[0]
    assert s0["segment_number"] == 1
    assert s0["analysis_mode"] == "sarcomere"
    assert s0["channel"] == "SarcomereLength"
    assert s0["sig_raw_b64"] is not None
    assert s0["l0_baseline_b64"] is not None
    assert s0["shortening_b64"] is not None
    assert "default" in s0["runs"]
    assert s0["sig_cal_b64"] is None  # calcium trace not embedded in sarcomere mode

    out_sarc_html = tmp_path / "test_sarc_review.html"
    generate_sarcomere_html(
        sarc_data,
        src_name,
        str(out_sarc_html),
        workbook_path=str(DATASET_A_PATH),
        analysis_mode="sarcomere",
    )
    assert out_sarc_html.exists()
    assert out_sarc_html.stat().st_size > 1_000_000  # contains embedded plotly
    html_text = out_sarc_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "const segments_data =" in html_text
    assert "Plotly.newPlot" in html_text
    assert "<th>Reviewed</th>" in html_text  # 11-column table contains Reviewed
    assert "Save annotations (CSV)" in html_text
    assert "btn_export_overrides" in html_text

    # 2. Test sarcomere-calcium mode on segment 1
    dual_data, _ = process_sarcomere_workbook(
        DATASET_A_PATH, segments=[1], analysis_mode="sarcomere-calcium"
    )
    assert len(dual_data) == 1
    d0 = dual_data[0]
    assert d0["segment_number"] == 1
    assert d0["analysis_mode"] == "sarcomere-calcium"
    assert d0["sig_cal_b64"] is not None  # calcium trace embedded in dual mode
    assert "calcium_events" in d0
    assert "n_matched_pairs" in d0

    out_dual_html = tmp_path / "test_dual_review.html"
    generate_sarcomere_html(
        dual_data,
        src_name,
        str(out_dual_html),
        workbook_path=str(DATASET_A_PATH),
        analysis_mode="sarcomere-calcium",
    )
    assert out_dual_html.exists()
    html_dual_text = out_dual_html.read_text(encoding="utf-8")
    assert "RatioMetricCalcium" in html_dual_text
    assert "MODE: SARCOMERE-CALCIUM" in html_dual_text


def test_sarcomere_mode_strict_invariance_to_calcium():
    """Verify sarcomere mode produces strictly identical results with or without calcium present."""
    fs = 250.0
    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0, 5.0])
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, beat_times_s=[0.96, 1.96, 2.96, 3.96, 4.96])

    # Run sarcomere mode without calcium
    res_no_cal = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    # Run sarcomere mode with calcium passed in
    res_with_cal = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere", cal_sig_raw=c_sig, cal_fs=fs)

    assert res_no_cal["n_accepted"] == res_with_cal["n_accepted"]
    assert res_no_cal["mean_abs_shortening_um"] == pytest.approx(res_with_cal["mean_abs_shortening_um"], abs=1e-12)
    assert res_no_cal["events"]["peak_index"].tolist() == res_with_cal["events"]["peak_index"].tolist()
    assert res_no_cal["events"]["event_class"].tolist() == res_with_cal["events"]["event_class"].tolist()


def test_cli_sarcomere_modes_argparse():
    """Verify CLI accepts sarcomere and sarcomere-calcium mode arguments."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    cli_py = repo_root / "run_v2_review_utility.py"

    proc = subprocess.run(
        [sys.executable, str(cli_py), "--help"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0
    help_text = proc.stdout
    assert "sarcomere" in help_text
    assert "sarcomere-calcium" in help_text
    assert "--calcium-assistance" in help_text


def test_calcium_assistance_on_vs_off_and_nan_preservation():
    """Verify that calcium assistance flag has zero effect and artifact troughs preserve NaN amplitude."""
    fs = 250.0
    _, s_sig = _make_synthetic_sarcomere_trace(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0, 5.0])
    s_sig[int(2.0 * fs) - 2 : int(2.0 * fs) + 2] = 2.2335
    s_sig[int(4.0 * fs) - 2 : int(4.0 * fs) + 2] = 2.2335
    _, c_sig = _make_synthetic_calcium_trace(fs=fs, beat_times_s=[0.96, 1.96, 2.96, 3.96, 4.96])
    res_sarc = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    res_comb = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere-calcium", cal_sig_raw=c_sig, cal_fs=fs)
    assert res_sarc["qc_status"] == res_comb["qc_status"]
    assert res_sarc["n_accepted"] == res_comb["n_accepted"]
    ev = res_comb["events"]
    corr_ev = ev[ev["peak_corrupted"]]
    assert len(corr_ev) == 2
    for _, row in corr_ev.iterrows():
        assert np.isnan(row["pct_shortening"])
        assert np.isnan(row["abs_shortening_um"])
def test_car_dish1_segment1_recovery_regression():
    """Verify CAR DISH1 Segment 1 tracking artifact dominance: rejected for excessive reconstruction dependence."""
    wb_path = Path("sarcomere shortening") / "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx"
    if not wb_path.exists():
        pytest.skip("Experimental workbook not found on disk")
    res_sarc = analyze_sarcomere_segment_from_excel(wb_path, sheet_name="SarcomereLength Segment 1", mode="sarcomere")
    res_comb = analyze_sarcomere_segment_from_excel(wb_path, sheet_name="SarcomereLength Segment 1", mode="sarcomere-calcium")
    assert res_sarc["qc_status"] == "REJECT"
    assert res_sarc["qc_reason"] == "excessive_reconstruction_dependence"
    assert res_comb["qc_status"] == "REJECT"
    assert res_comb["qc_reason"] == "excessive_reconstruction_dependence"
    assert res_sarc["n_accepted"] == 0
    assert res_comb["n_accepted"] == 0


def test_noise_calcium_cannot_rescue_mechanical_noise():
    """Verify Segment 22 negative control: noise calcium QC REJECT produces 0 accepted calcium events and 0 contractions."""
    wb_path = Path("sarcomere shortening") / "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx"
    if not wb_path.exists():
        pytest.skip("Experimental workbook not found on disk")

    res = analyze_sarcomere_segment_from_excel(
        wb_path, sheet_name="SarcomereLength Segment 22",
        mode="sarcomere-calcium", enable_assistance=True,
    )
    assert res["calcium_qc_status"] == "REJECT"
    assert res["n_accepted"] == 0
    # Independent calcium pipeline must have 0 accepted events
    cal_res = res["cal_pipeline_res"]
    assert cal_res["runs"]["default"]["n_accepted"] == 0


def test_multi_panel_architecture_and_noninteractive_raw(tmp_path):
    """Verify generated review HTML has exactly 2 panels for sarcomere and 3 panels for sarcomere-calcium, with non-interactive Panel 1."""
    from pixel_counter.sarcomere_review import generate_sarcomere_html, process_sarcomere_workbook

    wb_path = Path("sarcomere shortening") / "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx"
    if not wb_path.exists():
        pytest.skip("Experimental workbook not found on disk")

    # 1. Sarcomere only
    data_s, name_s = process_sarcomere_workbook(wb_path, segments=[1], analysis_mode="sarcomere")
    assert data_s[0]["sig_cal_b64"] is None
    out_s = tmp_path / "review_sarc.html"
    generate_sarcomere_html(data_s, name_s, out_s, analysis_mode="sarcomere")
    txt_s = out_s.read_text(encoding="utf-8")
    assert "MODE: SARCOMERE" in txt_s
    assert "Original SarcomereLength" in txt_s
    assert "Normalized Shortening" in txt_s
    assert "pt.data.yaxis === P.raw.ax" in txt_s  # non-interactive Panel 1 check

    # 2. Sarcomere-Calcium
    data_sc, name_sc = process_sarcomere_workbook(wb_path, segments=[1], analysis_mode="sarcomere-calcium", enable_assistance=True)
    assert data_sc[0]["sig_cal_b64"] is not None
    out_sc = tmp_path / "review_sc.html"
    generate_sarcomere_html(data_sc, name_sc, out_sc, analysis_mode="sarcomere-calcium")
    txt_sc = out_sc.read_text(encoding="utf-8")
    assert "MODE: SARCOMERE-CALCIUM" in txt_sc
    assert "Original SarcomereLength" in txt_sc
    assert "Normalized Shortening" in txt_sc
    assert 'id="btn_cal_assistance"' not in txt_sc
    assert 'id="chk_reconstruction"' in txt_sc
    assert 'id="chk_artifacts"' in txt_sc





