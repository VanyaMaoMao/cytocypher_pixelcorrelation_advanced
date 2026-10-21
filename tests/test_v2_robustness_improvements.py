"""Stage D regression and unit tests for Cytocypher v2 real-world robustness improvements.

Covers all 12 required scientific and technical scenarios:
  1. Isolated impulse spikes.
  2. Impulse spikes embedded within genuine contractions.
  3. Clean records with no contractions.
  4. Noisy records with no reliably detectable contractions.
  5. Mixed clean/noisy intervals and valid analysed duration frequency denominator.
  6. Baseline drift invariance.
  7. Overlapping identical samples (including banker's rounding parity regression).
  8. Overlapping contradictory samples (permanent NaN masking and thresholded QC).
  9. Correct preservation of unaffected samples.
 10. Legitimate narrow contractions (and notched plateau contractions).
 11. Closely spaced genuine contractions vs shallow-recovery double peaks.
 12. Borderline amplitude events (genuine weak beats vs post-contraction secondary bounce).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixel_counter.config import BeatCounterConfig
from pixel_counter.io_utils import load_cytocypher_excel
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.v2_detector import ACCEPTED, REJECTED, DetectorParams, detect_contractions
from pixel_counter.v2_pipeline import analyze_v2_pipeline
from tests.v2_synth import FS, gauss, make_amplitude, to_raw

CENTERS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _make_stitched_df(rows_data, fs=250.0):
    """Build a synthetic Cytocypher segment DataFrame with Begin/End timestamps and y columns."""
    max_len = max(len(r["y"]) for r in rows_data)
    y_cols = [f"y {i}" for i in range(max_len)]
    records = []
    for idx, r in enumerate(rows_data):
        rec = {
            "Transient": idx + 1,
            "Begin": float(r["begin"]),
            "End": float(r["begin"] + len(r["y"]) / fs),
        }
        for j in range(max_len):
            rec[f"y {j}"] = float(r["y"][j]) if j < len(r["y"]) else np.nan
        records.append(rec)
    return pd.DataFrame(records), y_cols


# ---------------------------------------------------------------------------
# 1. Isolated impulse spikes
# ---------------------------------------------------------------------------
def test_isolated_impulse_spikes_rejected():
    """Isolated 1-3 sample technical impulses (even taller than real contractions) must be
    rejected as isolated_impulse_artefact without inflating reference_amplitude."""
    fs = 250.0
    t = np.arange(int(7.0 * fs)) / fs
    a = np.zeros_like(t)
    for c in CENTERS:
        a += gauss(t, c, 0.065, 0.12)
    rng = np.random.default_rng(11)
    a += rng.normal(0, 0.0015, size=t.size)

    # Inject two isolated narrow impulses (4 samples = 16 ms wide, taller than the 0.12 contractions)
    imp1_idx = int(1.50 * fs)
    a[imp1_idx : imp1_idx + 4] = 0.22
    imp2_idx = int(4.45 * fs)
    a[imp2_idx : imp2_idx + 4] = 0.26

    res = detect_contractions(a, fs)
    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]
    rej_imp = ev[ev["reason"] == "isolated_impulse_artefact"]

    assert len(acc) == len(CENTERS)
    np.testing.assert_allclose(acc["time_rel_s"].to_numpy(), CENTERS, atol=0.03)
    assert len(rej_imp) == 2
    # Reference amplitude must reflect the ~0.12 contractions, not the 0.22-0.26 spikes
    assert res["reference_amplitude"] < 0.15


# ---------------------------------------------------------------------------
# 2. Impulse spikes embedded within genuine contractions
# ---------------------------------------------------------------------------
def test_embedded_impulse_spikes_preserve_contraction_and_localize_on_plateau():
    """When a narrow vertical spike occurs on top of a genuine contraction:
      - The genuine contraction must be preserved as ACCEPTED (not rejected).
      - The spike must not be counted as a second contraction.
      - The event timestamp and amplitude must be localized on the underlying contraction plateau.
      - Embedded spikes must not inflate reference_amplitude and cause false rejection of normal beats.
    """
    fs = 250.0
    t = np.arange(int(8.0 * fs)) / fs
    a = np.zeros_like(t)
    # Beats at 1..7 s: five beats have plateau height 0.10, two normal beats (3.0s, 6.0s) have height 0.058
    beat_amps = [0.10, 0.10, 0.058, 0.10, 0.10, 0.058, 0.10]
    beat_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for c, amp_val in zip(beat_times, beat_amps):
        a += gauss(t, c, 0.075, amp_val)
    rng = np.random.default_rng(22)
    a += rng.normal(0, 0.0012, size=t.size)

    # Embed sharp 4-sample (16 ms) vertical impulse spikes (height ~0.24) on the four 0.10 beats at 1, 2, 4, 5 s
    for c in [1.0, 2.0, 4.0, 5.0]:
        sp_idx = int((c + 0.02) * fs)
        a[sp_idx : sp_idx + 4] = 0.24

    raw = 0.85 - a
    res = analyze_v2_pipeline(raw, fs, BeatCounterConfig(), t0=10.0, meta={"time_origin_s": 10.0})
    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]

    # All 7 genuine contractions (including the 0.058 normal beats at 3.0s and 6.0s) must be ACCEPTED
    assert len(acc) == 7
    np.testing.assert_allclose(acc["time_rel_s"].to_numpy(), beat_times, atol=0.06)

    # Every accepted event must have its amplitude localized on the ~0.058-0.11 plateau (never on the 0.24 spike)
    assert np.all(acc["amplitude"].to_numpy() < 0.13)
    # Exact coordinate invariant: amplitude == waveform['amplitude'][index]
    wf_amp = res["waveform"]["amplitude"]
    np.testing.assert_allclose(acc["amplitude"].to_numpy(), wf_amp[acc["index"].to_numpy()])


# ---------------------------------------------------------------------------
# 3. Clean records with no contractions
# ---------------------------------------------------------------------------
def test_clean_silent_record_with_no_contractions():
    """A technically valid, quiet recording with zero contractions must be classified as
    PASS (no_accepted_contractions) and clean_silent_quiescent, not defective."""
    fs = 250.0
    rng = np.random.default_rng(33)
    raw = 0.80 + rng.normal(0, 0.0010, int(6.0 * fs))

    res = analyze_v2_pipeline(
        raw, fs, BeatCounterConfig(), t0=0.0, meta={"time_origin_s": 0.0}, polarity_override=-1
    )
    md = res["metadata"]
    assert md["qc_status"] == "PASS"
    assert md["qc_reason"] == "no_accepted_contractions"
    assert md["qc_layer1_status"] == "PASS"
    assert md["qc_layer2_status"] == "PASS"
    assert md["signal_quality_category"] == "clean_silent_quiescent"
    assert md["n_accepted"] == 0


# ---------------------------------------------------------------------------
# 4. Noisy records with no reliably detectable contractions
# ---------------------------------------------------------------------------
def test_noise_dominated_uninterpretable_record_rejected():
    """A recording dominated by noise/fluctuations with zero reliable contractions must be
    rejected as noise_dominated_uninterpretable rather than reported as a clean zero-beat trace."""
    fs = 250.0
    rng = np.random.default_rng(44)
    # High background noise (sigma ~ 0.006) without coherent contractions
    raw = 0.80 + rng.normal(0, 0.0065, int(8.0 * fs))

    res = analyze_v2_pipeline(
        raw, fs, BeatCounterConfig(), t0=0.0, meta={"time_origin_s": 0.0}, polarity_override=-1
    )
    md = res["metadata"]
    assert md["qc_status"] == "REJECT"
    assert md["qc_reason"] == "noise_dominated_uninterpretable"
    assert md["qc_layer1_status"] == "PASS"
    assert md["qc_layer2_status"] == "REJECT"
    assert md["signal_quality_category"] == "noise_dominated_uninterpretable"


# ---------------------------------------------------------------------------
# 5. Mixed clean/noisy intervals & valid analysed duration frequency denominator
# ---------------------------------------------------------------------------
def test_mixed_clean_and_noisy_intervals_excludes_corrupted_window_from_rate():
    """A recording with clean contractions in [0, 6s] and a corrupted high-noise burst in [6.5, 10s]
    must preserve the valid contractions, identify the excluded interval, and use analyzable_duration_s
    as the denominator for contraction_frequency_hz."""
    fs = 250.0
    t = np.arange(int(10.0 * fs)) / fs
    a = np.zeros_like(t)
    beat_centers = [0.8, 1.8, 2.8, 3.8, 4.8, 5.8]
    for c in beat_centers:
        a += gauss(t, c, 0.06, 0.14)
    rng = np.random.default_rng(55)
    a += rng.normal(0, 0.0012, size=t.size)
    # Corrupt the last 3.2 seconds [6.8s, 10.0s] with high-frequency alternating step noise (small enough not to pass SNR*A_ref)
    burst_start = int(6.8 * fs)
    burst = 0.012 * ((-1.0) ** np.arange(t.size - burst_start)) + rng.normal(0, 0.0055, size=t.size - burst_start)
    raw = 0.85 - a
    raw[burst_start:] -= burst

    res = analyze_v2_pipeline(raw, fs, BeatCounterConfig(), t0=100.0, meta={"time_origin_s": 100.0})
    md = res["metadata"]
    assert md["qc_status"] == "PASS"
    assert md["n_accepted"] == 6
    assert md["signal_quality_category"] == "mixed_clean_and_unusable_intervals"
    assert len(md["excluded_intervals_s"]) >= 1
    assert md["analyzable_duration_s"] < md["valid_duration_s"] - 1.5
    expected_rate = 6.0 / md["analyzable_duration_s"]
    assert md["contraction_frequency_hz"] == pytest.approx(expected_rate, rel=1e-3)


# ---------------------------------------------------------------------------
# 6. Baseline drift
# ---------------------------------------------------------------------------
def test_baseline_drift_preserves_contractions_and_reports_drift_range():
    """Substantial slow baseline drift must not prevent contraction detection or shift timestamps."""
    fs = 250.0
    t = np.arange(int(7.0 * fs)) / fs
    a = np.zeros_like(t)
    for c in CENTERS:
        a += gauss(t, c, 0.06, 0.12)
    drift = 0.20 * np.sin(2 * np.pi * t / 14.0) + 0.03 * t
    raw = 0.60 + drift - a

    res = analyze_v2_pipeline(raw, fs, BeatCounterConfig(), t0=0.0, meta={"time_origin_s": 0.0})
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["metadata"]["n_accepted"] == len(CENTERS)
    np.testing.assert_allclose(res["candidates"]["time_s"], CENTERS, atol=0.03)
    assert res["metadata"]["baseline_drift_range"] > 0.15


# ---------------------------------------------------------------------------
# 7. Overlapping identical samples & banker's rounding parity regression
# ---------------------------------------------------------------------------
def test_overlapping_identical_samples_and_bankers_rounding_parity():
    """Verify that overlapping rows with identical samples (including timestamps at half-integer
    sample boundaries like 372.566s and 376.570s at 250 Hz) stitch with zero conflicts."""
    fs = 250.0
    cfg = BeatCounterConfig()
    full_signal = 0.85 - 0.12 * np.sin(np.linspace(0, 6 * np.pi, 1500))
    # Simulate Dataset B Segment 12 exact timestamp grid:
    # Row 0..3 start at 372.566 + k*1.000 (265 samples), Row 3 has 266 samples, Row 4 starts at 376.570
    rows = [
        {"begin": 372.566, "y": full_signal[0:265]},
        {"begin": 373.566, "y": full_signal[250:515]},
        {"begin": 374.566, "y": full_signal[500:765]},
        {"begin": 375.566, "y": full_signal[750:1016]},  # 266 samples
        {"begin": 376.570, "y": full_signal[1001:1266]},  # 15-sample identical overlap [1001:1016]
    ]
    df, y_cols = _make_stitched_df(rows, fs=fs)
    stitched, _, meta = build_concatenated_signal(df, y_cols, fs, cfg, force_invert=False)

    assert meta["stitch_overlap_conflicts"] == 0
    assert meta["stitch_identical_overlap_samples"] == 15 + 15 + 15 + 15
    assert meta["time_origin_s"] == pytest.approx(372.566)
    np.testing.assert_allclose(stitched, full_signal[:1266])


# ---------------------------------------------------------------------------
# 8 & 9. Overlapping contradictory samples & preservation of unaffected samples
# ---------------------------------------------------------------------------
def test_overlapping_contradictory_samples_permanent_masking_and_preservation():
    """Contradictory overlapping samples must be permanently masked as NaN (even if a 3rd row
    overlaps the same indices), while all non-conflicting samples remain untouched."""
    fs = 250.0
    cfg = BeatCounterConfig()
    t = np.arange(1500) / fs
    a = np.zeros_like(t)
    for c in [0.8, 1.8, 2.8, 3.8, 4.8]:
        a += gauss(t, c, 0.06, 0.15)
    base_sig = 0.85 - a

    row0 = base_sig[0:600].copy()
    row1 = base_sig[500:1100].copy()
    row2 = base_sig[500:1500].copy()
    # Introduce contradictory values at indices 520, 521, 522 in row1 (while row2 has the original values)
    row1[20:23] += 0.35

    df, y_cols = _make_stitched_df(
        [
            {"begin": 0.0, "y": row0},
            {"begin": 500 / fs, "y": row1},
            {"begin": 500 / fs, "y": row2},
        ],
        fs=fs,
    )
    stitched, _, meta = build_concatenated_signal(df, y_cols, fs, cfg, force_invert=False)

    # Indices 520, 521, 522 must remain NaN (not overwritten by row2!)
    assert np.all(np.isnan(stitched[520:523]))
    # All unaffected samples must match base_sig exactly
    unaffected = np.ones(1500, dtype=bool)
    unaffected[520:523] = False
    np.testing.assert_allclose(stitched[unaffected], base_sig[unaffected])

    # Because only 3 / 1500 (0.2%) samples conflicted and were safely masked as NaN, pipeline recovers the 5 beats
    res = analyze_v2_pipeline(stitched, fs, cfg, t0=meta["time_origin_s"], meta=meta)
    assert res["metadata"]["qc_status"] == "PASS"
    assert "masked_overlap_conflicts" in res["metadata"]["qc_flags"]
    assert res["metadata"]["n_accepted"] == 5

    # Conversely, if > 5% of samples have contradictory overlaps, Layer 1 QC rejects with overlap_conflict
    row1_heavy = base_sig[500:1100].copy()
    row1_heavy[:120] += 0.35  # 120 / 1500 = 8% conflict
    df_h, y_cols_h = _make_stitched_df(
        [{"begin": 0.0, "y": row0}, {"begin": 500 / fs, "y": row1_heavy}, {"begin": 1000 / fs, "y": base_sig[1000:1500]}],
        fs=fs,
    )
    stitched_h, _, meta_h = build_concatenated_signal(df_h, y_cols_h, fs, cfg, force_invert=False)
    res_h = analyze_v2_pipeline(stitched_h, fs, cfg, t0=meta_h["time_origin_s"], meta=meta_h)
    assert res_h["metadata"]["qc_status"] == "REJECT"
    assert res_h["metadata"]["qc_reason"] == "overlap_conflict"


# ---------------------------------------------------------------------------
# 10. Legitimate narrow contractions & notched plateau contractions
# ---------------------------------------------------------------------------
def test_legitimate_narrow_and_notched_contractions_preserved():
    """Smooth high-frequency (3 Hz) narrow contractions and broad notched contractions must not
    be falsely rejected by the impulse filter."""
    fs = 250.0
    t = np.arange(int(5.0 * fs)) / fs
    centers_3hz = [0.4 + k / 3.0 for k in range(13)]
    a = np.zeros_like(t)
    for c in centers_3hz:
        a += gauss(t, c, 0.022, 0.12)  # smooth narrow contraction (FWHM ~ 52 ms)
    res = detect_contractions(a, fs)
    assert len(res["events"][res["events"]["status"] == ACCEPTED]) == len(centers_3hz)

    # Notched plateau contraction (two sub-lobes 45 ms apart on a 220 ms wide contraction)
    a_notch = np.zeros_like(t)
    for c in [1.0, 2.0, 3.0, 4.0]:
        a_notch += gauss(t, c, 0.045, 0.25) + gauss(t, c + 0.055, 0.045, 0.23)
    res_notch = detect_contractions(a_notch, fs)
    acc_notch = res_notch["events"][res_notch["status"] == ACCEPTED] if "status" in res_notch else res_notch["events"][res_notch["events"]["status"] == ACCEPTED]
    assert len(acc_notch) == 4


# ---------------------------------------------------------------------------
# 11 & 12. Closely spaced contractions & borderline post-contraction bounce
# ---------------------------------------------------------------------------
def test_closely_spaced_and_borderline_secondary_excursions():
    """Verify:
      - Two closely spaced contractions (240 ms apart) with full recovery are both ACCEPTED.
      - A narrow post-contraction diastolic bounce barely clearing the 50% threshold (1.02 * thr)
        within 300 ms of a dominant contraction with incomplete recovery (< 70%) is rejected as
        borderline_secondary_excursion, while a genuine independent weak contraction at 0.55 * A_ref
        with normal width is ACCEPTED.
    """
    fs = 250.0
    t = np.arange(int(8.0 * fs)) / fs
    a = np.zeros_like(t)
    # 5 dominant contractions (height ~ 0.14, sigma = 0.08s -> w50 ~ 190ms)
    for c in [1.0, 2.2, 3.4, 4.6, 7.0]:
        a += gauss(t, c, 0.08, 0.14)
    # 1 genuine weak contraction at t = 5.8 s (height ~ 0.080 > 0.070 thr, normal width sigma = 0.08s)
    a += gauss(t, 5.8, 0.08, 0.080)
    # 1 narrow post-contraction secondary bounce at t = 3.69 s (290 ms after 3.4s beat, narrower sigma = 0.025s,
    # sitting on the relaxation tail so recovery < 0.70 and height barely clears threshold ~ 0.072)
    a += gauss(t, 3.55, 0.12, 0.025) + gauss(t, 3.69, 0.025, 0.065)

    res = detect_contractions(a, fs)
    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]
    bse = ev[ev["reason"] == "borderline_secondary_excursion"]

    # The 5 dominant beats + 1 genuine weak beat at 5.8 s are ACCEPTED (6 total)
    assert len(acc) == 6
    np.testing.assert_allclose(acc["time_rel_s"].to_numpy(), [1.0, 2.2, 3.4, 4.6, 5.8, 7.0], atol=0.04)
    # The post-contraction bounce at ~3.69 s is rejected as borderline_secondary_excursion
    assert len(bse) == 1
    assert bse.iloc[0]["time_rel_s"] == pytest.approx(3.69, abs=0.03)


# ---------------------------------------------------------------------------
# Real experimental workbook verification (Problems A, B, C, D, E, F)
# ---------------------------------------------------------------------------
def test_real_workbooks_problems_a_to_f():
    """Verify exact resolution of Problems A-F on the real experimental workbooks if present."""
    wb_a = Path("SCRAMBLED ISO individual transients result.xlsx")
    wb_b = Path("05-12-24.10D.CGP.SEARCH.BASE.NICONICO.ISOISO.5ISOMISTAKE individual transients result.xlsx")
    if not wb_a.exists() or not wb_b.exists():
        pytest.skip("Experimental neonatal workbooks not present in working directory.")

    cfg = BeatCounterConfig()
    xl_a = pd.ExcelFile(wb_a)
    xl_b = pd.ExcelFile(wb_b)

    # Problem A: Dataset B Segment 12 -> 0 overlap conflicts, QC PASS, 11 accepted contractions
    df12, y12, fs12, _ = load_cytocypher_excel(xl_b, sheet_name="PixelCorrelation Segment 12")
    sig12, _, meta12 = build_concatenated_signal(df12, y12, fs12, cfg, force_invert=False)
    assert meta12["stitch_overlap_conflicts"] == 0
    res12 = analyze_v2_pipeline(sig12, fs12, cfg, t0=meta12["time_origin_s"], meta=meta12)
    assert res12["metadata"]["qc_status"] == "PASS"
    assert res12["metadata"]["n_accepted"] == 11

    # Problem B: Dataset B Segment 13 -> impulse at ~404.896 s rejected, 10 accepted contractions
    df13, y13, fs13, _ = load_cytocypher_excel(xl_b, sheet_name="PixelCorrelation Segment 13")
    sig13, _, meta13 = build_concatenated_signal(df13, y13, fs13, cfg, force_invert=False)
    res13 = analyze_v2_pipeline(sig13, fs13, cfg, t0=meta13["time_origin_s"], meta=meta13)
    ev13 = res13["events"]
    assert res13["metadata"]["n_accepted"] == 10
    imp13 = ev13[ev13["reason"] == "isolated_impulse_artefact"]
    assert len(imp13) == 1 and imp13.iloc[0]["time_abs_s"] == pytest.approx(404.896, abs=0.02)

    # Problem C: Dataset A Segment 5 -> 21 accepted contractions, 3 isolated impulses rejected,
    # normal contractions at 359.656s and 365.644s rescued, embedded spikes localized on plateau
    df5, y5, fs5, _ = load_cytocypher_excel(xl_a, sheet_name="PixelCorrelation Segment 5")
    sig5, _, meta5 = build_concatenated_signal(df5, y5, fs5, cfg, force_invert=False)
    res5 = analyze_v2_pipeline(sig5, fs5, cfg, t0=meta5["time_origin_s"], meta=meta5)
    ev5 = res5["events"]
    acc5 = ev5[ev5["status"] == ACCEPTED]
    imp5 = ev5[ev5["reason"] == "isolated_impulse_artefact"]
    assert len(acc5) == 21
    assert len(imp5) == 3
    assert np.any(np.abs(acc5["time_abs_s"] - 359.656) < 0.02)
    assert np.any(np.abs(acc5["time_abs_s"] - 365.644) < 0.02)
    # All 1 Hz beats with embedded spikes (e.g. ~356.68s, ~360.69s, ~362.65s, ~366.65s, ~371.69s) are localized on <0.095 plateau
    spiked_beats = acc5[acc5["time_abs_s"] < 371.0]
    assert np.all(spiked_beats["amplitude"] < 0.095)

    # Problem D: Dataset A Segment 6 -> REJECT (noise_dominated_uninterpretable)
    df6, y6, fs6, _ = load_cytocypher_excel(xl_a, sheet_name="PixelCorrelation Segment 6")
    sig6, _, meta6 = build_concatenated_signal(df6, y6, fs6, cfg, force_invert=False)
    res6 = analyze_v2_pipeline(sig6, fs6, cfg, t0=meta6["time_origin_s"], meta=meta6)
    assert res6["metadata"]["qc_status"] == "REJECT"
    assert res6["metadata"]["qc_reason"] == "noise_dominated_uninterpretable"

    # Problem E: Dataset B Segment 42 -> 1045.192 s rejected as borderline_secondary_excursion, 10 primary beats accepted
    df42, y42, fs42, _ = load_cytocypher_excel(xl_b, sheet_name="PixelCorrelation Segment 42")
    sig42, _, meta42 = build_concatenated_signal(df42, y42, fs42, cfg, force_invert=False)
    res42 = analyze_v2_pipeline(sig42, fs42, cfg, t0=meta42["time_origin_s"], meta=meta42)
    ev42 = res42["events"]
    assert res42["metadata"]["n_accepted"] == 10
    bse42 = ev42[ev42["reason"] == "borderline_secondary_excursion"]
    assert len(bse42) == 1 and bse42.iloc[0]["time_abs_s"] == pytest.approx(1045.192, abs=0.02)

    # Problem F: Dataset B Segment 38 -> ambiguous 981.816 s preserved as duplicate_peak_within_cycle, 12 accepted
    df38, y38, fs38, _ = load_cytocypher_excel(xl_b, sheet_name="PixelCorrelation Segment 38")
    sig38, _, meta38 = build_concatenated_signal(df38, y38, fs38, cfg, force_invert=False)
    res38 = analyze_v2_pipeline(sig38, fs38, cfg, t0=meta38["time_origin_s"], meta=meta38)
    assert res38["metadata"]["n_accepted"] == 12
    dup38 = res38["events"][res38["events"]["reason"] == "duplicate_peak_within_cycle"]
    assert np.any(np.abs(dup38["time_abs_s"] - 981.816) < 0.03)


# ---------------------------------------------------------------------------
# 13. Continuous strictness coefficient mappings & monotonicity
# ---------------------------------------------------------------------------
def test_strictness_continuous_mappings_and_monotonicity():
    from pixel_counter.v2_detector import (
        amplitude_strictness_to_amplitude,
        separation_strictness_to_recovery,
        strictness_to_params,
    )

    grid = np.linspace(0.0, 2.0, 21)
    recs = [separation_strictness_to_recovery(s) for s in grid]
    amps = [amplitude_strictness_to_amplitude(s) for s in grid]

    # Monotonicity check
    assert np.all(np.diff(recs) >= 0.0)
    assert np.all(np.diff(amps) >= 0.0)

    # Valid bounds for DetectorParams: strictly within (0, 1)
    for r in recs:
        assert 0.0 < r < 1.0
    for a in amps:
        assert 0.0 < a < 1.0

    # Default at 1.0 produces 0.50
    assert separation_strictness_to_recovery(1.0) == pytest.approx(0.50)
    assert amplitude_strictness_to_amplitude(1.0) == pytest.approx(0.50)

    # Factory creates valid DetectorParams
    p_def = strictness_to_params(1.0, 1.0)
    assert p_def.min_rel_recovery == pytest.approx(0.50)
    assert p_def.min_rel_amplitude == pytest.approx(0.50)
    assert p_def.separation_strictness == pytest.approx(1.0)
    assert p_def.amplitude_strictness == pytest.approx(1.0)

    p_perm = strictness_to_params(0.0, 0.0)
    assert p_perm.min_rel_recovery == pytest.approx(0.02)
    assert p_perm.min_rel_amplitude == pytest.approx(0.001)

    p_strict = strictness_to_params(2.0, 2.0)
    assert p_strict.min_rel_recovery == pytest.approx(0.95)
    assert p_strict.min_rel_amplitude == pytest.approx(0.98)


# ---------------------------------------------------------------------------
# 14. CLI strictness parameters integration in review utility
# ---------------------------------------------------------------------------
def test_strictness_cli_and_config_integration():
    import run_v2_review_utility as rv

    # User supplies separation_strictness=0.75
    args_sep = type("Args", (), {
        "separation_strictness": 0.75,
        "amplitude_strictness": None,
        "min_rel_amplitude": None,
        "min_rel_recovery": None,
        "min_snr": None,
    })()
    runs_sep = rv.build_params_by_name(args_sep)
    assert "custom" in runs_sep
    assert runs_sep["custom"].separation_strictness == pytest.approx(0.75)
    assert runs_sep["custom"].min_rel_recovery == pytest.approx(0.375)
    assert runs_sep["custom"].amplitude_strictness == pytest.approx(1.0)

    # Precedence: separation_strictness overrides min_rel_recovery if both passed
    args_both = type("Args", (), {
        "separation_strictness": 1.20,
        "amplitude_strictness": 0.80,
        "min_rel_amplitude": 0.30,  # should be overridden by amplitude_strictness=0.80
        "min_rel_recovery": 0.20,   # should be overridden by separation_strictness=1.20
        "min_snr": None,
    })()
    runs_both = rv.build_params_by_name(args_both)
    assert "custom" in runs_both
    assert runs_both["custom"].separation_strictness == pytest.approx(1.20)
    assert runs_both["custom"].min_rel_recovery == pytest.approx(0.60)
    assert runs_both["custom"].amplitude_strictness == pytest.approx(0.80)
    assert runs_both["custom"].min_rel_amplitude == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# 15. Uninterpretable/rejected signals report NaN frequency, avoiding 0.00 Hz
# ---------------------------------------------------------------------------
def test_uninterpretable_recordings_report_nan_frequency():
    from pixel_counter.qc_v2 import compute_descriptive_metrics

    # Normal accepted beats -> valid frequency
    beats = np.array([1.0, 2.0, 3.0, 4.0])
    m_ok = compute_descriptive_metrics(beats, np.array([0.1, 0.1, 0.1, 0.1]), analyzable_duration_s=4.0)
    assert m_ok["contraction_frequency_hz"] == pytest.approx(1.0)

    # Zero beats on clean quiescent signal -> NaN frequency
    m_zero = compute_descriptive_metrics(np.array([]), np.array([]), analyzable_duration_s=4.0, is_rejected=False)
    assert np.isnan(m_zero["contraction_frequency_hz"])

    # Rejected uninterpretable recording -> NaN frequency (never misleading 0.00 Hz)
    m_rej = compute_descriptive_metrics(np.array([]), np.array([]), analyzable_duration_s=0.0, is_rejected=True)
    assert np.isnan(m_rej["contraction_frequency_hz"])

    # Pipeline output on pure noise
    fs = 250.0
    rng = np.random.default_rng(42)
    noise_sig = rng.normal(0.5, 0.015, size=int(10.0 * fs))
    cfg = BeatCounterConfig()
    res = analyze_v2_pipeline(noise_sig, fs, cfg)
    assert res["metadata"]["qc_status"] == "REJECT"
    assert np.isnan(res["metadata"]["contraction_frequency_hz"])


# ---------------------------------------------------------------------------
# 16. Noise-dominated uninterpretable recording rejection (SCRAMBLED ISO Seg 2)
# ---------------------------------------------------------------------------
def test_scrambled_iso_segment_2_noise_dominated_rejection():
    """SCRAMBLED ISO Segment 2 exhibits excessive noise, baseline instability, and dozens of
    noise excursions rejected as baseline_noise. Even with manual polarity override (+1),
    Layer 2 QC must reject it as noise_dominated_uninterpretable."""
    wb_scrambled = Path(__file__).resolve().parent.parent / "SCRAMBLED ISO individual transients result.xlsx"
    if not wb_scrambled.exists():
        pytest.skip(f"Workbook not found: {wb_scrambled}")

    cfg = BeatCounterConfig()
    df, y, fs, _ = load_cytocypher_excel(str(wb_scrambled), sheet_name="PixelCorrelation Segment 2")
    sig, _, meta = build_concatenated_signal(df, y, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    # Evaluate with manual polarity override +1
    res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=1)
    md = res["metadata"]

    assert md["qc_status"] == "REJECT"
    assert md["qc_reason"] == "noise_dominated_uninterpretable"
    assert md["signal_quality_category"] == "noise_dominated_uninterpretable"
    assert md["qc_layer2_status"] == "REJECT"
    assert np.isnan(md["contraction_frequency_hz"])


# ---------------------------------------------------------------------------
# 17. Clean and borderline recordings preserved (NICONICO Segments 70 & 84)
# ---------------------------------------------------------------------------
def test_clean_and_borderline_records_preserved_pass():
    """Valid recordings (including borderline neonatal recordings in NICONICO Segments 70 & 84)
    must continue to pass QC without false rejections."""
    wb_niconico = Path(__file__).resolve().parent.parent / "05-12-24.10D.CGP.SEARCH.BASE.NICONICO.ISOISO.5ISOMISTAKE individual transients result.xlsx"
    if not wb_niconico.exists():
        pytest.skip(f"Workbook not found: {wb_niconico}")

    cfg = BeatCounterConfig()
    for seg_num in [70, 84]:
        df, y, fs, _ = load_cytocypher_excel(str(wb_niconico), sheet_name=f"PixelCorrelation Segment {seg_num}")
        sig, _, meta = build_concatenated_signal(df, y, fs, cfg, force_invert=False)
        t0 = float(meta.get("time_origin_s", 0.0))

        res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=-1)
        md = res["metadata"]

        assert md["qc_status"] == "PASS"
        assert md["qc_layer2_status"] == "PASS"
        assert md["n_accepted"] > 0
        assert np.isfinite(md["contraction_frequency_hz"])
        assert md["contraction_frequency_hz"] > 1.0


# ---------------------------------------------------------------------------
# 18. Calcium-assisted tracking artifact detection
# ---------------------------------------------------------------------------
def test_calcium_assisted_tracking_artifact_rejection():
    """When calcium metadata indicates equal bidirectional matches (tracking slippage or motion
    across tracking boundaries), Layer 2 QC must reject with reason tracking_artifact."""
    fs = 250.0
    t = np.arange(int(6.0 * fs)) / fs
    sig = 0.50 + 0.10 * np.sin(2 * np.pi * 1.5 * t)
    cfg = BeatCounterConfig()

    meta = {
        "time_origin_s": 0.0,
        "calcium_matches_positive": 7,
        "calcium_matches_negative": 7,
    }
    res = analyze_v2_pipeline(sig, fs, cfg, t0=0.0, meta=meta, polarity_override=1)
    md = res["metadata"]

    assert md["qc_status"] == "REJECT"
    assert md["qc_reason"] == "tracking_artifact"
    assert md["signal_quality_category"] == "tracking_artifact"


# ---------------------------------------------------------------------------
# 19. Sidecar segment rejections persistence and loading
# ---------------------------------------------------------------------------
def test_sidecar_segment_rejections_persistence(tmp_path):
    """save_polarity_overrides and load_segment_rejections must cleanly round-trip segment rejections."""
    from pixel_counter.polarity_overrides import (
        load_polarity_overrides,
        load_segment_rejections,
        save_polarity_overrides,
    )

    fake_wb = tmp_path / "mock_workbook.xlsx"
    fake_wb.write_text("dummy")

    overrides = {17: -1, 45: 1}
    rejections = {
        39: {"rejected": True, "reason": "Tracking or motion artifact"},
        2: {"rejected": True, "reason": "Poor signal / excessive noise"},
    }

    out_json = save_polarity_overrides(fake_wb, overrides, segment_rejections=rejections)
    assert out_json.exists()

    loaded_ov = load_polarity_overrides(fake_wb, out_json)
    assert loaded_ov == overrides

    loaded_rej = load_segment_rejections(fake_wb, out_json)
    assert len(loaded_rej) == 2
    assert loaded_rej[39]["rejected"] is True
    assert loaded_rej[39]["reason"] == "Tracking or motion artifact"
    assert loaded_rej[2]["rejected"] is True
    assert loaded_rej[2]["reason"] == "Poor signal / excessive noise"


# ---------------------------------------------------------------------------
# 20. Review utility HTML includes manual rejection controls and initial state
# ---------------------------------------------------------------------------
def test_review_utility_html_includes_rejection_controls(tmp_path):
    """generate_html must embed rejection controls and initial segment rejections."""
    from run_v2_review_utility import generate_html

    out_html = tmp_path / "test_review.html"
    mock_seg = {
        "sheet_name": "PixelCorrelation Segment 39",
        "segment_number": 39,
        "sample_id": "TEST_39",
        "fs": 250.0,
        "t0": 0.0,
        "length": 100,
        "qc_status": "PASS",
        "qc_reason": "clean_signal",
        "raw_polarity": 1,
        "polarity_source": "automatic",
        "polarity_method": "baseline_noise_ratio",
        "polarity_reason": "test",
        "auto_polarity": 1,
        "auto_method": "test",
        "auto_reason": "test",
        "effective_polarity": 1,
        "has_amplitude": False,
        "sig_raw_b64": "",
        "baseline_raw_b64": None,
        "amplitude_b64": None,
        "sig_int_b64": None,
        "sig_cal_b64": None,
        "runs": {"default": {"qc_status": "PASS", "qc_reason": "clean_signal", "events": {"index": []}}},
    }
    seg_rejections = {"39": {"rejected": True, "reason": "Tracking or motion artifact"}}

    generate_html([mock_seg], "test_wb.xlsx", str(out_html), segment_rejections=seg_rejections)
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")

    assert "chk_segment_rejected" in content
    assert "segment_rejection_reason" in content
    assert "Tracking or motion artifact" in content
    assert "initialSegmentRejections" in content
    assert '"rejected": true' in content


# ---------------------------------------------------------------------------
# 21. Reconcile declared vs implied sampling frequency
# ---------------------------------------------------------------------------
def test_declared_vs_implied_sampling_frequency_reconciliation(tmp_path):
    """When metadata Sampling Frequency contradicts consistent row timestamps by > 1 Hz,
    load_cytocypher_excel should reconcile and adopt the implied frequency."""
    fs_true = 250.0
    dt_per_sample = 1.0 / fs_true
    n_samples = 265
    row_dur = (n_samples - 1) * dt_per_sample

    # Construct two rows with declared fs = 208.32 but row timestamps matching 250 Hz exactly
    records = []
    for r_idx in range(3):
        rec = {
            "Sampling Frequency": 208.32,  # Conflicting metadata
            "Begin": float(r_idx * 1.0),
            "End": float(r_idx * 1.0 + row_dur),
        }
        for j in range(n_samples):
            rec[f"y {j}"] = 0.5 + 0.1 * np.sin(2 * np.pi * j / 25.0)
        records.append(rec)

    df_test = pd.DataFrame(records)
    test_file = tmp_path / "test_implied_fs.xlsx"
    df_test.to_excel(test_file, sheet_name="PixelCorrelation Segment 1", index=False)

    df_loaded, y_cols, fs_inferred, _ = load_cytocypher_excel(test_file, sheet_name="PixelCorrelation Segment 1")
    assert fs_inferred == pytest.approx(250.0, abs=0.01), f"Expected 250.0 Hz, got {fs_inferred}"

    # Verify that stitching with this reconciled fs yields 0 overlap conflicts
    sig_raw, _, meta = build_concatenated_signal(df_loaded, y_cols, fs_inferred, BeatCounterConfig())
    assert meta.get("stitch_overlap_conflicts", 0) == 0


# ---------------------------------------------------------------------------
# 22. High-SNR contraction recordings protected from noise-domination rejection
# ---------------------------------------------------------------------------
def test_high_snr_baseline_noise_false_rejection_protection():
    """Recordings with high median contraction SNR (>= 20.0) must not be falsely rejected
    as noise_dominated_uninterpretable when tiny baseline excursions are filtered."""
    fs = 250.0
    t = np.arange(int(10.0 * fs)) / fs
    # Generate 10 strong contractions with amplitude 0.20 and SNR ~ 100
    amp = np.zeros_like(t)
    for c in np.arange(0.8, 9.8, 0.9):
        amp += gauss(t, c, 0.065, 0.20)

    # Add very small noise (sn ~ 0.001) with minor baseline wiggles between beats
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.001, size=len(t))
    sig = amp + noise
    # Artificially inject multiple small sub-threshold baseline bumps that get rejected
    for bump_t in np.arange(1.3, 9.3, 0.9):
        sig += gauss(t, bump_t, 0.040, 0.006)

    params = DetectorParams()
    res = detect_contractions(sig, fs, params)
    n_acc = len(res["events"][res["events"]["status"] == ACCEPTED])
    assert n_acc == 10

    # Test analyze_v2_pipeline
    meta = {"time_origin_s": 0.0, "stitch_overlap_conflicts": 0}
    pipe_res = analyze_v2_pipeline(sig, fs, BeatCounterConfig(), t0=0.0, meta=meta, polarity_override=1)
    assert pipe_res["metadata"]["qc_status"] == "PASS"
    assert pipe_res["metadata"]["qc_layer2_status"] == "PASS"
    assert pipe_res["metadata"]["qc_reason"] in ("clean_signal", "recoverable_contractions_in_noise")


# ---------------------------------------------------------------------------
# 23. Impulse filter peak bounding slice safety
# ---------------------------------------------------------------------------
def test_impulse_filter_peak_ordering_bounds_robustness():
    """Ensure _refine_events_for_impulses never encounters empty slice bounds even
    when candidate events are closely spaced or snapped."""
    fs = 250.0
    t = np.arange(int(4.0 * fs)) / fs
    sig = 0.1 * np.sin(2 * np.pi * t)
    # Add adjacent spikes at exactly identical / adjacent indices
    sig[250] = 0.50
    sig[251] = 0.49
    params = DetectorParams()
    # detect_contractions must complete without ValueError
    res = detect_contractions(sig, fs, params)
    assert "events" in res
    assert isinstance(res["events"], pd.DataFrame)


# ---------------------------------------------------------------------------
# 24. Review utility HTML template styling for auto vs manual additions
# ---------------------------------------------------------------------------
def test_review_utility_manual_vs_auto_marker_styling():
    """HTML_TEMPLATE must distinguish Automatically accepted (green) from Manually added (blue)
    with informative hover tooltips."""
    import run_v2_review_utility as rv
    html = rv.HTML_TEMPLATE
    assert ("Automatically accepted" in html) or ("AUTO ACCEPTED" in html)
    assert ("Manually added" in html) or ("MANUALLY ADDED" in html)
    assert "#28a745" in html  # Green auto accepted
    assert "#007bff" in html  # Blue manually added
    assert "Manual addition" in html


# ---------------------------------------------------------------------------
# 25. Problem A: Noise-only and excessive-noise segments rejected in Layer 2 QC
# ---------------------------------------------------------------------------
def test_problem_a_noise_only_and_excessive_noise_rejection():
    """All Problem A noise-only and excessive-noise segments must fail Layer 2 QC
    (noise_dominated_uninterpretable) even when manual polarity (+1 or -1) is applied."""
    all_data = Path(__file__).resolve().parent.parent / "all data"
    if not all_data.exists():
        pytest.skip("all data directory not found")

    cfg = BeatCounterConfig()
    problem_a_cases = [
        ("240926_IMR90_1,2,3HZ individual transients result.xlsx", [18]),
        ("DISH1TEST individual transients result.xlsx", [13, 14]),
        ("05-12-24.10D.CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx", [73]),
        (
            "17-12-24.8D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISO individual transients result.xlsx",
            [35, 37, 39, 45, 46, 49, 50, 52, 58, 59, 60, 61, 63, 64, 65, 66, 67, 70],
        ),
        (
            "DISH2CGP.BASE.BASE.NICO.NICO individual transients result.xlsx",
            [33, 34, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54, 55, 56],
        ),
    ]
    for wb_name, seg_ids in problem_a_cases:
        wb_path = all_data / wb_name
        if not wb_path.exists():
            continue
        xl = pd.ExcelFile(wb_path)
        for seg_id in seg_ids:
            df, ycols, fs, _ = load_cytocypher_excel(xl, sheet_name=f"PixelCorrelation Segment {seg_id}")
            sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
            t0 = float(meta.get("time_origin_s", 0.0))
            for pol_ov in (None, 1, -1):
                res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=pol_ov)
                md = res["metadata"]
                assert md["qc_status"] == "REJECT", f"{wb_name} Seg {seg_id} (pol={pol_ov}) should be REJECT"
                if pol_ov is not None:
                    assert md["qc_layer2_status"] == "REJECT"
                    assert md["qc_reason"] == "noise_dominated_uninterpretable"


# ---------------------------------------------------------------------------
# 26. Problem A Special Case: DISH1CONTROL Segment 55 preserves 2 real beats only
# ---------------------------------------------------------------------------
def test_problem_a_dish1control_segment_55_two_real_peaks_preserved():
    """DISH1CONTROL Segment 55 must pass QC and detect ONLY the 2 genuine contractions
    (~1171.362s and ~1177.362s) while rejecting all 15 tiny quantized baseline fluctuations."""
    wb_path = (
        Path(__file__).resolve().parent.parent
        / "all data"
        / "17-12-24.8D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISO individual transients result.xlsx"
    )
    if not wb_path.exists():
        pytest.skip("Workbook not found")

    cfg = BeatCounterConfig()
    df, ycols, fs, _ = load_cytocypher_excel(wb_path, sheet_name="PixelCorrelation Segment 55")
    sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta)
    md = res["metadata"]
    assert md["qc_status"] == "PASS"
    assert md["raw_polarity"] == -1
    assert md["n_accepted"] == 2

    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]
    assert len(acc) == 2
    peak_times = t0 + acc["time_rel_s"].to_numpy(dtype=float)
    peak_amps = acc["amplitude"].to_numpy(dtype=float)
    assert peak_times[0] == pytest.approx(1171.362, abs=0.05)
    assert peak_times[1] == pytest.approx(1177.362, abs=0.05)
    assert np.all(peak_amps > 0.18)


# ---------------------------------------------------------------------------
# 27. Problem B: Incompatible bidirectional contraction excursions rejected by QC
# ---------------------------------------------------------------------------
def test_problem_b_incompatible_bidirectional_contraction_rejection():
    """25092026_IMR90 Seg 39 and DISH1NICOTINE Seg 18 & 19 contain incompatible
    contraction directions / tracking reversal and must be rejected by Layer 2 QC
    using PixelCorrelation alone, even under manual polarity override (+1 or -1)."""
    all_data = Path(__file__).resolve().parent.parent / "all data"
    if not all_data.exists():
        pytest.skip("all data directory not found")

    cfg = BeatCounterConfig()
    assert not getattr(cfg, "v2_use_calcium_polarity", False)

    problem_b_cases = [
        ("25092026_IMR90_1,2,3HZ individual transients result.xlsx", [39]),
        ("DISH1NICOTINE individual transients result.xlsx", [18, 19]),
    ]
    for wb_name, seg_ids in problem_b_cases:
        wb_path = all_data / wb_name
        if not wb_path.exists():
            continue
        xl = pd.ExcelFile(wb_path)
        for seg_id in seg_ids:
            df, ycols, fs, _ = load_cytocypher_excel(xl, sheet_name=f"PixelCorrelation Segment {seg_id}")
            sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
            t0 = float(meta.get("time_origin_s", 0.0))
            for pol_ov in (None, 1, -1):
                res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=pol_ov)
                md = res["metadata"]
                assert md["qc_status"] == "REJECT", f"{wb_name} Seg {seg_id} (pol={pol_ov}) must fail QC"
                assert md["qc_layer2_status"] == "REJECT"
                assert md["qc_reason"] == "incompatible_contraction_directions"
                assert md["signal_quality_category"] == "tracking_artifact"

    # Verify clean segments in DISH1NICOTINE (e.g. Seg 17 and Seg 20) still PASS
    wb_nico = all_data / "DISH1NICOTINE individual transients result.xlsx"
    if wb_nico.exists():
        xl_nico = pd.ExcelFile(wb_nico)
        for clean_seg in (17, 20):
            df, ycols, fs, _ = load_cytocypher_excel(xl_nico, sheet_name=f"PixelCorrelation Segment {clean_seg}")
            sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
            res = analyze_v2_pipeline(sig, fs, cfg, t0=float(meta.get("time_origin_s", 0.0)), meta=meta)
            assert res["metadata"]["qc_status"] == "PASS"
            assert res["metadata"]["n_accepted"] == 10


# ---------------------------------------------------------------------------
# 28. Problem C: No accepted peak may fall below displayed amplitude_threshold
# ---------------------------------------------------------------------------
def test_problem_c_no_accepted_peaks_below_displayed_amplitude_threshold():
    """In DISH2CGP Segment 81 (polarity -1), peaks at 1476.510s (amp=0.0191) and
    1482.342s (amp=0.0147) lie below amplitude_threshold (0.0218) after embedded-spike
    refinement and must be classified as REJECTED (insufficient_amplitude)."""
    wb_path = (
        Path(__file__).resolve().parent.parent
        / "all data"
        / "17-12-24.8D.DISH2CGP.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    )
    if not wb_path.exists():
        pytest.skip("Workbook not found")

    cfg = BeatCounterConfig()
    df, ycols, fs, _ = load_cytocypher_excel(wb_path, sheet_name="PixelCorrelation Segment 81")
    sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=-1)
    thr = float(res["metadata"]["amplitude_threshold"])
    assert thr == pytest.approx(0.0218, abs=1e-3)

    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]
    assert len(acc) == 41
    assert np.all(acc["amplitude"].to_numpy(dtype=float) >= thr)
    assert np.all(acc["height"].to_numpy(dtype=float) >= thr)

    # Specifically check the two sub-threshold peaks near 1476.510s and 1482.342s
    ev_times = t0 + ev["time_rel_s"].to_numpy(dtype=float)
    for target_t in (1476.510, 1482.342):
        idx_near = int(np.argmin(np.abs(ev_times - target_t)))
        assert abs(ev_times[idx_near] - target_t) < 0.08
        assert ev.iloc[idx_near]["status"] == REJECTED
        assert ev.iloc[idx_near]["reason"] == "insufficient_amplitude"


# ---------------------------------------------------------------------------
# 29. Problem D / Corrective Patch A: DISH2ICI Segment 23 strict source fidelity & provenance
# ---------------------------------------------------------------------------
def test_problem_d_dish2ici_segment_23_missing_intervals_recovered():
    """13-12-24.4D.DISH2ICI Segment 23 has CytoSolver truncated Rows 3 and 5 where
    End - Begin is ~1.97s and ~1.76s but only 218 and 217 samples (~0.87s) were exported.
    build_concatenated_signal must NEVER shift later rows backward to close genuine missing
    intervals; it must preserve the full 11.61s (2,903-sample) span, explicitly mark the
    two missing intervals (~559.95-561.73s and ~562.78-563.62s), maintain 100% sample
    provenance to the XLSX sheet, and detect all 10 contractions."""
    wb_path = (
        Path(__file__).resolve().parent.parent
        / "all data"
        / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    )
    if not wb_path.exists():
        pytest.skip("Workbook not found")

    cfg = BeatCounterConfig()
    df, ycols, fs, _ = load_cytocypher_excel(wb_path, sheet_name="PixelCorrelation Segment 23")
    assert fs == pytest.approx(250.0, abs=0.01)

    sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
    assert len(sig) == 2903
    assert int(np.sum(np.isfinite(sig))) == 2433
    assert int(np.sum(np.isnan(sig))) == 470
    assert meta.get("stitch_row_duration_mismatch_rows") == [3, 5]

    # Verify 100% sample provenance against original DataFrame cells
    prov_r = meta["stitch_provenance_row"]
    prov_c = meta["stitch_provenance_col"]
    finite_indices = np.where(np.isfinite(sig))[0]
    for k in finite_indices:
        r_idx = int(prov_r[k])
        c_idx = int(prov_c[k])
        assert r_idx >= 0 and c_idx >= 0
        expected_val = float(df.loc[r_idx, ycols[c_idx]])
        assert sig[k] == pytest.approx(expected_val, abs=1e-12)

    t0 = float(meta.get("time_origin_s", 0.0))
    assert t0 == pytest.approx(557.6176, abs=1e-3)
    t_end = t0 + (len(sig) - 1) / fs
    assert t_end == pytest.approx(569.2256, abs=1e-3)

    res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta)
    md = res["metadata"]
    assert md["qc_status"] == "PASS"
    excl = md["excluded_intervals_s"]
    assert len(excl) == 2
    assert excl[0][0] == pytest.approx(559.95, abs=0.05)
    assert excl[0][1] == pytest.approx(561.73, abs=0.05)
    assert excl[1][0] == pytest.approx(562.78, abs=0.05)
    assert excl[1][1] == pytest.approx(563.62, abs=0.05)
    assert md["n_accepted"] == 10


# ---------------------------------------------------------------------------
# 30. Problem E & F: Blue manual markers and Per-Peak Strictness controls in HTML UI
# ---------------------------------------------------------------------------
def test_problem_e_and_f_review_ui_blue_manual_and_per_peak_strictness():
    """Verify that run_v2_review_utility HTML_TEMPLATE supports BLUE manual additions
    when clicking the amplitude trace or promoting a rejected candidate, and provides
    Per-Peak Detector Strictness controls (inherit, strict, default, permissive) with
    persistence and CSV/JSON export."""
    import run_v2_review_utility as rv
    html = rv.HTML_TEMPLATE

    # Problem E: Blue manual marker assignment & persistence
    assert 'id="peak_strictness_panel"' in html
    assert 'id="peak_strictness_select"' in html
    assert 'id="peak_select"' in html
    assert 'id="click_mode_select"' not in html
    assert 'origin: "manual"' in html
    assert "manualPeaksByPol" in html

    # Problem F: Per-peak strictness options, evaluation, deduplication, and export
    assert ('<option value="inherit">Use global setting</option>' in html) or ('<option value="inherit">Inherit global</option>' in html)
    assert '<option value="strict">Strict</option>' in html
    assert '<option value="default">Default</option>' in html
    assert '<option value="permissive">Permissive</option>' in html
    assert "perPeakStrictness" in html
    assert "findCandidateInRun" in html
    assert "syncPeakState" in html
    assert "per_peak_strictness" in html
    assert '"manual_peak_edits": manualEdits' in html


# ---------------------------------------------------------------------------
# 31. Corrective Patch B: QC gating precedes reportable peak detection
# ---------------------------------------------------------------------------
def test_qc_gating_precedes_reportable_peak_detection():
    """When a segment fails QC (e.g. 05-12-24.10D.CONTROL Segment 73), final reportable
    accepted contraction count must be 0, candidates must be empty, events must contain
    0 ACCEPTED rows, and contraction_frequency_hz must be NaN, while preliminary detector
    candidates remain available in diagnostic_events."""
    wb_path = (
        Path(__file__).resolve().parent.parent
        / "all data"
        / "05-12-24.10D.CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    )
    if not wb_path.exists():
        pytest.skip("Workbook not found")

    cfg = BeatCounterConfig()
    df, ycols, fs, _ = load_cytocypher_excel(wb_path, sheet_name="PixelCorrelation Segment 73")
    sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    for pol_ov in (None, 1, -1):
        res = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=pol_ov)
        md = res["metadata"]
        assert md["qc_status"] == "REJECT"
        assert md["n_accepted"] == 0
        assert len(res["candidates"]["time_s"]) == 0
        assert int((res["events"]["status"] == ACCEPTED).sum()) == 0
        assert np.isnan(md["contraction_frequency_hz"])
        if pol_ov is not None:
            # Preliminary candidates are preserved only in diagnostic_events
            diag_ev = res["diagnostic_events"]
            assert int((diag_ev["status"] == ACCEPTED).sum()) > 0
            assert np.all(res["events"]["reason"].str.startswith("qc_rejected_") | (res["events"]["status"] == REJECTED))


# ---------------------------------------------------------------------------
# 32. Corrective Patch D: SCRAMBLED ISO Segment 8 & 3 pass QC while Segment 2 & 6 reject
# ---------------------------------------------------------------------------
def test_scrambled_iso_segment_8_and_3_pass_while_noise_segments_reject():
    """SCRAMBLED ISO Segment 8 (27 contractions) and Segment 3 (40 contractions) have clear
    contraction trains on a drifting baseline and must PASS QC, whereas noise-dominated
    Segment 2 and Segment 6 must remain REJECT with 0 reportable contractions."""
    wb_path = (
        Path(__file__).resolve().parent.parent
        / "all data"
        / "SCRAMBLED ISO individual transients result.xlsx"
    )
    if not wb_path.exists():
        pytest.skip("Workbook not found")

    cfg = BeatCounterConfig()
    xl = pd.ExcelFile(wb_path)

    for seg_id, expected_acc in ((8, 27), (3, 40)):
        df, ycols, fs, _ = load_cytocypher_excel(xl, sheet_name=f"PixelCorrelation Segment {seg_id}")
        sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
        res = analyze_v2_pipeline(sig, fs, cfg, t0=float(meta.get("time_origin_s", 0.0)), meta=meta)
        md = res["metadata"]
        assert md["qc_status"] == "PASS", f"SCRAMBLED ISO Seg {seg_id} should PASS QC"
        assert md["n_accepted"] == expected_acc

    for rej_seg in (2, 6):
        df, ycols, fs, _ = load_cytocypher_excel(xl, sheet_name=f"PixelCorrelation Segment {rej_seg}")
        sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
        res = analyze_v2_pipeline(sig, fs, cfg, t0=float(meta.get("time_origin_s", 0.0)), meta=meta)
        md = res["metadata"]
        assert md["qc_status"] == "REJECT", f"SCRAMBLED ISO Seg {rej_seg} should REJECT QC"
        assert md["n_accepted"] == 0


# ---------------------------------------------------------------------------
# 33. Focused Final UI Corrections: Lifecycle of manual BLUE peaks & simplified legend
# ---------------------------------------------------------------------------
def test_interactive_review_manual_peak_lifecycle_and_simplified_legend():
    """Verify the focused final UI refinements:
    1. Manually added peaks follow NOTHING -> BLUE -> NOTHING lifecycle:
       - Clicking unmarked trace creates BLUE peak (origin='manual', action='ADDED').
       - Clicking the same BLUE peak again completely deletes it (calls deleteManualPeak,
         removes marker, decrements count, deletes from manualPeaksByPol, leaves NO red cross).
       - Explicit btn_remove_manual_peak completely deletes selected manual peak.
    2. Auto peaks toggle GREEN -> RED CROSS -> GREEN (origin='auto' preserved on restore).
    3. Simplified legend contains:
       - GREEN = Automatically accepted
       - BLUE = Manually added
       - RED = Rejected automatic peak
       - GREY excluded candidates are removed from the normal review legend and only shown
         when chk_prelim_diagnostic is ticked.
    4. Per-peak strictness options include:
       - 'Use global setting' (inherit)
       - 'Permissive'
       - 'Default'
       - 'Strict'
    5. Selected peak bar shows: t=...s, Amp=..., Origin, Strictness, State, Reason.
    6. Click action mode selector allows selecting peak for strictness without toggling accept/reject.
    """
    import run_v2_review_utility as rv
    html = rv.HTML_TEMPLATE

    assert "function deleteManualPeak(ev)" in html
    assert "function toggleEventState(evObj)" in html
    assert "delete manMap[k]" in html
    assert "delete ppMap[k]" in html
    assert "manually_rejected" in html
    assert 'id="peak_strictness_select"' in html
    assert '<option value="inherit">Use global setting</option>' in html
    assert '<option value="permissive">Permissive</option>' in html
    assert '<option value="default">Default</option>' in html
    assert '<option value="strict">Strict</option>' in html
    assert 'id="peak_select"' in html
    assert "function updatePeakSelector(anns, polKey)" in html
    assert 'id="click_mode_select"' not in html
    assert "Automatically accepted" in html
    assert "Manually added" in html
    assert "Rejected automatic peak" in html
    assert "function getBaseRunQCStatus(segIdx, activePol, runKey)" in html
    assert "connectgaps: false" in html
    assert "timesForCal(seg)" in html
    assert "timesForInt(seg)" in html


# ---------------------------------------------------------------------------
# 34. Focused Final QC Correction: SCRAMBLED ISO Segment 2 signal quality rejection
# ---------------------------------------------------------------------------
def test_scrambled_iso_segment_2_independent_technical_qc_rejection():
    """SCRAMBLED ISO Segment 2 exhibits severe high-frequency tracking bursts and baseline
    distortion. It must evaluate to REJECT (noise_dominated_uninterpretable) in Layer 2 QC,
    and manual polarity overrides (+1 or -1) must NOT rescue it."""
    wb_scrambled = Path(__file__).resolve().parent.parent / "SCRAMBLED ISO individual transients result.xlsx"
    if not wb_scrambled.exists():
        pytest.skip(f"Workbook not found: {wb_scrambled}")

    cfg = BeatCounterConfig()
    df, y, fs, _ = load_cytocypher_excel(str(wb_scrambled), sheet_name="PixelCorrelation Segment 2")
    sig, _, meta = build_concatenated_signal(df, y, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    # Automatic polarity
    res_auto = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta)
    assert res_auto["metadata"]["qc_status"] == "REJECT"

    # Manual polarity override +1
    res_pos = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=1)
    md_pos = res_pos["metadata"]
    assert md_pos["qc_status"] == "REJECT"
    assert md_pos["qc_reason"] == "noise_dominated_uninterpretable"
    assert md_pos["qc_layer2_status"] == "REJECT"
    assert md_pos["n_accepted"] == 0

    # Manual polarity override -1
    res_neg = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta, polarity_override=-1)
    md_neg = res_neg["metadata"]
    assert md_neg["qc_status"] == "REJECT"
    assert md_neg["qc_reason"] == "noise_dominated_uninterpretable"
    assert md_neg["qc_layer2_status"] == "REJECT"
    assert md_neg["n_accepted"] == 0


# ---------------------------------------------------------------------------
# 35. QC Priority: Severe noise takes precedence over undetermined polarity
# ---------------------------------------------------------------------------
def test_qc_severe_noise_takes_precedence_over_unresolved_polarity():
    """When automatic physiological polarity is undetermined, if BOTH orientations (+1 and -1)
    fail signal interpretability QC, the overall decision must be REJECT (noise_dominated_uninterpretable),
    not REJECT (polarity_undetermined) / POLARITY INPUT REQUIRED.

    Verified on DISH1CONTROL Segment 35."""
    wb_dish1 = Path(__file__).resolve().parent.parent / "all data" / "17-12-24.8D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISO individual transients result.xlsx"
    if not wb_dish1.exists():
        pytest.skip(f"Workbook not found: {wb_dish1}")

    cfg = BeatCounterConfig()
    df, y, fs, _ = load_cytocypher_excel(str(wb_dish1), sheet_name="PixelCorrelation Segment 35")
    sig, _, meta = build_concatenated_signal(df, y, fs, cfg, force_invert=False)
    t0 = float(meta.get("time_origin_s", 0.0))

    res_auto = analyze_v2_pipeline(sig, fs, cfg, t0=t0, meta=meta)
    md = res_auto["metadata"]
    assert md["qc_status"] == "REJECT"
    assert md["qc_reason"] == "noise_dominated_uninterpretable"
    assert md["qc_layer2_status"] == "REJECT"
    assert md["n_accepted"] == 0


# ---------------------------------------------------------------------------
# 36. Calcium and Intensity Data Integrity: Un-interpolated fidelity to Excel
# ---------------------------------------------------------------------------
def test_calcium_auxiliary_trace_fidelity_to_excel_without_interpolation():
    """PixelCorrelation and RatioMetricCalcium traces must be loaded directly and faithfully
    from original Excel data without artificial linear interpolation (np.interp).

    In DISH2ICI Segment 67 and Segment 70, calcium acquisition finished ~2 seconds earlier
    than pixel correlation in the original workbook; raw arrays must retain their original lengths
    (1793 vs 2291, 1822 vs 2319) rather than artificially padding with NaNs or warping sample rates."""
    wb_dish2 = Path(__file__).resolve().parent.parent / "all data" / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx"
    if not wb_dish2.exists():
        wb_dish2 = Path(__file__).resolve().parent.parent / "all data" / "13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISO individual transients result.xlsx"
    if not wb_dish2.exists():
        pytest.skip(f"Workbook not found: {wb_dish2}")

    import run_v2_review_utility as rv
    segments_data, _ = rv.process_workbook(str(wb_dish2), segments=[67, 70])

    seg67 = next((s for s in segments_data if s["segment_number"] == 67), None)
    assert seg67 is not None
    assert seg67["length"] == 2291
    assert seg67["length_cal"] == 1793
    assert seg67["t0_cal"] is not None
    assert seg67["fs_cal"] == pytest.approx(seg67["fs"], rel=1e-3)

    seg70 = next((s for s in segments_data if s["segment_number"] == 70), None)
    assert seg70 is not None
    assert seg70["length"] == 2320
    assert seg70["length_cal"] == 1822


# ---------------------------------------------------------------------------
# 37. HTML Review Report Generation Integrity and Client Initialization
# ---------------------------------------------------------------------------
def test_html_review_report_input_validation(tmp_path):
    """generate_html must reject empty segments_data with a clear ValueError."""
    from run_v2_review_utility import generate_html

    out_html = tmp_path / "empty_report.html"
    with pytest.raises(ValueError, match="segments_data is empty"):
        generate_html([], "test.xlsx", str(out_html))


def test_html_review_report_client_initialization(tmp_path):
    """Generated HTML report must successfully initialize in browser/DOM environment without ReferenceError."""
    import json
    import shutil
    import subprocess
    from run_v2_review_utility import HTML_TEMPLATE, generate_html

    # Static assertion: polInfo must be defined in populateSegmentSelector before getActiveQC
    assert "const polInfo = getActivePolarity(idx);" in HTML_TEMPLATE
    assert "client_error_banner" in HTML_TEMPLATE
    assert "window.addEventListener('error'" in HTML_TEMPLATE

    # Generate a report from mock data
    out_html = tmp_path / "test_init.html"
    mock_seg = {
        "sheet_name": "PixelCorrelation Segment 1",
        "segment_number": 1,
        "sample_id": "TEST_1",
        "fs": 250.0,
        "t0": 0.0,
        "length": 100,
        "qc_status": "PASS",
        "qc_reason": "clean_signal",
        "raw_polarity": 1,
        "polarity_source": "automatic",
        "polarity_method": "baseline_noise_ratio",
        "polarity_reason": "test",
        "auto_polarity": 1,
        "auto_method": "test",
        "auto_reason": "test",
        "effective_polarity": 1,
        "has_amplitude": False,
        "sig_raw_b64": "",
        "baseline_raw_b64": None,
        "amplitude_b64": None,
        "sig_int_b64": None,
        "sig_cal_b64": None,
        "runs": {
            "default": {
                "params": {},
                "separation_strictness": 1.0,
                "amplitude_strictness": 1.0,
                "qc_status": "PASS",
                "qc_reason": "clean_signal",
                "analyzable_duration_s": 0.4,
                "analyzable_fraction": 1.0,
                "excluded_intervals_s": [],
                "contraction_frequency_hz": 0.0,
                "reference_amplitude": 0.5,
                "amplitude_threshold": 0.25,
                "noise_sigma": 0.01,
                "events": {"index": [], "time_s": [], "raw_value": [], "status": [], "reason": []},
            }
        },
    }
    generate_html([mock_seg], "test_workbook.xlsx", str(out_html))
    assert out_html.exists()

    # If node is available, execute headless DOM initialization test
    node_bin = shutil.which("node")
    if node_bin:
        runner_script = Path(__file__).resolve().parent.parent / "scratch_test_runner.js"
        if runner_script.exists():
            res = subprocess.run([node_bin, str(runner_script), str(out_html)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert res.returncode == 0, res.stderr
            data = json.loads(res.stdout.strip())
            assert data["runtimeError"] is None
            assert data["clientErrorDisplay"] == "none"
            assert data["sourceName"] == "test_workbook.xlsx"
            assert data["segmentCount"] == 1
            assert data["polarityCount"] == 3








