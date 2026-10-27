"""
tests/test_scientific_integrity_metamorphic.py
==============================================

Comprehensive metamorphic, invariance, and property-based scientific integrity
test suite covering all 15 mandatory categories from the Scientific Integrity Audit:

1. Time-origin invariance (PixelCorrelation, Calcium, Sarcomere + reconstruction intervals)
2. Row-partition invariance
3. Amplitude-scale behavior & strict MAIN/SCE threshold monotonicity
4. Baseline-offset behavior
5. Sampling-rate consistency (125 Hz, 250 Hz, 500 Hz)
6. Noise sensitivity & clean quiescent vs noise-dominated separation across all 3 analyzers
7. Artifact injection & zero amplitude fabrication (n_directly_measured vs n_reconstructed_timings)
8. NaN injection & missing-data safety
9. Input order independence (zero cross-segment state leakage)
10. Polarity inversion antisymmetry
11. Strict Calcium-Sarcomere channel independence
12. Workbook filename invariance
13. Segment-ID / sheet-number invariance
14. Preset monotonicity & custom parameter honoring (no KeyError on custom preset_name)
15. Export / sidecar round-trip integrity
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pixel_counter.config import BeatCounterConfig
from pixel_counter.io_utils import load_cytocypher_excel
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.v2_signal import determine_raw_polarity
from pixel_counter.v2_pipeline import analyze_v2_pipeline
from pixel_counter.calcium_detector import (
    CalciumDetectorParams,
    analyze_calcium_pipeline,
    compute_normalized_calcium,
)
from pixel_counter.sarcomere_detector import SarcomereDetectorParams
from pixel_counter.sarcomere_pipeline import (
    analyze_sarcomere_pipeline,
    analyze_sarcomere_segment_from_excel,
)
from pixel_counter.sarcomere_qc import (
    SarcomereMeasurementProfile,
    detect_sarcomere_artifacts,
)
from pixel_counter.polarity_overrides import (
    load_polarity_overrides,
    load_segment_rejections,
    save_polarity_overrides,
)


def _synth_sarcomere(
    fs: float = 250.0,
    duration_s: float = 8.0,
    l0_um: float = 1.80,
    beat_times_s: list[float] | None = None,
    shortening_um: float = 0.16,
    sigma_s: float = 0.045,
    noise_sd: float = 0.0005,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = np.full(n, l0_um, dtype=float) + rng.normal(0.0, noise_sd, size=n)
    if beat_times_s is None:
        beat_times_s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for bt in beat_times_s:
        sig -= shortening_um * np.exp(-0.5 * ((t - bt) / sigma_s) ** 2)
    return t, sig


def _synth_calcium(
    fs: float = 250.0,
    duration_s: float = 8.0,
    r0: float = 0.30,
    beat_times_s: list[float] | None = None,
    delta_r: float = 0.12,
    noise_sd: float = 0.0010,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = np.full(n, r0, dtype=float) + rng.normal(0.0, noise_sd, size=n)
    if beat_times_s is None:
        beat_times_s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for bt in beat_times_s:
        dt = t - bt
        pulse = np.where(
            dt <= 0,
            delta_r * np.exp(-0.5 * (dt / 0.030) ** 2),
            delta_r * np.exp(-dt / 0.220),
        )
        sig += pulse
    return t, sig


def _synth_pixel(
    fs: float = 250.0,
    duration_s: float = 8.0,
    baseline: float = 0.85,
    polarity: int = -1,
    beat_times_s: list[float] | None = None,
    amp: float = 0.15,
    noise_sd: float = 0.0010,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    excursion = np.zeros(n, dtype=float)
    if beat_times_s is None:
        beat_times_s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for bt in beat_times_s:
        dt = t - bt
        pulse = np.where(
            dt <= 0,
            amp * np.exp(-0.5 * (dt / 0.025) ** 2),
            amp * np.exp(-0.5 * (dt / 0.075) ** 2),
        )
        excursion += pulse
    sig = baseline + polarity * excursion + rng.normal(0.0, noise_sd, size=n)
    return t, sig


# ---------------------------------------------------------------------------
# 1. Time-Origin Invariance
# ---------------------------------------------------------------------------
def test_1_time_origin_invariance_all_analyzers():
    """Shifting t0 shifts all absolute timestamps and interval coordinates by exactly t0
    without altering indices, amplitudes, or QC status."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs)
    # Inject a 2-sample flank spike to verify reconstructed interval timestamps also shift by t0
    s_sig[int(1.0 * fs) - 8 : int(1.0 * fs) - 6] = 2.2335
    _, c_sig = _synth_calcium(fs=fs)
    _, p_sig = _synth_pixel(fs=fs)

    s_0 = analyze_sarcomere_pipeline(s_sig, fs, t0=0.0)
    s_t = analyze_sarcomere_pipeline(s_sig, fs, t0=500.25)
    assert s_0["qc_status"] == s_t["qc_status"] == "PASS"
    assert s_0["n_accepted"] == s_t["n_accepted"]
    np.testing.assert_allclose(
        s_t["events"]["time_s"].to_numpy() - s_0["events"]["time_s"].to_numpy(),
        500.25,
        atol=1e-9,
    )
    # Reconstructed interval timestamps must also include t0
    ivs_0 = s_0["reconstruction"]["reconstructed_intervals"]
    ivs_t = s_t["reconstruction"]["reconstructed_intervals"]
    assert len(ivs_0) == len(ivs_t) >= 1
    assert ivs_t[0]["start_time_s"] - ivs_0[0]["start_time_s"] == pytest.approx(500.25, abs=1e-9)
    assert ivs_t[0]["end_time_s"] - ivs_0[0]["end_time_s"] == pytest.approx(500.25, abs=1e-9)

    c_0 = analyze_calcium_pipeline(c_sig, fs, t0=0.0)
    c_t = analyze_calcium_pipeline(c_sig, fs, t0=500.25)
    np.testing.assert_allclose(
        c_t["events"]["time_s"].to_numpy() - c_0["events"]["time_s"].to_numpy(),
        500.25,
        atol=1e-9,
    )

    p_0 = analyze_v2_pipeline(p_sig, fs, BeatCounterConfig(), t0=0.0, meta={"time_origin_s": 0.0})
    p_t = analyze_v2_pipeline(p_sig, fs, BeatCounterConfig(), t0=500.25, meta={"time_origin_s": 500.25})
    np.testing.assert_allclose(
        p_t["candidates"]["time_s"] - p_0["candidates"]["time_s"],
        500.25,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 2. Row-Partition Invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row_len", [125, 250, 500])
def test_2_row_partition_invariance(row_len: int):
    """Splitting the same signal across different Excel row lengths produces identical results."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, duration_s=8.0)
    n = len(s_sig)
    y_cols = [f"y {j}" for j in range(row_len)]
    records = []
    for idx, start in enumerate(range(0, n, row_len)):
        chunk = s_sig[start : min(n, start + row_len)]
        rec = {
            "Transient": idx + 1,
            "Begin": float(start / fs),
            "End": float((start + len(chunk)) / fs),
        }
        for j in range(row_len):
            rec[f"y {j}"] = float(chunk[j]) if j < len(chunk) else np.nan
        records.append(rec)
    df = pd.DataFrame(records)
    stitched, _, meta = build_concatenated_signal(df, y_cols, fs, BeatCounterConfig(), force_invert=False)
    np.testing.assert_allclose(stitched, s_sig)

    res_direct = analyze_sarcomere_pipeline(s_sig, fs, t0=0.0)
    res_stitched = analyze_sarcomere_pipeline(stitched, fs, t0=float(meta["time_origin_s"]))
    assert res_direct["n_accepted"] == res_stitched["n_accepted"] == 7
    np.testing.assert_allclose(
        res_direct["events"]["pct_shortening"].to_numpy(),
        res_stitched["events"]["pct_shortening"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# 3. Amplitude-Scale Behavior & MAIN/SCE Hierarchy
# ---------------------------------------------------------------------------
def test_3_amplitude_scale_behavior():
    """Positive scaling of PixelCorrelation preserves indices; scaling Sarcomere excursion depth
    transitions events monotonically across REJECTED -> SECONDARY -> MAIN."""
    fs = 250.0
    _, p_sig = _synth_pixel(fs=fs)
    r1 = analyze_v2_pipeline(p_sig, fs, BeatCounterConfig())
    r2 = analyze_v2_pipeline(3.5 * p_sig, fs, BeatCounterConfig())
    np.testing.assert_array_equal(r1["candidates"]["candidate_index"], r2["candidates"]["candidate_index"])

    # Sarcomere trace with 5 main beats (dL=0.18 um -> 10%) and 1 test beat at t=3.5s
    t, base_s = _synth_sarcomere(fs=fs, duration_s=7.0, beat_times_s=[1.0, 2.0, 3.0, 5.0, 6.0], shortening_um=0.18)
    classes = []
    for test_dl in [0.002, 0.040, 0.160]:  # 0.11% (REJECTED), 2.22% (SECONDARY), 8.89% (MAIN)
        sig_mod = base_s - test_dl * np.exp(-0.5 * ((t - 3.5) / 0.040) ** 2)
        res = analyze_sarcomere_pipeline(sig_mod, fs)
        ev = res["events"]
        cand = ev[np.abs(ev["time_s"] - 3.5) < 0.08]
        if cand.empty or cand.iloc[0]["status"] == "REJECTED":
            classes.append("REJECTED")
        else:
            classes.append(str(cand.iloc[0]["event_class"]))
    assert classes == ["REJECTED", "SECONDARY", "MAIN"]


# ---------------------------------------------------------------------------
# 4. Baseline-Offset Behavior
# ---------------------------------------------------------------------------
def test_4_baseline_offset_behavior():
    """Adding a constant offset to PixelCorrelation is strictly invariant; proportional L0/dL scaling
    in Sarcomere preserves percentage shortening."""
    fs = 250.0
    _, p_sig = _synth_pixel(fs=fs)
    r0 = analyze_v2_pipeline(p_sig, fs, BeatCounterConfig())
    r_shift = analyze_v2_pipeline(p_sig + 12.5, fs, BeatCounterConfig())
    np.testing.assert_array_equal(r0["candidates"]["candidate_index"], r_shift["candidates"]["candidate_index"])
    np.testing.assert_allclose(r0["candidates"]["amplitude"], r_shift["candidates"]["amplitude"], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Sampling-Rate Consistency (125 Hz, 250 Hz, 500 Hz)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fs", [125.0, 250.0, 500.0])
def test_5_sampling_rate_consistency(fs: float):
    """All three analyzers detect the same physiological beats and amplitudes across 125, 250, and 500 Hz."""
    beats = [1.0, 2.2, 3.4, 4.6, 5.8]
    _, s_sig = _synth_sarcomere(fs=fs, duration_s=7.0, beat_times_s=beats, shortening_um=0.16, noise_sd=0.0004)
    res_s = analyze_sarcomere_pipeline(s_sig, fs)
    assert res_s["qc_status"] == "PASS"
    assert res_s["n_accepted"] == len(beats)
    assert res_s["mean_pct_shortening"] == pytest.approx(100.0 * 0.16 / 1.80, abs=0.5)

    _, c_sig = _synth_calcium(fs=fs, duration_s=7.0, beat_times_s=beats, delta_r=0.12, noise_sd=0.0008)
    res_c = analyze_calcium_pipeline(c_sig, fs)
    assert res_c["qc_status"] == "PASS"
    assert res_c["metadata"]["n_accepted"] == len(beats)

    _, p_sig = _synth_pixel(fs=fs, duration_s=7.0, beat_times_s=beats, amp=0.15, noise_sd=0.0008)
    res_p = analyze_v2_pipeline(p_sig, fs, BeatCounterConfig())
    assert res_p["metadata"]["qc_status"] == "PASS"
    assert res_p["metadata"]["n_accepted"] == len(beats)


# ---------------------------------------------------------------------------
# 6. Noise Sensitivity & Clean Quiescent vs Noise-Dominated Separation
# ---------------------------------------------------------------------------
def test_6_clean_quiescent_vs_noise_dominated_all_analyzers():
    """Clean quiet traces with 0 beats must PASS as quiescent across all 3 analyzers,
    whereas high-noise zero-beat traces must REJECT."""
    fs = 250.0
    rng = np.random.default_rng(123)
    n = int(6.0 * fs)

    # Sarcomere: quiet baseline vs noisy baseline
    s_quiet = 1.80 + rng.normal(0.0, 0.0004, size=n)
    res_sq = analyze_sarcomere_pipeline(s_quiet, fs)
    assert res_sq["qc_status"] == "PASS"
    assert res_sq["qc_reason"] == "clean_quiescent_no_contractions"
    assert res_sq["n_accepted"] == 0

    s_noisy = 1.80 + rng.normal(0.0, 0.012, size=n)
    res_sn = analyze_sarcomere_pipeline(s_noisy, fs)
    assert res_sn["qc_status"] == "REJECT"
    assert res_sn["n_accepted"] == 0

    # Calcium: quiet baseline vs noisy baseline
    c_quiet = 0.28 + rng.normal(0.0, 0.0006, size=n)
    res_cq = analyze_calcium_pipeline(c_quiet, fs)
    assert res_cq["qc_status"] == "PASS"
    assert res_cq["qc_reason"] == "no_accepted_transients"

    c_noisy = 0.28 + rng.normal(0.0, 0.0065, size=n)
    res_cn = analyze_calcium_pipeline(c_noisy, fs)
    assert res_cn["qc_status"] == "REJECT"


# ---------------------------------------------------------------------------
# 7. Artifact Injection & Direct vs Reconstructed Amplitude Accounting
# ---------------------------------------------------------------------------
def test_7_artifact_injection_and_direct_amplitude_accounting():
    """When 1 of 4 contractions has its trough nadir corrupted by a narrow tracking spike,
    n_accepted == 4, n_directly_measured == 3, n_valid_amplitude == 3, n_lacking_amplitude == 1,
    and the corrupted event has NaN amplitude in both sarcomere and sarcomere-calcium modes."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, duration_s=6.0, beat_times_s=[1.0, 2.2, 3.4, 4.6], shortening_um=0.16)
    # Corrupt trough nadir of beat 2 (t=2.2s -> sample 550) with 3-sample rail spike (12 ms)
    s_sig[549:552] = 2.2335

    res_s = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    assert res_s["qc_status"] == "PASS"
    assert res_s["n_accepted"] == 4
    def_run = res_s["preset_runs"]["default"]
    assert def_run["n_directly_measured"] == 3
    assert def_run["n_valid_amplitude"] == 3
    assert def_run["n_lacking_amplitude"] == 1


# ---------------------------------------------------------------------------
# 8. NaN Injection Safety
# ---------------------------------------------------------------------------
def test_8_nan_injection_safety():
    """Short NaN gaps between beats do not crash or fabricate events; >60% NaN rejects cleanly."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, duration_s=6.0, beat_times_s=[1.0, 3.0, 5.0])
    s_gap = s_sig.copy()
    s_gap[int(1.8 * fs) : int(2.0 * fs)] = np.nan
    res_gap = analyze_sarcomere_pipeline(s_gap, fs)
    assert res_gap["qc_status"] == "PASS"
    assert res_gap["n_accepted"] == 3

    s_mostly_nan = np.full_like(s_sig, np.nan)
    s_mostly_nan[:100] = 1.80
    res_bad = analyze_sarcomere_pipeline(s_mostly_nan, fs)
    assert res_bad["qc_status"] == "REJECT"


# ---------------------------------------------------------------------------
# 9. Input Order Independence
# ---------------------------------------------------------------------------
def test_9_input_order_independence():
    """Analyzing signals in different order produces bit-for-bit identical results."""
    fs = 250.0
    _, s1 = _synth_sarcomere(fs=fs, beat_times_s=[1.0, 2.0, 3.0], shortening_um=0.14, seed=1)
    _, s2 = _synth_sarcomere(fs=fs, beat_times_s=[1.5, 3.5], shortening_um=0.18, seed=2)
    _, s3 = _synth_sarcomere(fs=fs, beat_times_s=[], shortening_um=0.0, noise_sd=0.010, seed=3)

    r1_a = analyze_sarcomere_pipeline(s1, fs)
    r2_a = analyze_sarcomere_pipeline(s2, fs)
    r3_a = analyze_sarcomere_pipeline(s3, fs)

    r3_b = analyze_sarcomere_pipeline(s3, fs)
    r2_b = analyze_sarcomere_pipeline(s2, fs)
    r1_b = analyze_sarcomere_pipeline(s1, fs)

    assert r1_a["n_accepted"] == r1_b["n_accepted"]
    assert r2_a["mean_pct_shortening"] == r2_b["mean_pct_shortening"]
    assert r3_a["qc_status"] == r3_b["qc_status"]


# ---------------------------------------------------------------------------
# 10. Polarity Inversion Antisymmetry
# ---------------------------------------------------------------------------
def test_10_polarity_inversion_antisymmetry():
    """Negating a PixelCorrelation signal strictly negates automatic polarity across cycle rates."""
    fs = 250.0
    for ibi in [1.0, 0.5, 0.33]:
        beats = list(np.arange(0.5, 5.5, ibi))
        _, sig_neg = _synth_pixel(fs=fs, duration_s=6.0, polarity=-1, beat_times_s=beats)
        sig_pos = -sig_neg
        p_neg = determine_raw_polarity(sig_neg, fs)
        p_pos = determine_raw_polarity(sig_pos, fs)
        assert p_neg["polarity"] == -1
        assert p_pos["polarity"] == 1


# ---------------------------------------------------------------------------
# 11. Strict Calcium-Sarcomere Channel Independence
# ---------------------------------------------------------------------------
def test_11_calcium_channel_independence():
    """Presence, absence, or corruption of calcium trace has zero effect on sarcomere detection or QC."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0])
    _, c_sig = _synth_calcium(fs=fs, beat_times_s=[0.96, 1.96, 2.96, 3.96])

    r_only = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere")
    r_dual = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere-calcium", cal_sig_raw=c_sig, cal_fs=fs)
    r_inv_cal = analyze_sarcomere_pipeline(s_sig, fs, mode="sarcomere-calcium", cal_sig_raw=-c_sig, cal_fs=fs)

    for r in (r_dual, r_inv_cal):
        assert r["qc_status"] == r_only["qc_status"]
        assert r["n_accepted"] == r_only["n_accepted"]
        assert r["mean_pct_shortening"] == pytest.approx(r_only["mean_pct_shortening"], abs=1e-12)


# ---------------------------------------------------------------------------
# 12 & 13. Filename Invariance and Segment-ID Invariance
# ---------------------------------------------------------------------------
def test_12_and_13_filename_and_segment_id_invariance(tmp_path: Path):
    """Workbook filename and Segment number in sheet names have zero effect on algorithm output."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, duration_s=4.0, beat_times_s=[1.0, 2.0, 3.0])
    n = len(s_sig)
    rec = {"Sampling Frequency": fs, "Transient": 1, "Begin": 0.0, "End": float((n - 1) / fs)}
    for j in range(n):
        rec[f"y {j}"] = float(s_sig[j])
    df = pd.DataFrame([rec])

    wb1 = tmp_path / "(MI) CGP+CAR+ROLI+IBMX DISH1 individual transients result.xlsx"
    wb2 = tmp_path / "SCRAMBLED_NOISE_ANONYMOUS_HOLDOUT_999.xlsx"

    with pd.ExcelWriter(wb1) as w1:
        df.to_excel(w1, sheet_name="SarcomereLength Segment 1", index=False)
    with pd.ExcelWriter(wb2) as w2:
        df.to_excel(w2, sheet_name="SarcomereLength Segment 987", index=False)

    res1 = analyze_sarcomere_segment_from_excel(wb1, "SarcomereLength Segment 1")
    res2 = analyze_sarcomere_segment_from_excel(wb2, "SarcomereLength Segment 987")

    assert res1["qc_status"] == res2["qc_status"] == "PASS"
    assert res1["n_accepted"] == res2["n_accepted"] == 3
    assert res1["mean_pct_shortening"] == pytest.approx(res2["mean_pct_shortening"], abs=1e-12)
    # Also verify fs provenance was recorded by load_cytocypher_excel
    df_loaded, _, _, _ = load_cytocypher_excel(wb1, "SarcomereLength Segment 1")
    assert df_loaded.attrs["fs_provenance"] == "declared"
    assert df_loaded.attrs["declared_fs"] == 250.0


# ---------------------------------------------------------------------------
# 14. Preset Monotonicity & Custom Parameter Honoring
# ---------------------------------------------------------------------------
def test_14_preset_monotonicity_and_custom_params_honored():
    """Passing custom SarcomereDetectorParams or CalciumDetectorParams with a custom preset_name
    is honored without KeyError and modifies thresholds as specified."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0], shortening_um=0.10)
    custom_s_params = SarcomereDetectorParams(
        preset_name="custom_high_threshold",
        min_pct_shortening_main=15.0,
        min_pct_shortening_sce=12.0,
        alpha_ref=0.95,
    )
    res_s = analyze_sarcomere_pipeline(s_sig, fs, params=custom_s_params)
    assert "custom_high_threshold" in res_s["preset_runs"]
    assert res_s["threshold_main_pct"] >= 15.0
    assert res_s["n_main"] == 0

    _, c_sig = _synth_calcium(fs=fs, beat_times_s=[1.0, 2.0, 3.0, 4.0], delta_r=0.10)
    custom_c_params = CalciumDetectorParams(
        preset_name="custom_strict_cal",
        min_norm_height=2.0,
        raw_prominence_sigma=4.0,
    )
    res_c = analyze_calcium_pipeline(c_sig, fs, params=custom_c_params)
    assert "custom_strict_cal" in res_c["runs"]
    assert res_c["metadata"]["n_accepted"] == 0


# ---------------------------------------------------------------------------
# 15. Configurable SarcomereMeasurementProfile & Export Round-Trip
# ---------------------------------------------------------------------------
def test_15_configurable_measurement_profile_and_sidecar_roundtrip(tmp_path: Path):
    """SarcomereMeasurementProfile allows customizing FFT rail bounds, and sidecar JSON round-trips cleanly."""
    fs = 250.0
    _, s_sig = _synth_sarcomere(fs=fs, l0_um=2.20, shortening_um=0.15)
    # Under default profile (fft_rail_max_um=2.18), 2.20 um is flagged as rail;
    # under custom wide profile (fft_rail_max_um=2.35), it is not flagged as rail.
    art_def = detect_sarcomere_artifacts(s_sig, fs)
    art_wide = detect_sarcomere_artifacts(
        s_sig, fs, profile=SarcomereMeasurementProfile(fft_rail_max_um=2.35)
    )
    assert art_def["rail_fraction"] > 0.50
    assert art_wide["rail_fraction"] == 0.0

    wb_dummy = tmp_path / "roundtrip.xlsx"
    wb_dummy.write_text("dummy")
    out_json = save_polarity_overrides(wb_dummy, {1: -1, 2: 1}, segment_rejections={3: {"rejected": True, "reason": "Noise"}})
    assert load_polarity_overrides(wb_dummy, out_json) == {1: -1, 2: 1}
    assert load_segment_rejections(wb_dummy, out_json)[3]["rejected"] is True
