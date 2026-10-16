import inspect
import os
from pathlib import Path

import numpy as np
import pytest

import pixel_counter.qc_v2 as qc_v2
from pixel_counter.config import BeatCounterConfig
from pixel_counter.v2_pipeline import analyze_v2_pipeline
from pixel_counter.v2_signal import determine_raw_polarity
from tests.v2_synth import FS, make_amplitude, to_raw

CENTERS = [0.8, 1.8, 2.8, 3.8, 4.8]


# ---------------------------------------------------------------- polarity ----------------
@pytest.mark.parametrize("sigma", [0.03, 0.06])
def test_polarity_both_orientations(sigma):
    t, a = make_amplitude(CENTERS, sigma=sigma)
    up = determine_raw_polarity(to_raw(a, 1), FS)
    down = determine_raw_polarity(to_raw(a, -1), FS)
    assert up["polarity"] == 1 and down["polarity"] == -1


def test_polarity_invariance_to_shift_scale_and_start_phase():
    t, a = make_amplitude(CENTERS + [5.8], dur=7.0)
    raw = to_raw(a, 1)
    for variant in (raw + 4.0, 5.0 * raw, raw[int(0.4 * FS):], raw[int(0.9 * FS):], raw[: -int(0.6 * FS)]):
        assert determine_raw_polarity(variant, FS)["polarity"] == 1
        assert determine_raw_polarity(-variant, FS)["polarity"] == -1


def test_polarity_does_not_modify_input():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, -1)
    copy = raw.copy()
    determine_raw_polarity(raw, FS)
    np.testing.assert_array_equal(raw, copy)


@pytest.mark.parametrize("sig", [np.full(1500, 0.5), np.array([]), np.random.default_rng(5).normal(0, 1e-3, 1500)])
def test_polarity_undetermined_instead_of_invented(sig):
    res = determine_raw_polarity(sig, FS)
    assert res["polarity"] == 0 and res["status"] == "UNDETERMINED"


def test_undetermined_polarity_is_a_segment_failure_without_events():
    rng = np.random.default_rng(7)
    res = analyze_v2_pipeline(0.5 + rng.normal(0, 1e-3, 1500), FS, BeatCounterConfig())
    assert res["metadata"]["qc_status"] == "REJECT"
    assert res["metadata"]["qc_reason"] == "polarity_undetermined"
    assert len(res["candidates"]["time_s"]) == 0


# ---------------------------------------------------------------- technical QC --------------
def test_qc_invalid_input():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, 1)
    assert analyze_v2_pipeline(raw, -1.0, BeatCounterConfig())["metadata"]["qc_status"] == "ERROR"
    assert analyze_v2_pipeline(raw, float("nan"), BeatCounterConfig())["metadata"]["qc_status"] == "ERROR"
    assert analyze_v2_pipeline(np.array([]), FS, BeatCounterConfig())["metadata"]["qc_reason"] == "empty_signal"
    res = analyze_v2_pipeline(raw, FS, BeatCounterConfig(), t0=float("nan"), meta={"time_origin_s": float("nan")})
    assert res["metadata"]["qc_status"] == "ERROR"


def test_qc_missing_and_insufficient_data():
    assert analyze_v2_pipeline(np.full(2000, np.nan), FS, BeatCounterConfig())["metadata"]["qc_reason"] == "no_valid_data"
    sig = np.full(2000, np.nan)
    sig[:100] = 0.5
    assert analyze_v2_pipeline(sig, FS, BeatCounterConfig())["metadata"]["qc_reason"] == "insufficient_valid_data"


def test_qc_overlap_conflict_and_low_fs():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, 1)
    res = analyze_v2_pipeline(raw, FS, BeatCounterConfig(), meta={"stitch_overlap_conflicts": 3})
    assert res["metadata"]["qc_status"] == "REJECT" and res["metadata"]["qc_reason"] == "overlap_conflict"
    res = analyze_v2_pipeline(raw[::50], 5.0, BeatCounterConfig())
    assert res["metadata"]["qc_status"] == "REJECT"


def test_qc_clipping_is_a_flag_not_an_uncertainty_class():
    t, a = make_amplitude(CENTERS)
    raw = np.minimum(to_raw(a, 1), 0.55)               # heavy clipping of the peaks
    res = analyze_v2_pipeline(raw, FS, BeatCounterConfig())
    assert "signal_clipping" in res["metadata"]["qc_flags"]
    assert res["metadata"]["qc_status"] in ("PASS", "REJECT")   # no REVIEW class exists any more


# ---------------------------------------------------------------- removal of the +-15 % mechanism
def test_sensitivity_analysis_removed():
    src = inspect.getsource(qc_v2)
    assert "prom0" not in src and "detect_events_v2" not in src      # no repeated detection runs
    assert "1.15" not in src                                          # (0.85 remains: unrelated spike-fraction QC limit)
    assert "sensitivity" not in src.lower() and "stability" not in src.lower()
    cfg = BeatCounterConfig()
    assert not hasattr(cfg, "v2_two_pass_enabled") and not hasattr(cfg, "v2_seed_prom_mad_mult")
    t, a = make_amplitude(CENTERS)
    meta = analyze_v2_pipeline(to_raw(a, 1), FS, cfg)["metadata"]
    assert not any(k.startswith("stability_") for k in meta)
    assert meta["qc_status"] == "PASS" and meta["qc_reason"] != "sensitivity_instability"


def test_prom0_does_not_influence_v2():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, 1)
    base = analyze_v2_pipeline(raw, FS, BeatCounterConfig(prom0=0.002))
    other = analyze_v2_pipeline(raw, FS, BeatCounterConfig(prom0=0.3))
    np.testing.assert_array_equal(base["candidates"]["candidate_index"], other["candidates"]["candidate_index"])


def test_polarity_inversion_preserves_event_timestamps():
    t, a = make_amplitude(CENTERS)
    raw_neg = to_raw(a, -1)
    raw_pos = to_raw(a, 1)
    res_neg = analyze_v2_pipeline(raw_neg, FS, BeatCounterConfig())
    res_pos = analyze_v2_pipeline(raw_pos, FS, BeatCounterConfig())
    assert res_neg["metadata"]["raw_polarity"] == -1
    assert res_pos["metadata"]["raw_polarity"] == 1
    np.testing.assert_allclose(res_neg["candidates"]["time_s"], res_pos["candidates"]["time_s"], atol=1e-4)


def test_polarity_symmetric_cycle_handling():
    # Pure symmetric sinusoid (dwell=0, onset=0, curv=0)
    t = np.arange(int(5 * FS)) / FS
    sig = 0.5 + 0.3 * np.sin(2 * np.pi * 1.5 * t)
    res = determine_raw_polarity(sig, FS)
    assert res["status"] in ("OK", "UNDETERMINED")
    # If inverted, must be perfectly antisymmetric
    res_inv = determine_raw_polarity(-sig, FS)
    assert res["polarity"] == -res_inv["polarity"]


def test_polarity_asymmetric_duration():
    # Asymmetric physiological pulse: abrupt active contraction (40 ms) followed by slower relaxation decay (200 ms)
    t = np.arange(int(5 * FS)) / FS
    centers = [0.8, 1.8, 2.8, 3.8]
    a = np.zeros_like(t)
    for c in centers:
        dt = t - c
        mask_r = (dt >= -0.04) & (dt < 0)
        a[mask_r] += np.exp(-0.5 * (dt[mask_r] / 0.02) ** 2)
        mask_d = (dt >= 0) & (dt < 0.40)
        a[mask_d] += np.exp(-dt[mask_d] / 0.15)
    
    # Upward contractions (polarity +1): rest at 0.2, excursions go up
    res_rise = determine_raw_polarity(0.2 + 0.6 * a, FS)
    # Downward contractions (polarity -1): rest at 0.9, excursions go down
    res_fall = determine_raw_polarity(0.9 - 0.6 * a, FS)
    assert res_fall["polarity"] == -1
    assert res_rise["polarity"] == 1


def test_polarity_noise_difference_not_sole_determinant():
    # Signal with negative polarity, but noise is added at the TOP (resting state)
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, -1)
    rng = np.random.default_rng(42)
    # Add noise specifically when signal is high (resting state)
    top_mask = raw > np.median(raw)
    noisy_top = raw.copy()
    noisy_top[top_mask] += rng.normal(0, 0.03, int(np.sum(top_mask)))
    res = determine_raw_polarity(noisy_top, FS)
    assert res["polarity"] == -1, "High noise on resting state must not invert polarity"


# ---------------------------------------------------------------- development workbook ----------
DEV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "25092026_IMR90_1,2,3HZ individual transients result.xlsx")
NEGATIVE_REFERENCE = [1, 2, 3, 4, 5, 12, 19, 20, 24, 26, 27, 29, 34, 38]
POSITIVE_REFERENCE = [30, 35, 39, 40, 41, 44]
HIGH_FREQ_AMBIGUOUS = [33, 36]
CHALLENGE_SEGMENTS = [16, 17, 18, 42, 45]


@pytest.fixture(scope="module")
def dev_segments():
    if not os.path.exists(DEV):
        pytest.skip("development workbook not available")
    from pixel_counter.io_utils import load_cytocypher_excel
    from pixel_counter.preprocessing import build_concatenated_signal
    cfg = BeatCounterConfig()
    out = {}
    for seg in set(NEGATIVE_REFERENCE + POSITIVE_REFERENCE + HIGH_FREQ_AMBIGUOUS + CHALLENGE_SEGMENTS):
        df, ycols, fs, _ = load_cytocypher_excel(DEV, sheet_name=f"PixelCorrelation Segment {seg}")
        sig, _, meta = build_concatenated_signal(df, ycols, fs, cfg, force_invert=False)
        out[seg] = (sig, fs, meta)
    return out


def test_dev_raw_polarity_references(dev_segments):
    for seg in NEGATIVE_REFERENCE:
        assert determine_raw_polarity(dev_segments[seg][0], dev_segments[seg][1])["polarity"] == -1, f"Seg {seg} failed"
    for seg in POSITIVE_REFERENCE:
        assert determine_raw_polarity(dev_segments[seg][0], dev_segments[seg][1])["polarity"] == 1, f"Seg {seg} failed"


def test_dev_high_frequency_safe_abstention(dev_segments):
    """Verify that high-frequency ambiguous segments abstain safely rather than making false decisions."""
    for seg in HIGH_FREQ_AMBIGUOUS:
        pol = determine_raw_polarity(dev_segments[seg][0], dev_segments[seg][1])["polarity"]
        assert pol == 0, f"Seg {seg} should safely abstain as UNDETERMINED in pixel-only mode, got {pol}"


def test_dev_positive_control_and_timestamps(dev_segments):
    sig, fs, meta = dev_segments[5]
    res = analyze_v2_pipeline(sig, fs, BeatCounterConfig(), t0=meta["time_origin_s"], meta=meta)
    assert res["metadata"]["n_accepted"] == 4          # four complete contractions in segment 5
    sig, fs, meta = dev_segments[2]
    res = analyze_v2_pipeline(sig, fs, BeatCounterConfig(), t0=meta["time_origin_s"], meta=meta)
    t = res["candidates"]["time_s"]
    ref = np.array([44.75, 45.8, 46.75, 47.8, 48.8])     # approximate researcher references
    assert len(t) == 5
    assert np.all(np.abs(t - ref) < 0.15)


def test_dev_automatic_polarity_detects_segment3(dev_segments):
    sig, fs, meta = dev_segments[3]
    # Now automatically detected without manual override!
    res = analyze_v2_pipeline(sig, fs, BeatCounterConfig(), t0=meta["time_origin_s"], meta=meta)
    assert res["metadata"]["raw_polarity"] == -1
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["metadata"]["n_accepted"] == 4
    t = res["candidates"]["time_s"]
    assert np.all(np.abs(t - np.array([68.6, 69.6, 70.6, 71.6])) < 0.15)


def test_dev_automatic_polarity_detects_segment16(dev_segments):
    sig, fs, meta = dev_segments[16]
    # Segment 16 now automatically resolves to -1 via onset and curvature consensus
    res = analyze_v2_pipeline(sig, fs, BeatCounterConfig(), t0=meta["time_origin_s"], meta=meta)
    assert res["metadata"]["raw_polarity"] == -1
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["metadata"]["n_accepted"] >= 8
    t = res["candidates"]["time_s"]
    # Researcher anchor: ~506.556 s
    assert np.any(np.abs(t - 506.556) < 0.10)


def test_dev_conflicting_segments_fail_safe_undetermined(dev_segments):
    # Segments 17, 18, 42 have severe conflict between dwell and kinetics.
    # The algorithm must refuse to guess and return UNDETERMINED rather than an incorrect inversion.
    for seg in [17, 18, 42]:
        sig, fs, _ = dev_segments[seg]
        p = determine_raw_polarity(sig, fs)
        assert p["polarity"] == 0, f"Seg {seg} should be UNDETERMINED but got {p['polarity']}"
        assert p["status"] == "UNDETERMINED"


def test_dev_manual_overrides_enable_successful_analysis(dev_segments):
    # Verified ground-truth orientations with manual fallback
    overrides = {
        17: (-1, 515.604),
        18: (-1, 524.248),
        29: (-1, 621.096),
        42: (-1, 765.416),
        45: (1, 792.044),
    }
    cfg = BeatCounterConfig()
    for seg, (target_pol, ref_anchor) in overrides.items():
        sig, fs, meta = dev_segments[seg]
        res = analyze_v2_pipeline(sig, fs, cfg, t0=meta["time_origin_s"], meta=meta, polarity_override=target_pol)
        assert res["metadata"]["raw_polarity"] == target_pol
        assert res["metadata"]["polarity_source"] == "manual_override"
        assert res["metadata"]["qc_status"] == "PASS"
        assert res["metadata"]["n_accepted"] >= 5
        t = res["candidates"]["time_s"]
        assert np.any(np.abs(t - ref_anchor) < 0.15), f"Seg {seg} failed to detect event near {ref_anchor}"


# ---------------------------------------------------------------------------
# Focused Calcium-Assisted Polarity Unit Tests
# ---------------------------------------------------------------------------
def _synth_calcium_and_mech(fs=250.0, duration_s=5.0, freq=2.0, delay_s=0.070, polarity=1,
                            base_mech=0.5, amp_mech=0.3, noise_level=0.01,
                            biphasic=False, duty_cycle=0.30):
    """Generate synthetic synchronized calcium and mechanical signals."""
    t = np.arange(int(duration_s * fs)) / fs
    period = 1.0 / freq
    n_beats = int(duration_s * freq)

    sig_ca = np.full_like(t, 0.5)
    sig_mech = np.full_like(t, base_mech)

    for b in range(n_beats):
        t_ca_pk = (b + 0.5) * period
        t_mech_pk = t_ca_pk + delay_s

        # Calcium transient: fast rise, exponential decay
        ca_dt = t - t_ca_pk
        ca_pulse = np.exp(-((ca_dt) ** 2) / (2 * (0.025 ** 2)))
        ca_decay = np.where(ca_dt > 0, np.exp(-ca_dt / 0.12), 1.0)
        sig_ca += 0.4 * ca_pulse * ca_decay

        # Mechanical excursion: smooth contraction
        m_dt = t - t_mech_pk
        m_pulse = np.exp(-((m_dt) ** 2) / (2 * ((0.05 * duty_cycle / 0.3) ** 2)))
        if biphasic:
            # Add secondary shoulder / aftercontraction
            m_pulse += 0.4 * np.exp(-((m_dt - 0.08) ** 2) / (2 * (0.03 ** 2)))
        sig_mech += polarity * amp_mech * m_pulse

    rng = np.random.default_rng(42)
    sig_ca += rng.normal(0, noise_level, size=len(t))
    sig_mech += rng.normal(0, noise_level, size=len(t))

    return sig_mech, sig_ca, t


def test_calcium_ecc_both_orientations():
    """Verify correct detection of +1 and -1 orientations under known physiological latency."""
    fs = 250.0
    # +1 orientation (contractions upward)
    s_pos, ca, _ = _synth_calcium_and_mech(fs=fs, polarity=1, delay_s=0.065)
    res_pos = determine_raw_polarity(s_pos, fs, sig_cal=ca, fs_cal=fs)
    assert res_pos["polarity"] == 1
    assert res_pos["status"] == "OK"
    assert res_pos["method"] == "calcium_ecc_timing"

    # -1 orientation (contractions downward)
    s_neg, ca, _ = _synth_calcium_and_mech(fs=fs, polarity=-1, delay_s=0.065)
    res_neg = determine_raw_polarity(s_neg, fs, sig_cal=ca, fs_cal=fs)
    assert res_neg["polarity"] == -1
    assert res_neg["status"] == "OK"
    assert res_neg["method"] == "calcium_ecc_timing"


def test_calcium_ecc_invariances_drift_scale_phase():
    """Verify invariance to baseline offsets, slow drift, amplitude scaling, and start phase."""
    fs = 250.0
    for pol in (1, -1):
        s, ca, t = _synth_calcium_and_mech(fs=fs, polarity=pol, delay_s=0.080)
        # Add huge baseline offset and linear drift
        s_distorted = 100.0 + 5.0 * s + 0.5 * t
        res = determine_raw_polarity(s_distorted, fs, sig_cal=ca, fs_cal=fs)
        assert res["polarity"] == pol
        assert res["status"] == "OK"


def test_calcium_ecc_different_pacing_and_duty_cycles():
    """Verify performance across 1 Hz, 2 Hz, and 3 Hz pacing with varied duty cycles."""
    fs = 250.0
    for freq in (1.0, 2.0, 3.0):
        for pol in (1, -1):
            s, ca, _ = _synth_calcium_and_mech(fs=fs, freq=freq, polarity=pol, delay_s=0.060, duty_cycle=0.45)
            res = determine_raw_polarity(s, fs, sig_cal=ca, fs_cal=fs)
            assert res["polarity"] == pol


def test_calcium_ecc_double_peaks_and_aftercontractions():
    """Verify resilience to biphasic contractions and aftercontractions."""
    fs = 250.0
    for pol in (1, -1):
        s, ca, _ = _synth_calcium_and_mech(fs=fs, polarity=pol, biphasic=True, delay_s=0.075)
        res = determine_raw_polarity(s, fs, sig_cal=ca, fs_cal=fs)
        assert res["polarity"] == pol


def test_calcium_ecc_missing_and_noisy_calcium_fallback():
    """Verify seamless fallback to morphological cues when calcium is absent, flat, or pure noise."""
    fs = 250.0
    s_pos, _, _ = _synth_calcium_and_mech(fs=fs, polarity=1, delay_s=0.060, duty_cycle=0.25)

    # 1. Calcium is None -> morphological decision
    res_none = determine_raw_polarity(s_pos, fs, sig_cal=None)
    assert res_none["polarity"] == 1
    assert res_none["method"] != "calcium_ecc_timing"

    # 2. Calcium is flat line -> morphological fallback
    res_flat = determine_raw_polarity(s_pos, fs, sig_cal=np.ones_like(s_pos))
    assert res_flat["polarity"] == 1
    assert res_flat["method"] != "calcium_ecc_timing"

    # 3. Calcium is pure noise -> morphological fallback
    rng = np.random.default_rng(99)
    res_noise = determine_raw_polarity(s_pos, fs, sig_cal=rng.normal(0, 1, size=len(s_pos)))
    assert res_noise["polarity"] == 1
    assert res_noise["method"] != "calcium_ecc_timing"


def test_calcium_ecc_timestamp_alignment():
    """Verify correct handling when time origin of calcium differs from mechanical."""
    fs = 250.0
    s, ca, _ = _synth_calcium_and_mech(fs=fs, polarity=-1, delay_s=0.070)
    # Calcium starts 0.5s later in absolute recording time
    t0_m = 10.0
    t0_c = 10.5
    ca_shifted = ca[int(0.5 * fs):]  # slice calcium so its t0 is 10.5
    res = determine_raw_polarity(s, fs, sig_cal=ca_shifted, fs_cal=fs, t0_raw=t0_m, t0_cal=t0_c)
    assert res["polarity"] == -1
    assert res["status"] == "OK"


def test_calcium_ecc_real_problematic_segments():
    """Verify automatic resolution of verified problematic segments in real experimental datasets."""
    wb_24 = r"C:\Users\talif\Desktop\240926_IMR90_1,2,3HZ individual transients result.xlsx"
    if not Path(wb_24).exists():
        pytest.skip("September 24 workbook not found.")

    from pixel_counter.io_utils import load_cytocypher_excel
    from pixel_counter.preprocessing import build_concatenated_signal

    cfg = BeatCounterConfig()
    # Problematic segments on Sept 24 and their ground truth
    targets = {
        29: -1,
        35: 1,
        38: 1,
        40: 1,
        41: -1,
        42: 1,
        43: 1,
        46: 1,
        17: -1,
        20: -1,
    }
    for seg, expected_pol in targets.items():
        sname_c = f"PixelCorrelation Segment {seg}"
        sname_ca = f"RatioMetricCalcium Segment {seg}"
        df_c, y_c, fs_c, _ = load_cytocypher_excel(wb_24, sheet_name=sname_c)
        df_ca, y_ca, fs_ca, _ = load_cytocypher_excel(wb_24, sheet_name=sname_ca)
        sig_c, _, meta_c = build_concatenated_signal(df_c, y_c, fs_c, cfg, force_invert=False)
        sig_ca, _, meta_ca = build_concatenated_signal(df_ca, y_ca, fs_ca, cfg, force_invert=False)
        t0_c = float(meta_c.get("time_origin_s", 0.0))
        t0_ca = float(meta_ca.get("time_origin_s", 0.0))

        pol = determine_raw_polarity(sig_c, fs_c, sig_cal=sig_ca, fs_cal=fs_ca, t0_raw=t0_c, t0_cal=t0_ca)
        assert pol["polarity"] == expected_pol, f"Seg {seg} expected {expected_pol}, got {pol['polarity']} ({pol['reason']})"
        assert pol["status"] == "OK"


def test_v2_default_is_pixel_only():
    """Verify BeatCounterConfig defaults to pixel-only mode (v2_use_calcium_polarity=False)."""
    cfg = BeatCounterConfig()
    assert cfg.v2_use_calcium_polarity is False


def test_default_mode_ignores_auxiliary_channels():
    """Verify that in default mode, presence, absence, or modification of RatioMetricCalcium
    or PixelIntensity in metadata does NOT alter the polarity determination."""
    from dataclasses import replace
    from pixel_counter.v2_pipeline import resolve_polarity

    fs = 250.0
    s_pos, ca, _ = _synth_calcium_and_mech(fs=fs, polarity=1, delay_s=0.065)
    cfg_default = BeatCounterConfig()
    assert cfg_default.v2_use_calcium_polarity is False

    # 1. No auxiliary channels in meta
    res_bare = resolve_polarity(s_pos, fs, cfg_default, meta={})
    assert res_bare["polarity"] == 1
    assert res_bare["method"] != "calcium_ecc_timing"

    # 2. Calcium and Intensity present in meta
    meta_with_aux = {
        "sig_cal": ca,
        "sig_int": np.ones_like(ca),
        "fs_cal": fs,
        "time_origin_s": 0.0,
        "t0_cal": 0.0,
    }
    res_with_aux = resolve_polarity(s_pos, fs, cfg_default, meta=meta_with_aux)
    assert res_with_aux["polarity"] == res_bare["polarity"]
    assert res_with_aux["method"] == res_bare["method"]
    assert res_with_aux["method"] != "calcium_ecc_timing"

    # 3. Deliberately corrupted/inverted calcium in meta
    meta_corrupted = {
        "sig_cal": -ca,
        "sig_int": np.random.randn(len(ca)),
        "fs_cal": fs,
        "time_origin_s": 0.0,
        "t0_cal": 0.0,
    }
    res_corrupted = resolve_polarity(s_pos, fs, cfg_default, meta=meta_corrupted)
    assert res_corrupted["polarity"] == res_bare["polarity"]
    assert res_corrupted["method"] == res_bare["method"]

    # 4. Explicitly enabled calcium mode activates ECC timing
    cfg_cal = replace(cfg_default, v2_use_calcium_polarity=True)
    res_enabled = resolve_polarity(s_pos, fs, cfg_cal, meta=meta_with_aux)
    assert res_enabled["polarity"] == 1
    assert res_enabled["method"] == "calcium_ecc_timing"


def test_cycle_morphology_cues_antisymmetry_and_invariances():
    """Verify cycle morphology cues have mathematical antisymmetry under inversion,
    and strict invariance to baseline shifts, scale, and phase offsets."""
    from pixel_counter.v2_signal import _cycle_morphology_cues

    fs = 250.0
    tc = 1.0 / 3.0  # 3 Hz pacing
    t = np.arange(int(5.0 * fs)) / fs

    # Asymmetric physiological cycle: fast contraction (30 ms), slow relaxation (180 ms)
    # Peak amplitude varies across beats (e.g. alternans), resting floor is flat
    a = np.zeros_like(t)
    n_beats = int(5.0 * 3.0)
    for b in range(n_beats):
        t_c = (b + 0.5) * tc
        dt = t - t_c
        amp = 0.5 + 0.2 * (b % 3)  # variable peak amplitude
        pulse = np.where((dt >= -0.03) & (dt <= 0), np.exp(-0.5 * (dt / 0.015) ** 2), 0.0)
        decay = np.where((dt > 0) & (dt < 0.20), np.exp(-dt / 0.06), 0.0)
        a += amp * (pulse + decay)

    raw_pos = 0.2 + a
    raw_neg = -raw_pos

    res_pos = _cycle_morphology_cues(raw_pos, fs, tc)
    res_neg = _cycle_morphology_cues(raw_neg, fs, tc)

    # 1. Antisymmetry under inversion
    assert np.isclose(res_pos["cycle_dur_ratio"], -res_neg["cycle_dur_ratio"], atol=1e-5)
    assert np.isclose(res_pos["cycle_var_ratio"], -res_neg["cycle_var_ratio"], atol=1e-5)

    # For positive orientation, dur_ratio > 0 and var_ratio > 0
    assert res_pos["cycle_dur_ratio"] > 0
    assert res_pos["cycle_var_ratio"] > 0

    # 2. Invariance to baseline shift
    res_shift = _cycle_morphology_cues(raw_pos + 15.0, fs, tc)
    assert np.isclose(res_shift["cycle_dur_ratio"], res_pos["cycle_dur_ratio"], atol=1e-5)
    assert np.isclose(res_shift["cycle_var_ratio"], res_pos["cycle_var_ratio"], atol=1e-5)

    # 3. Invariance to scale (multiplication by positive constant)
    res_scale = _cycle_morphology_cues(4.2 * raw_pos, fs, tc)
    assert np.isclose(res_scale["cycle_dur_ratio"], res_pos["cycle_dur_ratio"], atol=1e-5)
    assert np.isclose(res_scale["cycle_var_ratio"], res_pos["cycle_var_ratio"], atol=1e-5)

    # 4. Phase shift resilience
    raw_sliced = raw_pos[int(0.12 * fs):]
    res_sliced = _cycle_morphology_cues(raw_sliced, fs, tc)
    assert res_sliced["cycle_dur_ratio"] > 0
    assert res_sliced["cycle_var_ratio"] > 0


def test_sept24_3hz_resolved_segments():
    """Verify that previously undetermined 3 Hz segments on Sept 24 (39, 44, 45)
    are now automatically correctly resolved in pixel-only mode."""
    wb_24 = r"C:\Users\talif\Desktop\240926_IMR90_1,2,3HZ individual transients result.xlsx"
    if not Path(wb_24).exists():
        pytest.skip("September 24 workbook not found.")

    from pixel_counter.io_utils import load_cytocypher_excel
    from pixel_counter.preprocessing import build_concatenated_signal

    cfg = BeatCounterConfig()
    # Verified ground truth for newly resolved 3 Hz segments
    expected = {
        39: 1,
        44: 1,
        45: 1,
    }
    for seg, exp_pol in expected.items():
        df, y, fs, _ = load_cytocypher_excel(wb_24, sheet_name=f"PixelCorrelation Segment {seg}")
        sig, _, _ = build_concatenated_signal(df, y, fs, cfg, force_invert=False)
        res = determine_raw_polarity(sig, fs)
        assert res["polarity"] == exp_pol, f"Sept 24 Seg {seg} expected {exp_pol}, got {res['polarity']} ({res['reason']})"
        assert res["status"] == "OK"




