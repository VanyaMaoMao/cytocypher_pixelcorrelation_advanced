from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths

from .config import BeatCounterConfig
from .preprocessing import compute_spike_fraction
from .v2_signal import _dwell_asymmetry, estimate_resting_baseline, valid_blocks


def detect_incompatible_contraction_directions(
    sig_raw: np.ndarray,
    baseline_raw: np.ndarray,
    fs: float,
    sn: float,
) -> bool:
    """Detect tracking-direction reversal or incompatible opposite-direction contraction excursions.

    Uses ``sig_raw`` and ``baseline_raw`` directly (independent of chosen polarity and without
    requiring calcium data) to identify two classes of severe optical-tracking reversal:
      1. Temporal contraction-direction reversal across recording halves (e.g. downward
         contractions in the first half switching to upward contractions in the second half).
      2. Interleaved major excursions of comparable contraction width in opposite directions
         around an interior stationary baseline.
    """
    sig_arr = np.asarray(sig_raw, dtype=float)
    base_arr = np.asarray(baseline_raw, dtype=float)
    if sig_arr.size != base_arr.size or not np.isfinite(fs) or fs <= 0:
        return False
    valid_mask = np.isfinite(sig_arr) & np.isfinite(base_arr)
    fin = sig_arr[valid_mask]
    base_fin = base_arr[valid_mask]
    if fin.size < int(2.0 * fs):
        return False

    k = max(3, int(round(0.08 * fs)) | 1)
    xs = uniform_filter1d(median_filter(fin, size=k, mode="nearest"), k, mode="nearest")
    dx = np.abs(np.gradient(xs)) * fs

    stat_global = dx <= np.percentile(dx, 25.0)
    b_stat_global = float(np.median(xs[stat_global]))
    p99 = float(np.percentile(xs, 99.0))
    p01 = float(np.percentile(xs, 1.0))
    up_exc_g = p99 - b_stat_global
    dn_exc_g = b_stat_global - p01
    rng = float(np.percentile(xs, 98.0) - np.percentile(xs, 2.0))
    if rng < max(0.025, 10.0 * sn):
        return False

    win = max(15, int(round(2.0 * fs)))
    step = max(5, int(round(0.25 * fs)))
    t_c, b_v = [], []
    for ws in range(0, max(1, fin.size - win + 1), step):
        we = min(fin.size, ws + win)
        w_xs = xs[ws:we]
        w_dx = dx[ws:we]
        stat = w_dx <= np.percentile(w_dx, 30.0)
        b_v.append(float(np.median(w_xs[stat])))
        t_c.append(0.5 * (ws + we) / fs)

    idx_s = np.arange(fin.size) / fs
    b_diast = (
        gaussian_filter1d(np.interp(idx_s, t_c, b_v), max(1.0, 0.35 * fs), mode="nearest")
        if len(t_c) >= 2
        else np.full_like(fin, b_v[0])
    )
    sm = gaussian_filter1d(fin, max(1.0, 0.012 * fs), mode="nearest")
    up_wave = np.maximum(0.0, sm - b_diast)
    dn_wave = np.maximum(0.0, b_diast - sm)

    min_dist = max(5, int(0.18 * fs))
    min_prom = max(6.0 * sn, 0.12 * rng)
    pu, _ = find_peaks(up_wave, distance=min_dist, prominence=min_prom)
    pd_i, _ = find_peaks(dn_wave, distance=min_dist, prominence=min_prom)
    if len(pu) < 4 or len(pd_i) < 4:
        return False

    wu = peak_widths(up_wave, pu, rel_height=0.5)[0] / fs
    wd = peak_widths(dn_wave, pd_i, rel_height=0.5)[0] / fs
    med_hu = float(np.median(up_wave[pu]))
    med_hd = float(np.median(dn_wave[pd_i]))
    med_wu = float(np.median(wu))
    med_wd = float(np.median(wd))
    h_ratio = min(med_hu, med_hd) / max(med_hu, med_hd, 1e-9)

    # 1. Temporal contraction-direction reversal across first vs second half of detrended trace
    b_pos = estimate_resting_baseline(fin, fs)
    det = fin - (b_pos if np.all(np.isfinite(b_pos)) else base_fin)
    mid = det.size // 2
    d_det1 = _dwell_asymmetry(det[:mid], fs)
    d_det2 = _dwell_asymmetry(det[mid:], fs)
    if (
        np.isfinite(d_det1)
        and np.isfinite(d_det2)
        and (d_det1 * d_det2 < 0.0)
        and min(abs(d_det1), abs(d_det2)) >= 0.65
        and h_ratio >= 0.65
    ):
        return True

    # 2. Interleaved opposite-direction contractions at anti-phase around an interior stationary baseline
    bidir_g = min(up_exc_g, dn_exc_g) / max(up_exc_g, dn_exc_g, 1e-9)
    w_ratio = min(med_wu, med_wd) / max(med_wu, med_wd, 1e-9)
    if (
        bidir_g >= 0.35
        and min(up_exc_g, dn_exc_g) >= 0.14
        and len(pu) >= 5
        and len(pd_i) >= 5
        and abs(len(pu) - len(pd_i)) <= 1
        and h_ratio >= 0.30
        and w_ratio >= 0.60
        and min(med_wu, med_wd) >= 0.08
    ):
        if len(pu) >= 3:
            ibi_u = float(np.median(np.diff(pu)))
            if ibi_u > 0:
                phases = [float(np.min(np.abs(pd_i - u_idx))) / ibi_u for u_idx in pu]
                med_phase = float(np.median(phases))
                if 0.38 <= med_phase <= 0.62:
                    return True

    return False


def compute_descriptive_metrics(
    times: np.ndarray,
    amplitudes: np.ndarray,
    analyzable_duration_s: Optional[float] = None,
    is_rejected: bool = False,
) -> Dict[str, float]:
    """Descriptive statistics of the ACCEPTED contractions (never used to accept/reject)."""
    metrics: Dict[str, float] = {
        "ibi_cv": np.nan,
        "amp_cv": np.nan,
        "max_pause_s": np.nan,
        "contraction_frequency_hz": np.nan,
    }
    times = np.asarray(times, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if is_rejected or times.size == 0:
        return metrics
    if times.size >= 3:
        ibis = np.diff(times)
        metrics["max_pause_s"] = float(np.max(ibis))
        if np.mean(ibis) > 0:
            metrics["ibi_cv"] = float(np.std(ibis) / np.mean(ibis))
    if amplitudes.size >= 2 and np.mean(amplitudes) > 0:
        metrics["amp_cv"] = float(np.std(amplitudes) / np.mean(amplitudes))
    if analyzable_duration_s is not None and np.isfinite(analyzable_duration_s) and analyzable_duration_s > 0.0:
        metrics["contraction_frequency_hz"] = float(times.size / analyzable_duration_s)
    return metrics


def evaluate_qc_v2(
    sig: np.ndarray,
    fs: float,
    meta: Dict[str, Any],
    config: BeatCounterConfig,
) -> Tuple[str, str, Dict[str, Any]]:
    """Layer 1: Technical input-validity QC for the experimental v2 pipeline.

    Assesses whether the underlying data are suitable for numerical analysis.

    Returns
    -------
    status : "ERROR"  the input is invalid (bad sampling frequency, empty signal)
             "REJECT" the recording is fundamentally unusable (no/too little valid data,
                      unmasked or excessive conflicting overlapping samples, sub-10 Hz sampling)
             "PASS"   the recording is technically valid for analysis
    reason : short machine-readable reason
    qc_meta: technical diagnostics, including non-fatal ``qc_flags`` (e.g. clipping, NaN gaps)
    """
    qc_meta: Dict[str, Any] = {}
    flags: List[str] = []
    qc_meta["qc_flags"] = flags

    def _ret(st: str, rs: str) -> Tuple[str, str, Dict[str, Any]]:
        qc_meta["qc_layer1_status"] = st
        qc_meta["qc_layer1_reason"] = rs
        return st, rs, qc_meta

    # 1. Sampling frequency / time axis
    if not np.isfinite(fs) or fs <= 0.0:
        qc_meta["fs_valid"] = False
        return _ret("ERROR", "invalid_sampling_frequency")
    qc_meta["fs_valid"] = True

    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        qc_meta["empty_signal"] = True
        return _ret("ERROR", "empty_signal")

    t0 = meta.get("time_origin_s", None)
    if t0 is not None and not np.isfinite(t0):
        return _ret("ERROR", "invalid_time_origin")

    # 2. Coverage and genuine missing intervals
    valid_mask = np.isfinite(sig)
    valid_count = int(np.sum(valid_mask))
    qc_meta["coverage_fraction"] = float(valid_count / sig.size)
    if valid_count == 0:
        return _ret("REJECT", "no_valid_data")
    qc_meta["valid_duration_s"] = float(valid_count / fs)
    if valid_count / fs < 1.0:
        return _ret("REJECT", "insufficient_valid_data")
    n_gaps = len(valid_blocks(sig)) - 1
    qc_meta["nan_gap_count"] = int(max(0, n_gaps))
    if n_gaps > 0:
        flags.append("nan_gaps")

    # 3. Conflicting overlapping samples from the stitching stage
    n_conflicts = int(meta.get("stitch_overlap_conflicts", 0))
    if n_conflicts > 0:
        qc_meta["overlap_conflicts"] = n_conflicts
        conflicts_masked = bool(meta.get("stitch_conflicts_masked", False))
        conflict_frac = float(meta.get("stitch_conflict_fraction", 1.0))
        trimmed_indices = meta.get("stitch_trimmed_conflict_indices", None)
        properly_masked = (
            conflicts_masked
            and conflict_frac <= 0.05
            and (
                trimmed_indices is None
                or all(0 <= int(idx) < sig.size and np.isnan(sig[int(idx)]) for idx in trimmed_indices)
            )
        )
        if not properly_masked:
            qc_meta["overlap_conflicts_masked"] = False
            return _ret("REJECT", "overlap_conflict")
        qc_meta["overlap_conflicts_masked"] = True
        qc_meta["overlap_conflict_fraction"] = conflict_frac
        flags.append("masked_overlap_conflicts")

    # 4. Temporal resolution
    if fs < 10.0:
        return _ret("REJECT", "insufficient_temporal_resolution")

    # 5. Corrupted local noise and severe tracking noise bursts
    spike_frac = compute_spike_fraction(sig[valid_mask])
    qc_meta["spike_fraction"] = float(spike_frac)
    if spike_frac > 0.85:
        return _ret("REJECT", "excessive_local_noise")

    # 6. Clipping (non-fatal: amplitudes may be compressed but the waveform is analysable)
    valid_sig = sig[valid_mask]
    max_val = np.max(valid_sig)
    if np.sum(valid_sig == max_val) > 0.05 * valid_sig.size and max_val > 0.1:
        qc_meta["clipping_detected"] = True
        flags.append("signal_clipping")

    return _ret("PASS", "clean_signal")


def evaluate_signal_interpretability_qc(
    sig_raw: np.ndarray,
    amplitude: np.ndarray,
    baseline_raw: np.ndarray,
    fs: float,
    t0: float,
    events: pd.DataFrame,
    noise_sigma: float,
    config: BeatCounterConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Layer 2 & Layer 3 QC: Signal interpretability, local window quality, and event summary.

    Distinguishes scientific recording states:
      1. ``clean_with_contractions``: clear contractions above a quiet background.
      2. ``clean_silent_quiescent``: technically valid, quiet baseline with zero contractions.
      3. ``noisy_with_recoverable_contractions``: elevated noise/artefacts but recoverable beats.
      4. ``mixed_clean_and_unusable_intervals``: valid contractions alongside localized corrupted
         intervals (excluded from valid analysed duration and frequency denominator).
      5. ``noise_dominated_uninterpretable``: dominated by noise/artefacts with no reliable
         contraction evidence -> Layer 2 status ``REJECT`` (``noise_dominated_uninterpretable``).
      6. ``tracking_artifact``: bidirectional motion or tracking slippage -> Layer 2 status ``REJECT``.
    """
    amp = np.asarray(amplitude, dtype=float)
    base = np.asarray(baseline_raw, dtype=float)
    valid_mask = np.isfinite(amp)
    valid_count = int(np.sum(valid_mask))
    valid_duration_s = float(valid_count / fs) if (np.isfinite(fs) and fs > 0) else 0.0

    acc = events[events["status"] == "ACCEPTED"] if len(events) else events
    rej = events[events["status"] != "ACCEPTED"] if len(events) else events
    n_acc = int(len(acc))
    n_rej = int(len(rej))

    reasons = rej["reason"].astype(str).tolist() if len(rej) else []
    layer3_summary = {
        "n_candidates_total": int(len(events)),
        "n_accepted": n_acc,
        "n_rejected": n_rej,
        "n_rejected_impulse": int(sum(1 for r in reasons if r == "isolated_impulse_artefact")),
        "n_rejected_borderline_secondary": int(sum(1 for r in reasons if r == "borderline_secondary_excursion")),
        "n_rejected_duplicate_cycle": int(sum(1 for r in reasons if r == "duplicate_peak_within_cycle")),
        "n_rejected_secondary_morphology": int(sum(1 for r in reasons if r == "secondary_morphological_feature")),
        "n_rejected_insufficient_amplitude": int(sum(1 for r in reasons if r == "insufficient_amplitude")),
        "n_rejected_baseline_noise": int(sum(1 for r in reasons if r == "baseline_noise")),
    }

    sn = float(noise_sigma) if (np.isfinite(noise_sigma) and noise_sigma > 0) else 1e-9
    valid_amp = amp[valid_mask]
    p95_amp = float(np.percentile(valid_amp, 95.0)) if valid_amp.size else 0.0
    valid_base = base[np.isfinite(base)]
    baseline_drift_range = (
        float(np.percentile(valid_base, 95.0) - np.percentile(valid_base, 5.0))
        if valid_base.size
        else 0.0
    )

    # Window-wise local quality assessment (2.0 s sliding windows, 0.5 s step)
    win_n = max(10, int(round(2.0 * fs)))
    step_n = max(5, int(round(0.5 * fs)))
    n = amp.size
    acc_indices = acc["index"].to_numpy(dtype=int) if n_acc > 0 else np.array([], dtype=int)

    # Mark sample-level unusable mask across windows
    sample_unusable = ~valid_mask
    if n >= win_n and getattr(config, "v2_enable_signal_interpretability_qc", True):
        starts = list(range(0, n - win_n + 1, step_n))
        if starts[-1] + win_n < n:
            starts.append(max(0, n - win_n))
        win_flags = np.zeros(n, dtype=bool)
        for ws in starts:
            we = min(n, ws + win_n)
            w_amp = amp[ws:we]
            w_val = w_amp[np.isfinite(w_amp)]
            if w_val.size < 0.5 * (we - ws):
                win_flags[ws:we] = True
                continue
            has_acc_in_win = bool(np.any((acc_indices >= ws) & (acc_indices < we)))
            diffs = np.diff(w_val)
            med_d = float(np.median(diffs)) if diffs.size else 0.0
            mad_d = float(np.median(np.abs(diffs - med_d))) if diffs.size else 0.0
            loc_sn = (1.4826 * mad_d) / np.sqrt(2.0)
            w_p95 = float(np.percentile(w_val, 95.0))
            step_jump_frac = float(np.mean(np.abs(diffs) > max(0.015, 5.5 * sn))) if diffs.size else 0.0

            # Flag window as corrupted if it has no accepted beats and exhibits severe local noise or burst artefacts,
            # or in extended recordings (>= 10 s) when a long terminal cessation (>= 2.5 s) occurs past all contractions
            corrupted_win = (not has_acc_in_win) and (
                (loc_sn > max(0.0045, 2.2 * sn))
                or (step_jump_frac > 0.10 and loc_sn > 0.0032)
                or (n_acc == 0 and loc_sn >= 0.0035 and w_p95 >= 0.018)
                or (n_acc > 0 and (n / fs >= 10.0) and (ws > acc_indices[-1] + int(2.5 * fs)))
            )
            if corrupted_win:
                win_flags[ws:we] = True
        # Never mark within +-150 ms of an accepted contraction as unusable
        protect_hw = int(round(0.15 * fs))
        for a_idx in acc_indices:
            win_flags[max(0, a_idx - protect_hw) : min(n, a_idx + protect_hw + 1)] = False
        sample_unusable = sample_unusable | win_flags

    # Extract contiguous excluded intervals [t_start_s, t_end_s]
    excluded_intervals_s: List[List[float]] = []
    if np.any(sample_unusable):
        bad_idx = np.where(sample_unusable)[0]
        r_start = int(bad_idx[0])
        r_prev = int(bad_idx[0])
        for bi in bad_idx[1:]:
            bi_i = int(bi)
            if bi_i != r_prev + 1:
                if (r_prev + 1 - r_start) / fs >= 0.20:
                    excluded_intervals_s.append([round(float(t0 + r_start / fs), 4), round(float(t0 + (r_prev + 1) / fs), 4)])
                r_start = bi_i
            r_prev = bi_i
        if (r_prev + 1 - r_start) / fs >= 0.20:
            excluded_intervals_s.append([round(float(t0 + r_start / fs), 4), round(float(t0 + (r_prev + 1) / fs), 4)])

    excluded_duration_s = float(sum(iv[1] - iv[0] for iv in excluded_intervals_s))
    analyzable_duration_s = float(max(0.0, valid_duration_s - excluded_duration_s))
    analyzable_fraction = float(analyzable_duration_s / max(valid_duration_s, 1e-9))

    med_acc_snr = float(np.median(acc["snr"])) if n_acc > 0 else 0.0

    # Determine Layer 2 status, reason, and classification category
    enable_l2 = bool(getattr(config, "v2_enable_signal_interpretability_qc", True))
    meta = dict(meta or {})
    ca_m_pos = meta.get("calcium_matches_positive")
    ca_m_neg = meta.get("calcium_matches_negative")
    is_ca_tracking_artifact = enable_l2 and (
        ca_m_pos is not None
        and ca_m_neg is not None
        and ca_m_pos >= 5
        and ca_m_neg >= 5
        and abs(ca_m_pos - ca_m_neg) <= 2
    )
    is_incompatible_directions = enable_l2 and detect_incompatible_contraction_directions(
        sig_raw, base, fs, sn
    )

    sig_arr = np.asarray(sig_raw, dtype=float)
    valid_raw = sig_arr[np.isfinite(sig_arr)]
    diffs_raw = np.diff(valid_raw) if valid_raw.size >= 3 else np.array([], dtype=float)
    ad_raw = np.abs(diffs_raw - np.median(diffs_raw)) if diffs_raw.size else np.array([], dtype=float)
    zero_like_frac = float(np.mean(ad_raw <= 5e-5)) if ad_raw.size else 0.0

    if is_ca_tracking_artifact:
        l2_status = "REJECT"
        l2_reason = "tracking_artifact"
        sig_cat = "tracking_artifact"
    elif is_incompatible_directions:
        l2_status = "REJECT"
        l2_reason = "incompatible_contraction_directions"
        sig_cat = "tracking_artifact"
    elif n_acc == 0:
        is_noise_dominated = enable_l2 and (
            (sn >= 0.0045)
            or (sn >= 0.0035 and p95_amp >= 0.018)
            or (sn >= 0.0018 and p95_amp >= 0.008)
            or (analyzable_fraction < 0.50 and sn >= 0.0030)
            or (analyzable_fraction < 0.35)
            or (baseline_drift_range >= 0.040 and (p95_amp >= 0.015 or sn >= 0.0010))
            or (zero_like_frac >= 0.44)
        )
        if is_noise_dominated:
            l2_status = "REJECT"
            l2_reason = "noise_dominated_uninterpretable"
            sig_cat = "noise_dominated_uninterpretable"
        else:
            l2_status = "PASS"
            l2_reason = "clean_quiescent_no_contractions"
            sig_cat = "clean_silent_quiescent"
    else:
        med_h = (
            float(np.median(acc["height"]))
            if "height" in acc.columns
            else (float(np.median(acc["amplitude"])) if "amplitude" in acc.columns else 0.0)
        )
        med_w = float(np.median(acc["fwhm_s"])) if "fwhm_s" in acc.columns else 0.15
        drift_to_h = baseline_drift_range / max(med_h, 1e-6)

        det_sym = (
            valid_raw - uniform_filter1d(valid_raw, max(5, int(2.0 * fs)), mode="nearest")
            if (valid_raw.size >= 5 and np.isfinite(fs) and fs > 0)
            else np.array([], dtype=float)
        )
        p98_pos = float(np.percentile(np.maximum(0.0, det_sym), 98.0)) if det_sym.size else 0.0
        p98_neg = float(np.percentile(np.maximum(0.0, -det_sym), 98.0)) if det_sym.size else 0.0
        bidir_sym = min(p98_pos, p98_neg) / max(p98_pos, p98_neg, 1e-9)
        raw_rng = float(np.percentile(valid_raw, 98.0) - np.percentile(valid_raw, 2.0)) if valid_raw.size else 0.0

        ac_max = 0.0
        if np.isfinite(fs) and fs > 0 and det_sym.size > int(0.5 * fs):
            det_c = det_sym - np.mean(det_sym)
            var = float(np.dot(det_c, det_c))
            if var > 1e-12:
                ac = np.correlate(det_c, det_c, mode="full")[det_c.size - 1 :] / var
                l0, l1 = max(1, int(0.22 * fs)), min(det_c.size - 1, int(3.5 * fs))
                if l1 > l0:
                    ac_max = float(np.max(ac[l0:l1]))

        max_hf = 0.0
        burst_ratio = 1.0
        if valid_raw.size >= int(2.0 * fs) and np.isfinite(fs) and fs > 0:
            k30 = max(3, int(round(0.03 * fs)) | 1)
            win_n = int(2.0 * fs)
            hop_n = int(0.5 * fs)
            n_wins = (valid_raw.size - win_n) // hop_n
            if n_wins >= 2:
                local_hf = []
                for w in range(n_wins):
                    sub = valid_raw[w * hop_n : w * hop_n + win_n]
                    sub_hf = sub - median_filter(sub, k30, mode="nearest")
                    local_hf.append(float(np.std(sub_hf)))
                local_hf_arr = np.asarray(local_hf, dtype=float)
                min_hf = float(np.min(local_hf_arr))
                max_hf = float(np.max(local_hf_arr))
                burst_ratio = max_hf / max(min_hf, 1e-6)

        n_bn = int(layer3_summary["n_rejected_baseline_noise"])
        n_cand = n_acc + n_rej

        is_noise_dominated_with_acc = enable_l2 and (
            (sn >= 0.0080 and analyzable_fraction < 0.25 and med_acc_snr < 9.5)
            or (analyzable_fraction < 0.30 and med_acc_snr < 9.0)
            or (med_acc_snr < 20.0 and n_bn >= 12)
            or (len(excluded_intervals_s) > 0 and n_rej >= n_acc and n_bn >= 6)
            or (med_acc_snr < 20.0 and n_rej >= 1.5 * n_acc and n_bn >= 10)
            or (drift_to_h >= 4.20 and (n_acc <= 5 or med_acc_snr < 20.0 or (bidir_sym >= 0.65 and ac_max < 0.35)))
            or (
                med_acc_snr < 19.0
                and (
                    raw_rng < 0.016
                    or med_h < 0.015
                    or (drift_to_h >= 1.45 and (n_rej > 0.30 * n_acc or med_h < 0.035))
                    or (bidir_sym >= 0.42 and n_rej >= 0.35 * n_acc and (n_bn >= 1 or drift_to_h >= 0.50))
                )
            )
            or (raw_rng <= 0.008 and n_acc <= 3)
            or (bidir_sym >= 0.80 and med_w <= 0.075 and n_cand >= 55 and n_rej >= 0.70 * n_acc)
            or (max_hf >= 0.010 and burst_ratio >= 4.0 and bidir_sym >= 0.70)
        )
        if is_noise_dominated_with_acc:
            l2_status = "REJECT"
            l2_reason = "noise_dominated_uninterpretable"
            sig_cat = "noise_dominated_uninterpretable"
        elif len(excluded_intervals_s) > 0:
            l2_status = "PASS"
            l2_reason = "partial_corrupted_intervals_excluded"
            sig_cat = "mixed_clean_and_unusable_intervals"
        elif sn >= 0.0035 or layer3_summary["n_rejected_impulse"] > 0 or med_acc_snr < 12.0 or (n_rej >= n_acc and n_acc > 0):
            l2_status = "PASS"
            l2_reason = "recoverable_contractions_in_noise"
            sig_cat = "noisy_with_recoverable_contractions"
        else:
            l2_status = "PASS"
            l2_reason = "clean_signal"
            sig_cat = "clean_with_contractions"

    l2_meta: Dict[str, Any] = {
        "qc_layer2_status": l2_status,
        "qc_layer2_reason": l2_reason,
        "signal_quality_category": sig_cat,
        "qc_layer3_summary": layer3_summary,
        "analyzable_duration_s": round(analyzable_duration_s, 4),
        "analyzable_fraction": round(analyzable_fraction, 4),
        "excluded_intervals_s": excluded_intervals_s,
        "baseline_drift_range": round(baseline_drift_range, 5),
        "p95_amplitude": round(p95_amp, 5),
        "median_accepted_snr": round(med_acc_snr, 2) if n_acc > 0 else np.nan,
        "incompatible_contraction_directions": bool(is_incompatible_directions),
    }
    return l2_status, l2_reason, l2_meta

