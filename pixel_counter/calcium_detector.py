"""
pixel_counter.calcium_detector
==============================

Independent scientific calcium transient detector, baseline estimator,
and quality control module for cardiomyocyte ratiometric fluorescence
recordings (e.g. IonOptix CytoCypher RatioMetricCalcium).

Key physiological & signal processing principles:
1. Positive Polarity: Calcium transients are always positive excursions
   above local baseline (F(t) >= F0). Polarity classification/inversion
   is never performed.
2. Authentic Time Axis: Sampling frequency, row boundaries, and timestamps
   are faithfully derived from the source Excel workbook without interpolation
   or grid resampling.
3. Sensor Dropout Handling: Hardware shutter closures / dark current drops
   (values < 0.05) are flagged and masked as non-physiological dropouts.
   Partial dropouts are excluded from peak detection while preserving valid
   transients; severe dropouts (>50% or <2.5s analyzable) trigger QC rejection.
4. Robust Baseline: Local baseline F0(t) is estimated using a rolling percentile
   filter (ignoring dropouts), followed by light Gaussian smoothing.
5. True Peak Height: Delta F = y_peak - F0(t_peak), reported in authentic
   source ratio units.
6. Spike & Tail Discrimination: Distinguishes physiological transients from
   isolated optical spikes (FWHM >= 35 ms) and avoids over-detecting minor
   fluctuations on prolonged decay tails via relative recovery checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass
class CalciumDetectorParams:
    """Parameters for calcium transient detection."""
    preset_name: str = "default"
    min_height: float = 0.015             # Minimum absolute peak height above baseline: y_peak - R0
    min_norm_height: float = 0.08         # Minimum normalized peak height: (y_peak - R0) / R0
    min_rel_amplitude: float = 0.45       # Minimum relative peak amplitude as fraction of reference amplitude
    min_prominence_sigma: float = 3.5     # Prominence threshold relative to noise sigma
    min_prominence_abs: float = 0.015     # Absolute floor on peak prominence
    min_recovery: float = 0.45            # Minimum relative drop from preceding peak height
    min_separation_s: float = 0.18        # Minimum refractory interval between peaks (seconds)
    smoothing_sigma_s: float = 0.015      # Canonical Gaussian smoothing sigma for candidate peak localization (seconds)
    snapping_window_s: float = 0.035      # Local raw maximum search window around candidate (seconds)
    baseline_window_s: float = 1.5        # Rolling percentile filter window (seconds)
    baseline_percentile: float = 15.0     # Percentile for local baseline estimation
    baseline_smooth_s: float = 0.20       # Baseline smoothing sigma (seconds)
    dropout_threshold: float = 0.05       # Values below this threshold are hardware sensor dropouts
    min_fwhm_s: float = 0.035             # Minimum FWHM to reject single-sample spikes (seconds)

    # Scientific Dual Threshold Parameters for Adult Calcium
    alpha_ref_main: float = 0.60
    k_main_sigma: float = 5.0
    min_norm_height_main: float = 0.22
    alpha_ref_sce: float = 0.28
    k_sce_sigma: float = 3.0
    min_norm_height_sce: float = 0.08
    is_adult_cardiomyocyte: bool = False


CALCIUM_STRICTNESS_PRESETS: Dict[str, CalciumDetectorParams] = {
    "permissive": CalciumDetectorParams(
        preset_name="permissive",
        min_height=0.010,
        min_norm_height=0.05,
        min_rel_amplitude=0.30,
        min_prominence_sigma=2.5,
        min_prominence_abs=0.010,
        min_recovery=0.30,
        min_separation_s=0.10,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.5,
        baseline_percentile=15.0,
        baseline_smooth_s=0.20,
        dropout_threshold=0.05,
        min_fwhm_s=0.030,
    ),
    "default": CalciumDetectorParams(
        preset_name="default",
        min_height=0.015,
        min_norm_height=0.08,
        min_rel_amplitude=0.45,
        min_prominence_sigma=3.5,
        min_prominence_abs=0.015,
        min_recovery=0.45,
        min_separation_s=0.12,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.5,
        baseline_percentile=15.0,
        baseline_smooth_s=0.20,
        dropout_threshold=0.05,
        min_fwhm_s=0.035,
    ),
    "strict": CalciumDetectorParams(
        preset_name="strict",
        min_height=0.025,
        min_norm_height=0.12,
        min_rel_amplitude=0.55,
        min_prominence_sigma=5.0,
        min_prominence_abs=0.025,
        min_recovery=0.55,
        min_separation_s=0.18,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.8,
        baseline_percentile=15.0,
        baseline_smooth_s=0.25,
        dropout_threshold=0.05,
        min_fwhm_s=0.040,
    ),
}

ADULT_CALCIUM_STRICTNESS_PRESETS: Dict[str, CalciumDetectorParams] = {
    "permissive": CalciumDetectorParams(
        preset_name="permissive",
        min_height=0.010,
        min_norm_height=0.05,
        min_rel_amplitude=0.25,
        min_prominence_sigma=2.5,
        min_prominence_abs=0.010,
        min_recovery=0.25,
        min_separation_s=0.10,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.5,
        baseline_percentile=15.0,
        baseline_smooth_s=0.20,
        dropout_threshold=0.05,
        min_fwhm_s=0.030,
        alpha_ref_main=0.45,
        k_main_sigma=4.0,
        min_norm_height_main=0.15,
        alpha_ref_sce=0.20,
        k_sce_sigma=2.2,
        min_norm_height_sce=0.05,
        is_adult_cardiomyocyte=True,
    ),
    "default": CalciumDetectorParams(
        preset_name="default",
        min_height=0.015,
        min_norm_height=0.08,
        min_rel_amplitude=0.45,
        min_prominence_sigma=3.5,
        min_prominence_abs=0.015,
        min_recovery=0.45,
        min_separation_s=0.11,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.5,
        baseline_percentile=15.0,
        baseline_smooth_s=0.20,
        dropout_threshold=0.05,
        min_fwhm_s=0.035,
        alpha_ref_main=0.60,
        k_main_sigma=5.0,
        min_norm_height_main=0.22,
        alpha_ref_sce=0.28,
        k_sce_sigma=3.0,
        min_norm_height_sce=0.08,
        is_adult_cardiomyocyte=True,
    ),
    "strict": CalciumDetectorParams(
        preset_name="strict",
        min_height=0.025,
        min_norm_height=0.12,
        min_rel_amplitude=0.55,
        min_prominence_sigma=5.0,
        min_prominence_abs=0.025,
        min_recovery=0.55,
        min_separation_s=0.14,
        smoothing_sigma_s=0.015,
        snapping_window_s=0.035,
        baseline_window_s=1.8,
        baseline_percentile=15.0,
        baseline_smooth_s=0.25,
        dropout_threshold=0.05,
        min_fwhm_s=0.040,
        alpha_ref_main=0.72,
        k_main_sigma=6.5,
        min_norm_height_main=0.28,
        alpha_ref_sce=0.35,
        k_sce_sigma=4.0,
        min_norm_height_sce=0.12,
        is_adult_cardiomyocyte=True,
    ),
}


def estimate_calcium_noise_sigma(
    sig: np.ndarray,
    fs: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Robust estimation of high-frequency fluorescence noise floor sigma.
    
    Uses median absolute deviation (MAD) of first differences on valid,
    non-dropout samples.
    """
    if len(sig) < 2:
        return 0.001

    if mask is not None:
        valid_vals = sig[mask]
    else:
        valid_vals = sig[np.isfinite(sig) & (sig >= 0.05)]

    if len(valid_vals) < 5:
        return 0.001

    diffs = np.diff(valid_vals)
    # Exclude large jumps across segment boundaries or spike edges
    med_d = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med_d)))
    sigma_est = (mad / 0.6745) / np.sqrt(2.0)
    return max(0.0001, float(sigma_est))


def detect_calcium_dropouts(
    sig: np.ndarray,
    fs: float,
    dropout_threshold: float = 0.05,
) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
    """Detect sensor dropouts, shutter closures, and missing measurements.
    
    Returns:
        is_dropout: boolean array matching sig length.
        dropout_intervals: list of (start_idx, end_idx) sample intervals.
        dropout_fraction: fraction of total recording spent in dropout.
    """
    n = len(sig)
    if n == 0:
        return np.array([], dtype=bool), [], 0.0

    raw_drop = (~np.isfinite(sig)) | (sig < dropout_threshold)
    if not np.any(raw_drop):
        return np.zeros(n, dtype=bool), [], 0.0

    # Dilate dropout mask by ~25 ms to absorb shutter opening/closing transition edges
    w_dil = max(3, int(round(0.025 * fs)))
    if w_dil % 2 == 0:
        w_dil += 1
    is_dropout = np.convolve(raw_drop.astype(float), np.ones(w_dil), mode="same") > 0.1

    # Also absorb samples adjacent to dropouts that remain severely below resting baseline
    valid_initial = sig[~raw_drop & np.isfinite(sig)]
    if len(valid_initial) > 10:
        med_val = float(np.median(valid_initial))
        sub_level = max(dropout_threshold, 0.60 * med_val)
        w_near = max(5, int(round(0.050 * fs)))
        near_drop = np.convolve(raw_drop.astype(float), np.ones(w_near), mode="same") > 0.1
        is_dropout |= (sig < sub_level) & near_drop

    is_dropout |= (~np.isfinite(sig))
    dropout_fraction = float(np.mean(is_dropout))

    intervals: List[Tuple[int, int]] = []
    in_interval = False
    start_idx = 0

    for i in range(n):
        if is_dropout[i]:
            if not in_interval:
                in_interval = True
                start_idx = i
        else:
            if in_interval:
                in_interval = False
                intervals.append((start_idx, i - 1))
    if in_interval:
        intervals.append((start_idx, n - 1))

    return is_dropout, intervals, dropout_fraction


def estimate_calcium_baseline(
    sig: np.ndarray,
    fs: float,
    params: CalciumDetectorParams,
    is_dropout: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate continuous local resting baseline R0(t) for ratiometric calcium.
    
    Evaluates each continuous valid segment independently without bridging
    or dragging the baseline down across zero-level acquisition dropouts.
    Ensures R0(t) remains strictly positive (>= 0.05) everywhere.
    """
    n = len(sig)
    if n == 0:
        return np.array([], dtype=float)

    if is_dropout is None:
        is_dropout, _, _ = detect_calcium_dropouts(sig, fs, params.dropout_threshold)

    valid_mask = ~is_dropout & np.isfinite(sig)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.full(n, 0.25, dtype=float)

    # Global median resting baseline for reference & fallback
    med_r0 = max(0.05, float(np.median(sig[valid_indices])))

    # Find contiguous valid chunks
    diff = np.diff(np.pad(valid_mask.astype(int), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    base = np.full(n, med_r0, dtype=float)
    w = max(5, int(params.baseline_window_s * fs))
    if w % 2 == 0:
        w += 1

    for s_idx, e_idx in zip(starts, ends):
        chunk_len = e_idx - s_idx + 1
        if chunk_len < 10:
            base[s_idx:e_idx + 1] = med_r0
            continue
        chunk = sig[s_idx:e_idx + 1]
        w_chunk = min(w, chunk_len if chunk_len % 2 == 1 else chunk_len - 1)
        if w_chunk < 5:
            b_chunk = np.full(chunk_len, float(np.median(chunk)))
        else:
            b_chunk = percentile_filter(chunk, percentile=params.baseline_percentile, size=w_chunk, mode="reflect")
            sigma_samples = max(1.0, params.baseline_smooth_s * fs)
            b_chunk = gaussian_filter1d(b_chunk, sigma=sigma_samples, mode="reflect")
        base[s_idx:e_idx + 1] = np.maximum(0.05, b_chunk)

    return base.astype(float)


def compute_normalized_calcium(
    sig_raw: np.ndarray,
    baseline: np.ndarray,
    is_dropout: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute scientifically normalized ratiometric calcium amplitude:
    
    Delta R / R0 = (R(t) - R0(t)) / R0(t)
    
    Resting baseline is exactly 0.0.
    Dropouts and missing values remain NaN to preserve authentic experimental gaps.
    """
    n = len(sig_raw)
    norm = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(sig_raw) & np.isfinite(baseline) & (baseline >= 0.05) & (sig_raw >= 0.05)
    if is_dropout is not None:
        valid = valid & (~np.asarray(is_dropout, dtype=bool))
    norm[valid] = (sig_raw[valid] - baseline[valid]) / baseline[valid]
    return norm


def detect_calcium_transients(
    sig_raw: np.ndarray,
    fs: float,
    t0: float = 0.0,
    params: Optional[CalciumDetectorParams] = None,
    return_metrics: bool = False,
) -> Any:
    """Detect calcium transient peaks in raw ratiometric fluorescence signal.
    
    Args:
        sig_raw: authentic measured calcium fluorescence trace (not inverted).
        fs: authentic sampling frequency in Hz.
        t0: authentic time origin in seconds.
        params: detector parameter configuration (default preset if None).
        return_metrics: if True, returns (events_df, baseline, raw_noise_sigma, is_dropout, detector_metrics).
        
    Returns:
        events_df: DataFrame with detected peaks and measurements.
        baseline: F0(t) array of length len(sig_raw).
        noise_sigma: estimated noise floor sigma.
        is_dropout: boolean array identifying dropout intervals.
        (optional) detector_metrics: dict of Stage 1/Stage 2 decision criteria and reference values.
    """
    if params is None:
        params = CALCIUM_STRICTNESS_PRESETS["default"]

    n = len(sig_raw)
    columns = [
        "peak_index", "time_s", "peak_raw_value", "local_baseline_value",
        "amplitude", "delta_r", "fwhm_s", "status", "reason", "origin", "per_peak_strictness",
        "diagnostic_status", "diagnostic_reason", "detection_stage"
    ]
    empty_metrics = {
        "ref_norm_amp": 0.50,
        "ref_prom": 0.05,
        "eff_min_norm_height": float(params.min_norm_height),
        "eff_prominence": float(params.min_prominence_abs),
        "prom_noise_floor": float(params.min_prominence_abs),
        "raw_noise_sigma": 0.001,
        "n_stage1_accepted": 0,
        "n_stage2_recovered": 0,
    }
    if n == 0:
        empty_df = pd.DataFrame(columns=columns)
        empty_df.attrs["detector_metrics"] = empty_metrics
        if return_metrics:
            return empty_df, np.array([], dtype=float), 0.001, np.array([], dtype=bool), empty_metrics
        return empty_df, np.array([], dtype=float), 0.001, np.array([], dtype=bool)

    # 1. Detect dropouts
    is_dropout, dropout_intervals, dropout_frac = detect_calcium_dropouts(
        sig_raw, fs, dropout_threshold=params.dropout_threshold
    )

    # 2. Estimate baseline and noise floor
    baseline = estimate_calcium_baseline(sig_raw, fs, params, is_dropout)
    raw_noise_sigma = estimate_calcium_noise_sigma(sig_raw, fs, mask=~is_dropout)

    valid_mask = ~is_dropout & np.isfinite(sig_raw)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < max(5, int(0.5 * fs)):
        empty_df = pd.DataFrame(columns=columns)
        empty_metrics["raw_noise_sigma"] = float(raw_noise_sigma)
        empty_df.attrs["detector_metrics"] = empty_metrics
        if return_metrics:
            return empty_df, baseline, raw_noise_sigma, is_dropout, empty_metrics
        return empty_df, baseline, raw_noise_sigma, is_dropout

    # 3. Canonical candidate peak finding on smoothed waveform
    smooth_sigma_samples = max(1.0, params.smoothing_sigma_s * fs)
    sig_bridged = np.interp(np.arange(n), valid_indices, sig_raw[valid_indices])
    sig_smooth = gaussian_filter1d(sig_bridged, sigma=smooth_sigma_samples, mode="reflect")
    smooth_noise_sigma = estimate_calcium_noise_sigma(sig_smooth, fs, mask=valid_mask)

    min_dist_samples = max(1, int(params.min_separation_s * fs))
    prominence_floor = max(0.008, 1.2 * raw_noise_sigma)

    candidate_peaks_raw, cand_props = find_peaks(
        sig_smooth,
        distance=min_dist_samples,
        prominence=prominence_floor,
    )
    if len(candidate_peaks_raw) > 0:
        proms_raw, lb_raw, rb_raw = peak_prominences(sig_smooth, candidate_peaks_raw)
    else:
        proms_raw, lb_raw, rb_raw = np.array([]), np.array([]), np.array([])

    candidate_peaks: List[int] = []
    cand_proms: List[float] = []
    for i, c in enumerate(candidate_peaks_raw):
        if is_dropout[c]:
            continue
        # For peaks whose decay or rise touches recording ends or dropout boundaries,
        # avoid artificial truncation of prominence by evaluating the unconstrained flank.
        bnd_margin = max(5, int(0.30 * fs))
        lr = float(sig_smooth[c] - sig_smooth[lb_raw[i]])
        rf = float(sig_smooth[c] - sig_smooth[rb_raw[i]])
        is_boundary = (
            lb_raw[i] <= bnd_margin
            or rb_raw[i] >= n - bnd_margin
            or np.any(is_dropout[max(0, lb_raw[i] - bnd_margin):lb_raw[i] + 1])
            or np.any(is_dropout[rb_raw[i]:min(n, rb_raw[i] + bnd_margin + 1)])
        )
        eff_p = max(lr, rf) if is_boundary else float(proms_raw[i])
        candidate_peaks.append(int(c))
        cand_proms.append(eff_p)

    if len(candidate_peaks) == 0:
        empty_df = pd.DataFrame(columns=columns)
        empty_metrics["raw_noise_sigma"] = float(raw_noise_sigma)
        empty_df.attrs["detector_metrics"] = empty_metrics
        if return_metrics:
            return empty_df, baseline, raw_noise_sigma, is_dropout, empty_metrics
        return empty_df, baseline, raw_noise_sigma, is_dropout

    # 4. Preliminary peak metrics & reference amplitude estimation
    w_snap = max(1, int(params.snapping_window_s * fs))
    widths_res = peak_widths(sig_smooth, candidate_peaks, rel_height=0.5)
    fwhm_s_arr = widths_res[0] / fs

    initial_cands = []
    for idx, c in enumerate(candidate_peaks):
        i0 = max(0, c - w_snap)
        i1 = min(n, c + w_snap + 1)
        sub_raw = sig_raw[i0:i1]

        # 3-point moving average crest window
        sub_3pt = np.convolve(sub_raw, [0.25, 0.5, 0.25], mode="same")
        best_3pt = i0 + int(np.argmax(sub_3pt))
        nbrs = [nbr for nbr in [best_3pt - 1, best_3pt, best_3pt + 1] if 0 <= nbr < n and valid_mask[nbr]]
        snapped_pk = int(nbrs[np.argmax([sig_raw[nbr] for nbr in nbrs])]) if nbrs else c

        raw_val = float(sig_raw[snapped_pk])
        base_val = float(baseline[snapped_pk])
        delta_r = raw_val - base_val
        norm_amp = delta_r / max(0.05, base_val)

        initial_cands.append({
            "cand_idx": c,
            "snapped_pk": snapped_pk,
            "raw_val": raw_val,
            "base_val": base_val,
            "delta_r": delta_r,
            "norm_amp": norm_amp,
            "fwhm_s": float(fwhm_s_arr[idx]),
            "prom": float(cand_proms[idx]),
        })

    # Estimate reference amplitude and prominence from top quartile of candidates
    # (Prevents small baseline noise ripples from dragging down the reference)
    if len(initial_cands) >= 2:
        sorted_a = np.sort([c["norm_amp"] for c in initial_cands])
        sorted_p = np.sort([c["prom"] for c in initial_cands])
        k = max(2, int(np.ceil(0.20 * len(sorted_a))))
        ref_norm_amp = float(np.median(sorted_a[-k:]))
        ref_prom = float(np.median(sorted_p[-k:]))
    elif len(initial_cands) == 1:
        ref_norm_amp = float(initial_cands[0]["norm_amp"])
        ref_prom = float(initial_cands[0]["prom"])
    else:
        ref_norm_amp = 0.50
        ref_prom = 0.05

    noise_mult = params.min_prominence_sigma * 0.43
    prom_noise_floor = max(params.min_prominence_abs, noise_mult * raw_noise_sigma)
    eff_min_norm_height = max(params.min_norm_height, params.min_rel_amplitude * ref_norm_amp)
    eff_prominence = max(prom_noise_floor, params.min_rel_amplitude * ref_prom)

    is_adult = bool(getattr(params, "is_adult_cardiomyocyte", False))
    if is_adult:
        # Scientific Adult Cardiomyocyte Dual-Threshold Detection
        med_base = float(np.median(baseline[valid_indices])) if len(valid_indices) > 0 else 1.0
        norm_sigma = raw_noise_sigma / max(0.05, med_base)
        qual_cands = [
            c for c in initial_cands
            if (c["norm_amp"] >= max(0.14, 4.0 * norm_sigma) and c["fwhm_s"] >= 0.08 and c["prom"] >= 2.5 * raw_noise_sigma)
        ]
        if qual_cands:
            sorted_a = np.sort([c["norm_amp"] for c in qual_cands])
            sorted_p = np.sort([c["prom"] for c in qual_cands])
            k = max(1, int(np.ceil(0.20 * len(sorted_a))))
            ref_norm_amp = float(np.median(sorted_a[-k:]))
            ref_prom = float(np.median(sorted_p[-k:]))
        else:
            ref_norm_amp = None
            ref_prom = None

        alpha_main = float(getattr(params, "alpha_ref_main", 0.60))
        k_main = float(getattr(params, "k_main_sigma", 5.0))
        t_abs_main = float(getattr(params, "min_norm_height_main", 0.22))

        alpha_sce = float(getattr(params, "alpha_ref_sce", 0.28))
        k_sce = float(getattr(params, "k_sce_sigma", 3.0))
        t_abs_sce = float(getattr(params, "min_norm_height_sce", 0.08))

        if ref_norm_amp is not None and np.isfinite(ref_norm_amp) and ref_norm_amp > 0:
            eff_main_floor = min(t_abs_main, alpha_main * ref_norm_amp)
            t_main_cal = max(alpha_main * ref_norm_amp, k_main * norm_sigma, eff_main_floor)
            eff_sce_floor = min(t_abs_sce, alpha_sce * ref_norm_amp)
            t_sce_cal = max(eff_sce_floor, k_sce * norm_sigma, eff_sce_floor)
        else:
            t_main_cal = max(k_main * norm_sigma, t_abs_main)
            t_sce_cal = max(k_sce * norm_sigma, t_abs_sce)

        t_main_cal = max(t_main_cal, t_sce_cal)
        eff_min_norm_height = t_main_cal

        prom_noise_floor = max(params.min_prominence_abs, 2.85 * raw_noise_sigma)
        if ref_prom is not None and np.isfinite(ref_prom):
            eff_prominence = max(prom_noise_floor, min(params.min_rel_amplitude * ref_prom, 2.5 * raw_noise_sigma))
        else:
            eff_prominence = max(prom_noise_floor, 2.5 * raw_noise_sigma)

        events_records = []
        for c_info in initial_cands:
            c = c_info["cand_idx"]
            snapped_pk = c_info["snapped_pk"]
            pk_time = float(t0 + snapped_pk / fs)
            raw_val = c_info["raw_val"]
            base_val = c_info["base_val"]
            delta_r = c_info["delta_r"]
            norm_amp = c_info["norm_amp"]
            fwhm_val = c_info["fwhm_s"]
            cand_prom = c_info["prom"]

            status = "ACCEPTED"
            reason = "accepted_main_transient"
            ev_class = "MAIN"
            det_stage = "stage1"

            if is_dropout[snapped_pk]:
                status = "REJECTED"
                reason = "in_sensor_dropout"
                ev_class = "REJECTED"
                det_stage = "none"
            elif fwhm_val < params.min_fwhm_s:
                status = "REJECTED"
                reason = "narrow_spike"
                ev_class = "REJECTED"
                det_stage = "none"
            elif cand_prom < prom_noise_floor:
                status = "REJECTED"
                reason = "below_noise_prominence_floor"
                ev_class = "REJECTED"
                det_stage = "none"
            else:
                if norm_amp >= t_main_cal:
                    status = "ACCEPTED"
                    reason = "accepted_main_transient"
                    ev_class = "MAIN"
                    det_stage = "stage1"
                elif norm_amp >= t_sce_cal:
                    status = "ACCEPTED"
                    reason = "accepted_secondary_transient"
                    ev_class = "SECONDARY"
                    det_stage = "stage2_sce"
                else:
                    status = "REJECTED"
                    reason = "below_calcium_sce_threshold"
                    ev_class = "REJECTED"
                    det_stage = "none"

            events_records.append({
                "cand_index": int(c),
                "peak_index": int(snapped_pk),
                "time_s": pk_time,
                "peak_raw_value": raw_val,
                "local_baseline_value": base_val,
                "amplitude": norm_amp,
                "delta_r": delta_r,
                "fwhm_s": fwhm_val,
                "prom": cand_prom,
                "status": status,
                "event_class": ev_class,
                "reason": reason,
                "diagnostic_status": status,
                "diagnostic_reason": reason,
                "origin": "auto",
                "per_peak_strictness": "inherit",
                "detection_stage": det_stage,
            })

        events_df = pd.DataFrame(events_records)
        n_main_cnt = sum(1 for r in events_records if r["status"] == "ACCEPTED" and r["event_class"] == "MAIN")
        n_sec_cnt = sum(1 for r in events_records if r["status"] == "ACCEPTED" and r["event_class"] == "SECONDARY")

        detector_metrics = {
            "ref_norm_amp": float(ref_norm_amp) if ref_norm_amp is not None else 0.50,
            "ref_prom": float(ref_prom) if ref_prom is not None else 0.05,
            "eff_min_norm_height": float(eff_min_norm_height),
            "eff_prominence": float(eff_prominence),
            "prom_noise_floor": float(prom_noise_floor),
            "raw_noise_sigma": float(raw_noise_sigma),
            "n_stage1_accepted": int(n_main_cnt),
            "n_stage2_recovered": int(n_sec_cnt),
            "n_main": int(n_main_cnt),
            "n_secondary": int(n_sec_cnt),
            "threshold_main": float(t_main_cal),
            "threshold_sce": float(t_sce_cal),
            "threshold_ordering_status": "distinct" if t_main_cal > t_sce_cal else "tied",
        }
        events_df.attrs["detector_metrics"] = detector_metrics
        if return_metrics:
            return events_df, baseline, raw_noise_sigma, is_dropout, detector_metrics
        return events_df, baseline, raw_noise_sigma, is_dropout

    # 5. Evaluate candidate peaks against scientific criteria (Stage 1: High confidence)
    events_records: List[Dict[str, Any]] = []

    for c_info in initial_cands:
        c = c_info["cand_idx"]
        snapped_pk = c_info["snapped_pk"]
        pk_time = float(t0 + snapped_pk / fs)
        raw_val = c_info["raw_val"]
        base_val = c_info["base_val"]
        delta_r = c_info["delta_r"]
        norm_amp = c_info["norm_amp"]
        fwhm_val = c_info["fwhm_s"]
        cand_prom = c_info["prom"]

        status = "ACCEPTED"
        reason = "valid_transient"

        if is_dropout[snapped_pk]:
            status = "REJECTED"
            reason = "in_sensor_dropout"
        elif fwhm_val < params.min_fwhm_s:
            status = "REJECTED"
            reason = "narrow_spike"
        elif cand_prom < prom_noise_floor:
            status = "REJECTED"
            reason = "below_noise_prominence_floor"
        elif cand_prom < eff_prominence:
            status = "REJECTED"
            reason = "below_prominence_threshold"
        elif norm_amp < eff_min_norm_height:
            status = "REJECTED"
            reason = "below_amplitude_threshold"

        events_records.append({
            "cand_index": int(c),
            "peak_index": int(snapped_pk),
            "time_s": pk_time,
            "peak_raw_value": raw_val,
            "local_baseline_value": base_val,
            "amplitude": norm_amp,
            "delta_r": delta_r,
            "fwhm_s": fwhm_val,
            "prom": cand_prom,
            "status": status,
            "reason": reason,
            "diagnostic_status": status,
            "diagnostic_reason": reason,
            "origin": "auto",
            "per_peak_strictness": "inherit",
            "detection_stage": "stage1" if status == "ACCEPTED" else "none",
        })

    # 6. Stage 2: Secondary candidate validation for coupled doublets and weaker transients
    if params.preset_name in ("default", "permissive"):
        stage1_acc = [i for i, r in enumerate(events_records) if r["status"] == "ACCEPTED"]

        for i, r in enumerate(events_records):
            if r["status"] == "ACCEPTED":
                continue
            if r["reason"] in ("in_sensor_dropout", "narrow_spike", "below_noise_prominence_floor"):
                continue

            c = r["cand_index"]
            c_info = initial_cands[i]
            delta_r = r["delta_r"]
            norm_amp = r["amplitude"]
            cand_prom = r["prom"]
            fwhm_val = r["fwhm_s"]

            # MANDATORY RULE: No automatic peak below displayed Stage 1 criterion
            if norm_amp < eff_min_norm_height:
                continue

            # Must be well above raw noise floor
            if delta_r < 3.0 * raw_noise_sigma:
                continue
            if cand_prom < prom_noise_floor:
                continue
            if fwhm_val < params.min_fwhm_s:
                continue

            # Check relation to nearby accepted transients
            nearby_acc = [j for j in stage1_acc if abs(events_records[j]["peak_index"] - r["peak_index"]) / fs < 1.2]

            # Sub-case A: Coupled premature / double transient (adjacent to an accepted peak within 0.12s - 0.35s)
            is_doublet_recovered = False
            for j in nearby_acc:
                dt = abs(events_records[j]["time_s"] - r["time_s"])
                if 0.12 <= dt <= 0.35:
                    c_adj = events_records[j]["cand_index"]
                    c_min = min(c, c_adj)
                    c_max = max(c, c_adj)
                    trough_val = float(np.nanmin(sig_smooth[c_min:c_max]))

                    drop_self = sig_smooth[c] - trough_val
                    drop_adj = sig_smooth[c_adj] - trough_val
                    rel_drop = drop_self / max(1e-4, delta_r)

                    # If candidate precedes accepted peak (premature beat):
                    if r["time_s"] < events_records[j]["time_s"]:
                        if (drop_self >= 2.0 * raw_noise_sigma and drop_adj >= 2.0 * raw_noise_sigma 
                            and rel_drop >= 0.20 and cand_prom >= max(params.min_prominence_abs, 1.8 * raw_noise_sigma)):
                            is_doublet_recovered = True
                            break
                    else:
                        # Candidate follows accepted peak on decay phase
                        adj_delta_r = events_records[j]["delta_r"]
                        rel_decay_adj = drop_adj / max(1e-4, adj_delta_r)
                        if (rel_decay_adj >= 0.35 and drop_self >= 2.5 * raw_noise_sigma 
                            and cand_prom >= max(params.min_prominence_abs, 1.8 * raw_noise_sigma)):
                            is_doublet_recovered = True
                            break

            if is_doublet_recovered:
                r["status"] = "ACCEPTED"
                r["reason"] = "valid_transient"
                r["diagnostic_status"] = "ACCEPTED"
                r["diagnostic_reason"] = "valid_transient"
                r["detection_stage"] = "stage2_doublet"
                continue

            # Sub-case B: Weaker regular / alternans transient (separated from all accepted by >= 0.35s)
            min_dt_to_acc = min([abs(events_records[j]["time_s"] - r["time_s"]) for j in stage1_acc]) if stage1_acc else 999.0
            if min_dt_to_acc >= 0.35:
                if cand_prom >= max(params.min_prominence_abs, 2.2 * raw_noise_sigma) and delta_r >= 3.5 * raw_noise_sigma and fwhm_val >= 0.08:
                    r["status"] = "ACCEPTED"
                    r["reason"] = "valid_transient"
                    r["diagnostic_status"] = "ACCEPTED"
                    r["diagnostic_reason"] = "valid_transient"
                    r["detection_stage"] = "stage2_alternans"
                    continue

    # 7. Recovery check between successive accepted events
    accepted_indices = [i for i, r in enumerate(events_records) if r["status"] == "ACCEPTED"]
    if len(accepted_indices) > 1 and params.min_recovery > 0.0:
        for i in range(len(accepted_indices) - 1):
            curr_idx = accepted_indices[i]
            next_idx = accepted_indices[i + 1]
            c1 = events_records[curr_idx]["cand_index"]
            c2 = events_records[next_idx]["cand_index"]

            if c2 > c1:
                trough_val = float(np.nanmin(sig_smooth[c1:c2]))
                amp1 = max(0.001, events_records[curr_idx]["delta_r"])
                amp2 = max(0.001, events_records[next_idx]["delta_r"])

                drop1 = sig_smooth[c1] - trough_val
                recovery_frac = drop1 / amp1
                dt = (events_records[next_idx]["peak_index"] - events_records[curr_idx]["peak_index"]) / fs
                required_recovery = 0.20 if dt <= 0.45 else params.min_recovery

                if recovery_frac < required_recovery:
                    if amp1 >= amp2:
                        events_records[next_idx]["status"] = "REJECTED"
                        events_records[next_idx]["reason"] = "insufficient_recovery"
                        events_records[next_idx]["diagnostic_status"] = "REJECTED"
                        events_records[next_idx]["diagnostic_reason"] = "insufficient_recovery"
                    else:
                        events_records[curr_idx]["status"] = "REJECTED"
                        events_records[curr_idx]["reason"] = "insufficient_recovery"
                        events_records[curr_idx]["diagnostic_status"] = "REJECTED"
                        events_records[curr_idx]["diagnostic_reason"] = "insufficient_recovery"

    events_df = pd.DataFrame(events_records)
    n_stage1_acc = sum(1 for r in events_records if r["status"] == "ACCEPTED" and r.get("detection_stage") == "stage1")
    n_stage2_rec = sum(1 for r in events_records if r["status"] == "ACCEPTED" and str(r.get("detection_stage", "")).startswith("stage2"))

    detector_metrics = {
        "ref_norm_amp": float(ref_norm_amp),
        "ref_prom": float(ref_prom),
        "eff_min_norm_height": float(eff_min_norm_height),
        "eff_prominence": float(eff_prominence),
        "prom_noise_floor": float(prom_noise_floor),
        "raw_noise_sigma": float(raw_noise_sigma),
        "n_stage1_accepted": int(n_stage1_acc),
        "n_stage2_recovered": int(n_stage2_rec),
        "threshold_main": float(eff_min_norm_height),
        "threshold_sce": float(eff_min_norm_height * 0.50),
        "threshold_ordering_status": "distinct",
        "n_main": int(n_stage1_acc + n_stage2_rec),
        "n_secondary": 0,
    }
    events_df.attrs["detector_metrics"] = detector_metrics
    if return_metrics:
        return events_df, baseline, raw_noise_sigma, is_dropout, detector_metrics
    return events_df, baseline, raw_noise_sigma, is_dropout


def evaluate_calcium_qc(
    sig_raw: np.ndarray,
    fs: float,
    events_df: pd.DataFrame,
    is_dropout: np.ndarray,
    noise_sigma: float,
    params: CalciumDetectorParams,
) -> Tuple[str, str, Dict[str, Any]]:
    """Evaluate quality control for calcium fluorescence channel.
    
    Distinguishes:
    - PASS: valid transients detected, stable baseline
    - PASS: clean quiescent recording (zero transients, very quiet baseline)
    - PASS: partial dropout interval excluded, remaining recording valid
    - REJECT: severe sensor dropout (>30%, multiple prolonged drops, or fragmented <3s)
    - REJECT: noise dominated / uninterpretable (no coherent transients, high noise floor)
    - REJECT: empty or absent recording
    """
    n = len(sig_raw)
    meta: Dict[str, Any] = {
        "channel": "RatioMetricCalcium",
        "length": n,
        "fs": fs,
        "noise_sigma": float(noise_sigma),
        "dropout_fraction": float(np.mean(is_dropout)) if n > 0 else 1.0,
        "analyzable_samples": int(np.sum(~is_dropout)) if n > 0 else 0,
        "analyzable_duration_s": float(np.sum(~is_dropout) / fs) if (n > 0 and fs > 0) else 0.0,
    }

    if n == 0 or fs <= 0:
        return "REJECT", "empty_channel", meta

    total_dur = n / fs
    analyzable_dur = meta["analyzable_duration_s"]
    drop_frac = meta["dropout_fraction"]

    # Criterion 1: Severe Sensor Dropout & Fragmentation
    valid_mask = ~is_dropout & np.isfinite(sig_raw)
    diff = np.diff(np.pad(valid_mask.astype(int), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    valid_durs = [(e - s + 1)/fs for s, e in zip(starts, ends)]
    max_valid_dur = max(valid_durs) if valid_durs else 0.0

    drop_diff = np.diff(np.pad(is_dropout.astype(int), (1, 1), "constant"))
    d_starts = np.where(drop_diff == 1)[0]
    d_ends = np.where(drop_diff == -1)[0] - 1
    long_drops = [(e - s + 1)/fs for s, e in zip(d_starts, d_ends) if (e - s + 1)/fs >= 0.5]

    meta["max_valid_duration_s"] = max_valid_dur
    meta["prolonged_dropouts_count"] = len(long_drops)

    if drop_frac > 0.30 or len(long_drops) >= 2 or (max_valid_dur < 3.0 and drop_frac > 0.15) or analyzable_dur < 2.5:
        meta["rejection_category"] = "severe_sensor_dropout"
        return "REJECT", "severe_sensor_dropout", meta

    # Count accepted events
    n_accepted = 0
    if not events_df.empty:
        n_accepted = int(np.sum(events_df["status"] == "ACCEPTED"))
    meta["n_accepted"] = n_accepted

    # Criterion 2: Autocorrelation / Coherence Check for Noise Dominated Signals
    valid_sig = sig_raw[valid_mask]
    max_ac_lag = 0.0
    if len(valid_sig) > int(0.5 * fs):
        sig_c = valid_sig - np.mean(valid_sig)
        ac = np.correlate(sig_c, sig_c, mode="full")
        mid = len(ac) // 2
        norm_ac = ac[mid:] / (ac[mid] + 1e-12)
        min_lag = int(0.15 * fs)
        if len(norm_ac) > min_lag:
            max_ac_lag = float(np.max(norm_ac[min_lag:]))
    meta["max_autocorr_lag"] = max_ac_lag

    # Signal-to-noise ratio calculation
    acc_amps = events_df[events_df["status"] == "ACCEPTED"]["amplitude"].values if not events_df.empty else []
    med_amp = float(np.median(acc_amps)) if len(acc_amps) > 0 else 0.0
    snr = med_amp / max(1e-5, noise_sigma)
    meta["snr"] = snr

    is_adult = bool(getattr(params, "is_adult_cardiomyocyte", False))
    if is_adult:
        n_main = int(np.sum((events_df["status"] == "ACCEPTED") & (events_df["event_class"] == "MAIN"))) if not events_df.empty else 0
        meta["n_main"] = n_main

        has_authentic_main = False
        if n_main >= 1 and not events_df.empty:
            main_evs = events_df[(events_df["status"] == "ACCEPTED") & (events_df["event_class"] == "MAIN")]
            if not main_evs.empty:
                max_delta_r = float(np.max(main_evs["delta_r"])) if "delta_r" in main_evs.columns else float(np.max(main_evs["amplitude"]))
                raw_snr = max_delta_r / max(1e-5, noise_sigma)
                max_fwhm = float(np.max(main_evs["fwhm_s"]))
                if raw_snr >= 4.5 and max_fwhm >= 0.10:
                    has_authentic_main = True

        if not has_authentic_main:
            if noise_sigma > 0.020 and (max_ac_lag < 0.25 or snr < 3.2):
                meta["rejection_category"] = "uninterpretable_noise"
                return "REJECT", "uninterpretable_noise", meta

        if n_accepted > 0 and n_main == 0 and n_accepted <= 2 and (noise_sigma > 0.012 or snr < 6.5):
            meta["rejection_category"] = "uninterpretable_noise"
            return "REJECT", "uninterpretable_noise", meta

        if not has_authentic_main:
            if max_ac_lag < 0.10 and noise_sigma > 0.0035:
                meta["rejection_category"] = "noise_dominated"
                return "REJECT", "noise_dominated", meta
    else:
        # If autocorrelation is low and noise floor is high, this is pure optical/electronic noise
        # (e.g. SCRAMBLED ISO or ERROR-MOTION-CONTROLLER)
        if max_ac_lag < 0.10 and noise_sigma > 0.0035:
            meta["rejection_category"] = "noise_dominated"
            return "REJECT", "noise_dominated", meta

        # Severe uninterpretable noise check (e.g. IMR90 Seg 37)
        if noise_sigma > 0.020 and (max_ac_lag < 0.25 or snr < 3.2):
            meta["rejection_category"] = "uninterpretable_noise"
            return "REJECT", "uninterpretable_noise", meta

    # If zero accepted events
    if n_accepted == 0:
        if noise_sigma <= 0.0030:
            meta["quiescent"] = True
            return "PASS", "no_accepted_transients", meta
        meta["rejection_category"] = "uninterpretable_noise"
        return "REJECT", "uninterpretable_noise", meta

    # Partial dropouts (e.g. DISH1CONTROL Seg 4, DISH2ICI Seg 7, 8)
    if drop_frac > 0.02:
        return "PASS", "partial_sensor_dropout_excluded", meta

    return "PASS", "valid_transients", meta


def analyze_calcium_pipeline(
    sig_raw: np.ndarray,
    fs: float,
    t0: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    params: Optional[CalciumDetectorParams] = None,
) -> Dict[str, Any]:
    """Complete end-to-end calcium transient analysis pipeline.
    
    Runs multi-preset detection (default, permissive, strict) and computes
    frequency, BPM, quality control metrics, and scientifically normalized amplitude.
    """
    is_adult_p = bool(getattr(params, "is_adult_cardiomyocyte", False)) if params else False
    preset_dict = ADULT_CALCIUM_STRICTNESS_PRESETS if is_adult_p else CALCIUM_STRICTNESS_PRESETS

    if params is None:
        params = preset_dict["default"]

    n = len(sig_raw)
    dur = float(n / fs) if (n > 0 and fs > 0) else 0.0

    runs: Dict[str, Dict[str, Any]] = {}
    primary_events_df: Optional[pd.DataFrame] = None
    primary_baseline: Optional[np.ndarray] = None
    primary_noise: float = 0.001
    primary_dropout: Optional[np.ndarray] = None

    for preset_name in ["default", "permissive", "strict"]:
        p = preset_dict[preset_name]
        ev_df, base, n_sig, is_drop = detect_calcium_transients(sig_raw, fs, t0, p)
        qc_st, qc_rs, qc_meta = evaluate_calcium_qc(sig_raw, fs, ev_df, is_drop, n_sig, p)

        n_acc = int(np.sum(ev_df["status"] == "ACCEPTED")) if not ev_df.empty else 0
        analyzable_dur = qc_meta.get("analyzable_duration_s", dur)

        # QC Gating Precedence: If QC rejects the recording, reportable accepted events = 0
        if qc_st == "PASS" and n_acc > 0 and analyzable_dur > 0:
            freq_hz = float(n_acc / analyzable_dur)
            freq_bpm = float(freq_hz * 60.0)
            reportable_acc = n_acc
        elif qc_st == "PASS" and n_acc == 0:
            freq_hz = 0.0
            freq_bpm = 0.0
            reportable_acc = 0
        else:
            freq_hz = None
            freq_bpm = None
            reportable_acc = 0

        # Stage 1 normalized amplitude decision criterion (ΔR/R₀)
        det_metrics = getattr(ev_df, "attrs", {}).get("detector_metrics", {})
        eff_s1_amp = float(det_metrics.get("eff_min_norm_height", p.min_norm_height))
        ref_norm_amp = float(det_metrics.get("ref_norm_amp", 0.50))
        eff_prom = float(det_metrics.get("eff_prominence", p.min_prominence_abs))
        prom_floor = float(det_metrics.get("prom_noise_floor", p.min_prominence_abs))
        s1_acc = int(det_metrics.get("n_stage1_accepted", n_acc))
        s2_rec = int(det_metrics.get("n_stage2_recovered", 0))

        t_main_met = float(det_metrics.get("threshold_main", eff_s1_amp))
        t_sce_met = float(det_metrics.get("threshold_sce", getattr(p, "min_norm_height_sce", 0.08)))
        n_main_met = int(det_metrics.get("n_main", s1_acc))
        n_sec_met = int(det_metrics.get("n_secondary", s2_rec))

        runs[preset_name] = {
            "events": ev_df,
            "qc_status": qc_st,
            "qc_reason": qc_rs,
            "qc_meta": qc_meta,
            "n_accepted": reportable_acc,
            "n_main": n_main_met if qc_st == "PASS" else 0,
            "n_secondary": n_sec_met if qc_st == "PASS" else 0,
            "frequency_hz": freq_hz,
            "frequency_bpm": freq_bpm,
            "effective_amplitude_threshold": eff_s1_amp,
            "threshold_main": t_main_met,
            "threshold_sce": t_sce_met,
            "threshold_ordering_status": det_metrics.get("threshold_ordering_status", "distinct"),
            "stage1_amplitude_criterion": eff_s1_amp,
            "reference_amplitude": ref_norm_amp,
            "effective_prominence": eff_prom,
            "prominence_noise_floor": prom_floor,
            "raw_noise_sigma": float(n_sig),
            "n_stage1_accepted": s1_acc,
            "n_stage2_recovered": s2_rec,
            "detector_metrics": det_metrics,
            "params": p,
        }

        if preset_name == params.preset_name or primary_events_df is None:
            primary_events_df = ev_df
            primary_baseline = base
            primary_noise = n_sig
            primary_dropout = is_drop

    default_run = runs.get(params.preset_name, runs["default"])

    ibi_mean_s = None
    ibi_cv = None
    if default_run["qc_status"] == "PASS" and default_run["n_accepted"] > 1:
        acc_times = primary_events_df[primary_events_df["status"] == "ACCEPTED"]["time_s"].values
        ibis = np.diff(acc_times)
        if len(ibis) > 0 and np.mean(ibis) > 0:
            ibi_mean_s = float(np.mean(ibis))
            ibi_cv = float(np.std(ibis) / np.mean(ibis))

    mean_amp = None
    med_amp = None
    mean_delta_r = None
    med_delta_r = None
    if default_run["qc_status"] == "PASS" and default_run["n_accepted"] > 0:
        acc_amps = primary_events_df[primary_events_df["status"] == "ACCEPTED"]["amplitude"].values
        mean_amp = float(np.mean(acc_amps))
        med_amp = float(np.median(acc_amps))
        if "delta_r" in primary_events_df.columns:
            acc_dr = primary_events_df[primary_events_df["status"] == "ACCEPTED"]["delta_r"].values
            mean_delta_r = float(np.mean(acc_dr))
            med_delta_r = float(np.median(acc_dr))

    norm_waveform = compute_normalized_calcium(sig_raw, primary_baseline)

    pipeline_metadata = {
        "channel": "RatioMetricCalcium",
        "t0": float(t0),
        "fs": float(fs),
        "length": int(n),
        "duration_s": float(dur),
        "analyzable_duration_s": float(default_run["qc_meta"].get("analyzable_duration_s", dur)),
        "noise_sigma": float(primary_noise),
        "dropout_fraction": float(default_run["qc_meta"].get("dropout_fraction", 0.0)),
        "qc_status": default_run["qc_status"],
        "qc_reason": default_run["qc_reason"],
        "n_accepted": default_run["n_accepted"],
        "frequency_hz": default_run["frequency_hz"],
        "frequency_bpm": default_run["frequency_bpm"],
        "ibi_mean_s": ibi_mean_s,
        "ibi_cv": ibi_cv,
        "mean_amplitude": mean_amp,
        "median_amplitude": med_amp,
        "mean_delta_r": mean_delta_r,
        "median_delta_r": med_delta_r,
        "reference_amplitude": default_run.get("reference_amplitude"),
        "amplitude_threshold": default_run.get("effective_amplitude_threshold"),
        "stage1_amplitude_criterion": default_run.get("stage1_amplitude_criterion"),
        "effective_prominence": default_run.get("effective_prominence"),
        "prominence_noise_floor": default_run.get("prominence_noise_floor"),
        "n_stage1_accepted": default_run.get("n_stage1_accepted"),
        "n_stage2_recovered": default_run.get("n_stage2_recovered"),
    }
    if meta:
        pipeline_metadata.update(meta)

    return {
        "channel": "RatioMetricCalcium",
        "signal_raw": sig_raw,
        "baseline": primary_baseline,
        "signal_norm": norm_waveform,
        "is_dropout": primary_dropout,
        "events": primary_events_df,
        "qc_status": default_run["qc_status"],
        "qc_reason": default_run["qc_reason"],
        "metadata": pipeline_metadata,
        "runs": runs,
    }
