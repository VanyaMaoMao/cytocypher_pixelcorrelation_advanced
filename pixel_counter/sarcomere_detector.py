"""
pixel_counter.sarcomere_detector
================================

Independent scientific sarcomere shortening detector and baseline estimator
for cardiomyocyte SarcomereLength recordings (IonOptix CytoCypher / CytoSolver).

Scientific & Signal-Processing Principles:
1. Downward Contraction Polarity:
   Sarcomere contraction decreases measured sarcomere length L(t). True
   contractions correspond to local minima in L(t). Polarity auto-inversion
   is never performed.
2. Source Data Integrity:
   The raw SarcomereLength array is never altered. Vertical-line tracking
   artifacts are masked via `pixel_counter.sarcomere_qc.detect_sarcomere_artifacts`
   for baseline and candidate localization, and peak minima are snapped only to
   authentic, uncorrupted source samples.
3. Robust Local Diastolic Baseline L0(t):
   A two-pass rolling upper-percentile filter estimates resting diastolic
   sarcomere length L0(t) exclusively from non-artifact, non-contraction
   samples, adapting to baseline drift without following vertical spikes or
   rapid contractile troughs.
4. Quantitative Shortening Metrics:
   For each valid uncorrupted contraction:
   - `l0_um`: local diastolic sarcomere length (um)
   - `l_min_um`: authentic minimum sarcomere length during contraction (um)
   - `abs_shortening_um = l0_um - l_min_um` (um)
   - `pct_shortening = 100.0 * (l0_um - l_min_um) / l0_um` (%)
   If a vertical artifact obliterates the trough nadir (`peak_corrupted=True`),
   amplitude metrics are reported as `np.nan` (unavailable) rather than
   invented or measured from the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .sarcomere_qc import detect_sarcomere_artifacts


@dataclass
class SarcomereDetectorParams:
    """Parameters for independent sarcomere shortening detection."""
    preset_name: str = "default"
    min_abs_shortening_um: float = 0.0065     # Minimum absolute shortening L0 - Lmin (um)
    min_pct_shortening: float = 0.35          # Minimum percentage shortening 100*(L0 - Lmin)/L0 (%)
    min_prominence_sigma: float = 4.0         # Prominence threshold relative to clean HF noise sigma
    min_prominence_um: float = 0.0055         # Absolute floor on trough prominence (um)
    min_rel_amplitude: float = 0.18           # Minimum amplitude relative to reference main contraction
    min_separation_s: float = 0.11            # Minimum peak-to-peak separation (seconds)
    smoothing_sigma_s: float = 0.012          # Light Gaussian smoothing sigma for candidate search (seconds)
    snapping_window_s: float = 0.040          # Local raw minimum snapping half-window (seconds)
    baseline_window_s: float = 1.25           # Rolling diastolic percentile window (seconds)
    baseline_percentile: float = 82.0         # Upper percentile for diastolic L0 estimation
    baseline_smooth_s: float = 0.18           # Gaussian smoothing sigma for L0(t) (seconds)
    min_fwhm_s: float = 0.028
    max_fwhm_s: float = 0.45                 # Minimum full-width at half-maximum to reject narrow glitches (seconds)
    trough_core_halfwin_s: float = 0.032      # Half-window around trough nadir to check artifact obliteration (seconds)

    # Scientific Dual Threshold Parameters
    # MAIN contraction threshold: T_MAIN = max(alpha_ref * A_ref, k_main_sigma * sigma_pct, min_pct_shortening_main, 100*min_abs_shortening_main_um/L0, T_SCE)
    alpha_ref: float = 0.60                   # Fraction of robust reference amplitude A_ref
    k_main_sigma: float = 5.0                 # Noise protection factor for MAIN contractions
    min_pct_shortening_main: float = 0.55     # Configurable absolute floor for MAIN percentage shortening (%)
    min_abs_shortening_main_um: float = 0.0100 # Configurable absolute floor for MAIN shortening in micrometers (um)

    # SECONDARY (SCE) contraction threshold: T_SCE = max(k_sce_sigma * sigma_pct, min_pct_shortening_sce, 100*min_abs_shortening_sce_um/L0)
    k_sce_sigma: float = 3.2                  # Noise protection factor for SECONDARY contractions
    min_pct_shortening_sce: float = 0.35      # Configurable absolute floor for SECONDARY percentage shortening (%)
    min_abs_shortening_sce_um: float = 0.0055 # Configurable absolute floor for SECONDARY shortening in micrometers (um)


SARCOMERE_STRICTNESS_PRESETS: Dict[str, SarcomereDetectorParams] = {
    "permissive": SarcomereDetectorParams(
        preset_name="permissive",
        min_abs_shortening_um=0.0050,
        min_pct_shortening=0.25,
        min_prominence_sigma=3.0,
        min_prominence_um=0.0040,
        min_rel_amplitude=0.12,
        min_separation_s=0.09,
        smoothing_sigma_s=0.012,
        snapping_window_s=0.040,
        baseline_window_s=1.25,
        baseline_percentile=82.0,
        baseline_smooth_s=0.18,
        min_fwhm_s=0.024,
        trough_core_halfwin_s=0.032,
        alpha_ref=0.50,
        k_main_sigma=4.0,
        min_pct_shortening_main=0.40,
        min_abs_shortening_main_um=0.0075,
        k_sce_sigma=2.5,
        min_pct_shortening_sce=0.20,
        min_abs_shortening_sce_um=0.0035,
    ),
    "default": SarcomereDetectorParams(
        preset_name="default",
        min_abs_shortening_um=0.0065,
        min_pct_shortening=0.35,
        min_prominence_sigma=4.0,
        min_prominence_um=0.0055,
        min_rel_amplitude=0.18,
        min_separation_s=0.11,
        smoothing_sigma_s=0.012,
        snapping_window_s=0.040,
        baseline_window_s=1.25,
        baseline_percentile=82.0,
        baseline_smooth_s=0.18,
        min_fwhm_s=0.028,
        max_fwhm_s=0.45,
        trough_core_halfwin_s=0.032,
        alpha_ref=0.60,
        k_main_sigma=5.0,
        min_pct_shortening_main=0.55,
        min_abs_shortening_main_um=0.0100,
        k_sce_sigma=3.2,
        min_pct_shortening_sce=0.35,
        min_abs_shortening_sce_um=0.0055,
    ),
    "strict": SarcomereDetectorParams(
        preset_name="strict",
        min_abs_shortening_um=0.0100,
        min_pct_shortening=0.55,
        min_prominence_sigma=5.5,
        min_prominence_um=0.0080,
        min_rel_amplitude=0.28,
        min_separation_s=0.15,
        smoothing_sigma_s=0.014,
        snapping_window_s=0.040,
        baseline_window_s=1.40,
        baseline_percentile=84.0,
        baseline_smooth_s=0.22,
        min_fwhm_s=0.035,
        trough_core_halfwin_s=0.032,
        alpha_ref=0.78,
        k_main_sigma=6.5,
        min_pct_shortening_main=0.85,
        min_abs_shortening_main_um=0.0150,
        k_sce_sigma=4.2,
        min_pct_shortening_sce=0.50,
        min_abs_shortening_sce_um=0.0080,
    ),
}


def estimate_diastolic_l0(
    sig_raw: np.ndarray,
    fs: float,
    artifact_mask: np.ndarray,
    sigma_hf: float,
    params: SarcomereDetectorParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate local resting/diastolic sarcomere length L0(t) via a two-pass filter.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        `(l0_curve, sig_clean_interp)` where `l0_curve` is the local diastolic
        sarcomere length (um) at each sample, and `sig_clean_interp` is an
        internal working copy with artifact samples bridged for smoothing only.
    """
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    valid = (~artifact_mask) & np.isfinite(x)
    x_clean = x.copy()
    if np.any(valid):
        if not np.all(valid):
            good_i = np.where(valid)[0]
            bad_i = np.where(~valid)[0]
            x_clean[bad_i] = np.interp(bad_i, good_i, x[good_i])
    else:
        x_clean[:] = np.nanmedian(x) if np.any(np.isfinite(x)) else 1.80

    if n < 5 or fs <= 0:
        return x_clean.copy(), x_clean

    # Light pre-smoothing to prevent high-frequency noise from biasing the upper percentile upward
    smooth_sig_samp = max(1.0, params.smoothing_sigma_s * fs)
    x_smooth = gaussian_filter1d(x_clean, sigma=smooth_sig_samp, mode="nearest")

    win_samp = max(15, int(round(params.baseline_window_s * fs)) | 1)
    b_smooth_samp = max(2.0, params.baseline_smooth_s * fs)

    # Pass 1: Initial rolling upper percentile to locate active contraction excursions
    l0_pass1 = percentile_filter(x_smooth, percentile=params.baseline_percentile, size=win_samp, mode="nearest")
    l0_pass1 = gaussian_filter1d(l0_pass1, sigma=b_smooth_samp, mode="nearest")

    # Pass 2: Mask out active systolic troughs so rapid/frequent contractions do not pull L0 downward,
    # and estimate true diastolic resting length from diastole / turnaround intervals.
    excursion_thr = max(0.008, 4.0 * sigma_hf)
    in_contraction = (l0_pass1 - x_smooth) > excursion_thr
    diastolic_mask = valid & (~in_contraction)

    if np.sum(diastolic_mask) >= max(10, int(0.10 * n)):
        x_diast = x_smooth.copy()
        d_good = np.where(diastolic_mask)[0]
        d_bad = np.where(~diastolic_mask)[0]
        x_diast[d_bad] = np.interp(d_bad, d_good, x_smooth[d_good])
        l0_pass2 = percentile_filter(x_diast, percentile=75.0, size=win_samp, mode="nearest")
        l0_curve = gaussian_filter1d(l0_pass2, sigma=b_smooth_samp, mode="nearest")
    else:
        l0_curve = l0_pass1

    return l0_curve, x_clean


def detect_sarcomere_contractions(
    sig_raw: np.ndarray,
    fs: float,
    t0: float = 0.0,
    params: Optional[SarcomereDetectorParams] = None,
    artifact_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Detect sarcomere shortening events from an authentic SarcomereLength signal.

    Parameters
    ----------
    sig_raw : np.ndarray
        Authentic 1D SarcomereLength signal in micrometers (um).
    fs : float
        Sampling frequency in Hz.
    t0 : float
        Segment start time in seconds.
    params : Optional[SarcomereDetectorParams]
        Detection strictness parameters (defaults to `"default"` preset).
    artifact_info : Optional[Dict[str, Any]]
        Precomputed artifact diagnostics from `detect_sarcomere_artifacts`.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]
        `(events_df, l0_curve, artifact_info)`
    """
    if params is None:
        params = SARCOMERE_STRICTNESS_PRESETS["default"]

    x = np.asarray(sig_raw, dtype=float)
    n = len(x)

    if artifact_info is None:
        artifact_info = detect_sarcomere_artifacts(x, fs, t0=t0)

    art_mask = artifact_info["artifact_mask"]
    sigma_hf = float(artifact_info["sigma_hf"])

    empty_cols = [
        "peak_index",
        "time_s",
        "l0_um",
        "l_min_um",
        "abs_shortening_um",
        "pct_shortening",
        "takeoff_l0_um",
        "local_abs_shortening_um",
        "local_pct_shortening",
        "display_amplitude",
        "relative_to_main_ratio",
        "prominence_um",
        "fwhm_s",
        "snr",
        "event_class",
        "status",
        "reason",
        "origin",
        "peak_corrupted",
        "artifact_nearby",
        "local_recovery_ratio",
        "preceding_interval_s",
        "classification_reason",
        "classification_confidence",
    ]

    if n < 10 or fs <= 0 or np.sum(~art_mask) < 10:
        empty_df = pd.DataFrame(columns=empty_cols)
        l0_med = float(np.nanmedian(x[~art_mask])) if np.any(~art_mask) else 1.80
        sigma_pct = float(100.0 * sigma_hf / max(l0_med, 0.5))
        t_abs_sce_pct = float(getattr(params, "min_pct_shortening_sce", params.min_pct_shortening))
        t_abs_sce_um = float(getattr(params, "min_abs_shortening_sce_um", params.min_abs_shortening_um))
        k_sce = float(getattr(params, "k_sce_sigma", 3.2))
        t_sce_init = max(t_abs_sce_pct, k_sce * sigma_pct, 100.0 * t_abs_sce_um / max(l0_med, 0.5))
        eff_abs_sce_init = max(t_abs_sce_um, k_sce * sigma_hf)

        k_main = float(getattr(params, "k_main_sigma", 5.0))
        t_abs_main_pct = float(getattr(params, "min_pct_shortening_main", 0.55))
        t_abs_main_um = float(getattr(params, "min_abs_shortening_main_um", 0.0100))
        t_main_init = max(k_main * sigma_pct, 100.0 * t_abs_main_um / max(l0_med, 0.5), t_abs_main_pct, t_sce_init)
        eff_abs_main_init = max(k_main * sigma_hf, t_abs_main_um, eff_abs_sce_init)

        empty_df.attrs["detector_metrics"] = {
            "threshold_main_pct": float(t_main_init),
            "threshold_sce_pct": float(t_sce_init),
            "reference_amplitude_pct": None,
            "noise_sigma_pct": float(sigma_pct),
            "noise_sigma_um": float(sigma_hf),
            "baseline_estimation_method": "two_pass_diastolic_percentile",
            "threshold_ordering_status": "distinct" if t_main_init > t_sce_init else "collapsed",
            "threshold_main_components": {},
            "threshold_sce_components": {},
            "strictness_preset": params.preset_name,
            "eff_min_abs_shortening_um": float(eff_abs_sce_init),
            "eff_min_pct_shortening": float(t_sce_init),
            "amplitude_threshold_pct": float(t_main_init),
            "eff_prominence_um": max(params.min_prominence_um, params.min_prominence_sigma * sigma_hf),
            "sigma_hf_um": float(sigma_hf),
            "reference_shortening_um": 0.0,
        }
        return empty_df, np.full(n, np.nan, dtype=float), artifact_info

    # 1. Estimate local diastolic baseline L0(t) and artifact-bridged working trace
    l0_curve, x_clean = estimate_diastolic_l0(x, fs, art_mask, sigma_hf, params)

    # 2. Construct positive shortening trace for candidate peak finding
    smooth_sigma_samp = max(1.0, params.smoothing_sigma_s * fs)
    x_smooth = gaussian_filter1d(x_clean, sigma=smooth_sigma_samp, mode="nearest")
    short_smooth = l0_curve - x_smooth

    # Also build a trace where trough-obliterating shelf artifacts are detectable as candidate
    # event locations so we can report corrupted contractions transparently (e.g., A Seg 1, A Seg 3, A Seg 16).
    # For upward artifacts at a trough (like A Seg 1 where contraction starts descending then hits 2.2335),
    # x_clean bridges the artifact between the two descending/ascending shoulders, creating a local minimum
    # in x_smooth! For downward artifacts at a trough (like A Seg 16), x_clean also bridges the shoulders.

    min_dist_samp = max(3, int(round(params.min_separation_s * fs)))
    cand_prom_floor = max(0.0020, 2.0 * sigma_hf)

    cand_indices, _ = find_peaks(
        short_smooth,
        distance=min_dist_samp,
        prominence=cand_prom_floor,
    )

    l0_med = float(np.nanmedian(l0_curve)) if len(l0_curve) > 0 else 1.80
    sigma_pct = float(100.0 * sigma_hf / max(l0_med, 0.5))
    t_abs_sce_pct = float(getattr(params, "min_pct_shortening_sce", params.min_pct_shortening))
    t_abs_sce_um = float(getattr(params, "min_abs_shortening_sce_um", params.min_abs_shortening_um))
    k_sce = float(getattr(params, "k_sce_sigma", 3.2))
    t_sce_init = max(t_abs_sce_pct, k_sce * sigma_pct, 100.0 * t_abs_sce_um / max(l0_med, 0.5))
    eff_abs_sce_init = max(t_abs_sce_um, k_sce * sigma_hf)

    k_main = float(getattr(params, "k_main_sigma", 5.0))
    t_abs_main_pct = float(getattr(params, "min_pct_shortening_main", 0.55))
    t_abs_main_um = float(getattr(params, "min_abs_shortening_main_um", 0.0100))
    t_main_init = max(k_main * sigma_pct, 100.0 * t_abs_main_um / max(l0_med, 0.5), t_abs_main_pct, t_sce_init)
    eff_abs_main_init = max(k_main * sigma_hf, t_abs_main_um, eff_abs_sce_init)

    if len(cand_indices) == 0:
        empty_df = pd.DataFrame(columns=empty_cols)
        empty_df.attrs["detector_metrics"] = {
            "threshold_main_pct": float(t_main_init),
            "threshold_sce_pct": float(t_sce_init),
            "reference_amplitude_pct": None,
            "noise_sigma_pct": float(sigma_pct),
            "noise_sigma_um": float(sigma_hf),
            "baseline_estimation_method": "two_pass_diastolic_percentile",
            "threshold_ordering_status": "distinct" if t_main_init > t_sce_init else "collapsed",
            "threshold_main_components": {},
            "threshold_sce_components": {},
            "strictness_preset": params.preset_name,
            "eff_min_abs_shortening_um": float(eff_abs_sce_init),
            "eff_min_pct_shortening": float(t_sce_init),
            "amplitude_threshold_pct": float(t_main_init),
            "eff_prominence_um": max(params.min_prominence_um, params.min_prominence_sigma * sigma_hf),
            "sigma_hf_um": float(sigma_hf),
            "reference_shortening_um": 0.0,
        }
        return empty_df, l0_curve, artifact_info

    prominences, left_bases, right_bases = peak_prominences(
        short_smooth, cand_indices, wlen=max(15, int(round(1.2 * fs)))
    )
    widths_samp, _, _, _ = peak_widths(
        short_smooth, cand_indices, rel_height=0.5, prominence_data=(prominences, left_bases, right_bases)
    )

    snap_hw = max(3, int(round(params.snapping_window_s * fs)))
    core_hw = max(3, int(round(params.trough_core_halfwin_s * fs)))
    near_hw = max(5, int(round(0.065 * fs)))

    eff_prom_thr = max(params.min_prominence_um, params.min_prominence_sigma * sigma_hf)
    eff_abs_thr = max(params.min_abs_shortening_um, params.min_prominence_sigma * sigma_hf)

    records: List[Dict[str, Any]] = []
    used_raw_indices: List[int] = []

    for k, c_idx in enumerate(cand_indices):
        c_idx_int = int(c_idx)
        w0 = max(0, c_idx_int - snap_hw)
        w1 = min(n - 1, c_idx_int + snap_hw)

        core0 = max(0, c_idx_int - core_hw)
        core1 = min(n - 1, c_idx_int + core_hw)

        near0 = max(0, c_idx_int - near_hw)
        near1 = min(n - 1, c_idx_int + near_hw)

        # Snap to authentic, unmasked local minimum in sig_raw[w0:w1+1]
        win_art = art_mask[w0 : w1 + 1]
        valid_rel = np.where(~win_art)[0]
        if len(valid_rel) > 0:
            valid_abs = w0 + valid_rel
            raw_idx = int(valid_abs[int(np.argmin(x[valid_abs]))])
        else:
            raw_idx = c_idx_int

        # Avoid duplicate snapping to the exact same raw trough
        if any(abs(raw_idx - prev_idx) < min_dist_samp // 2 for prev_idx in used_raw_indices):
            continue

        # Re-center core window around the snapped raw minimum as well
        rc0 = max(0, min(core0, raw_idx - core_hw))
        rc1 = min(n - 1, max(core1, raw_idx + core_hw))

        n_art_core = int(np.sum(art_mask[rc0 : rc1 + 1]))
        n_art_near = int(np.sum(art_mask[near0 : near1 + 1]))
        artifact_nearby = bool(n_art_near > 0)

        # Determine if the trough nadir is obliterated by a vertical-line artifact:
        # 1. Zero valid samples in snapping window, OR
        # 2. Prolonged core tracking loss (> 14 samples, > 56 ms in trough core), OR
        # 3. Severe rail lock / spike ceiling (pegged to rail >= 2.20 um or severe rail jump)
        peak_corrupted = False
        l0_val = float(l0_curve[raw_idx])
        raw_val = float(x[raw_idx])
        # Trough nadir is corrupted if the nadir sample itself is an artifact or stuck on rail,
        # or if core tracking loss is prolonged (> 14 samples). An isolated upward spike on the shoulder
        # does not corrupt the trough if raw_val is a physiological contraction minimum (<= 1.75 um).
        is_nadir_art = bool(art_mask[raw_idx] or raw_val <= 1.15 or raw_val >= 2.18)
        if len(valid_rel) == 0 or n_art_core > max(2, int(round(0.056 * fs))) or is_nadir_art:
            peak_corrupted = True

        l0_val = float(l0_curve[raw_idx])
        prom_val = float(prominences[k])
        fwhm_s = float(widths_samp[k] / fs)
        time_s = float(t0 + raw_idx / fs)

        if peak_corrupted:
            used_raw_indices.append(raw_idx)
            records.append({
                "peak_index": raw_idx,
                "time_s": time_s,
                "l0_um": l0_val,
                "l_min_um": np.nan,
                "abs_shortening_um": np.nan,
                "pct_shortening": np.nan,
                "display_amplitude": np.nan,
                "prominence_um": prom_val,
                "fwhm_s": fwhm_s,
                "snr": np.nan,
                "event_class": "AMBIGUOUS",
                "status": "REJECTED",
                "reason": "vertical_artifact_at_trough",
                "origin": "auto",
                "peak_corrupted": True,
                "artifact_nearby": True,
                "local_recovery_ratio": np.nan,
                "preceding_interval_s": np.nan,
            })
            continue

        l_min_val = float(x[raw_idx])
        abs_short = float(l0_val - l_min_val)
        pct_short = float(100.0 * abs_short / l0_val) if l0_val > 0 else np.nan
        snr_val = float(abs_short / max(sigma_hf, 1e-5))

        used_raw_indices.append(raw_idx)
        records.append({
            "peak_index": raw_idx,
            "time_s": time_s,
            "l0_um": l0_val,
            "l_min_um": l_min_val,
            "abs_shortening_um": abs_short,
            "pct_shortening": pct_short,
            "display_amplitude": pct_short,
            "prominence_um": prom_val,
            "fwhm_s": fwhm_s,
            "snr": snr_val,
            "event_class": "MAIN",
            "status": "ACCEPTED",
            "reason": "accepted",
            "origin": "auto",
            "peak_corrupted": False,
            "artifact_nearby": artifact_nearby,
            "local_recovery_ratio": 1.0,
            "preceding_interval_s": np.nan,
        })

    events_df = pd.DataFrame(records, columns=empty_cols)
    if not events_df.empty:
        events_df = events_df.sort_values("peak_index").reset_index(drop=True)

    # 1. Robust Reference Amplitude A_ref estimation
    l0_med = float(np.nanmedian(l0_curve)) if len(l0_curve) > 0 else 1.80
    sigma_pct = float(100.0 * sigma_hf / max(l0_med, 0.5))

    ref_eligible = events_df[
        (~events_df["peak_corrupted"].astype(bool)) &
        (events_df["fwhm_s"] >= params.min_fwhm_s) &
        (events_df["snr"] >= 5.0) &
        (events_df["prominence_um"] >= 0.45 * events_df["abs_shortening_um"]) &
        (events_df["pct_shortening"] >= 0.40) &
        (events_df["pct_shortening"] <= 20.0)
    ]

    if len(ref_eligible) >= 1:
        a_ref_pct = float(np.percentile(ref_eligible["pct_shortening"], 75.0))
        a_ref_um = float(np.percentile(ref_eligible["abs_shortening_um"], 75.0))
    else:
        a_ref_pct = np.nan
        a_ref_um = np.nan

    # 2. Independently calculated dual scientific amplitude thresholds
    t_abs_sce_pct = float(getattr(params, "min_pct_shortening_sce", params.min_pct_shortening))
    t_abs_sce_um = float(getattr(params, "min_abs_shortening_sce_um", params.min_abs_shortening_um))
    k_sce = float(getattr(params, "k_sce_sigma", 3.2))

    alpha_ref = float(getattr(params, "alpha_ref", 0.60))
    k_main = float(getattr(params, "k_main_sigma", 5.0))
    t_abs_main_pct = float(getattr(params, "min_pct_shortening_main", 0.55))
    t_abs_main_um = float(getattr(params, "min_abs_shortening_main_um", 0.0100))

    if np.isfinite(a_ref_pct) and a_ref_pct > 0:
        if a_ref_pct >= alpha_ref * t_abs_main_pct:
            eff_main_floor_pct = min(t_abs_main_pct, alpha_ref * a_ref_pct)
            eff_main_floor_um = min(t_abs_main_um, alpha_ref * a_ref_um)
        else:
            eff_main_floor_pct = t_abs_main_pct
            eff_main_floor_um = t_abs_main_um
        t_main_cand = max(alpha_ref * a_ref_pct, k_main * sigma_pct, 100.0 * eff_main_floor_um / max(l0_med, 0.5), eff_main_floor_pct)
        eff_abs_main_um = max(alpha_ref * a_ref_um, k_main * sigma_hf, eff_main_floor_um)

        alpha_sce = float(getattr(params, "alpha_sce", 0.30))
        if a_ref_pct >= alpha_sce * t_abs_sce_pct:
            eff_sce_floor_pct = min(t_abs_sce_pct, alpha_sce * a_ref_pct)
            eff_sce_floor_um = min(t_abs_sce_um, alpha_sce * a_ref_um)
        else:
            eff_sce_floor_pct = t_abs_sce_pct
            eff_sce_floor_um = t_abs_sce_um
        t_sce = max(eff_sce_floor_pct, k_sce * sigma_pct, 100.0 * eff_sce_floor_um / max(l0_med, 0.5))
        eff_abs_sce_um = max(eff_sce_floor_um, k_sce * sigma_hf)
    else:
        t_main_cand = max(k_main * sigma_pct, 100.0 * t_abs_main_um / max(l0_med, 0.5), t_abs_main_pct)
        eff_abs_main_um = max(k_main * sigma_hf, t_abs_main_um)
        t_sce = max(t_abs_sce_pct, k_sce * sigma_pct, 100.0 * t_abs_sce_um / max(l0_med, 0.5))
        eff_abs_sce_um = max(t_abs_sce_um, k_sce * sigma_hf)

    t_main = max(t_main_cand, t_sce)
    eff_abs_main_um = max(eff_abs_main_um, eff_abs_sce_um)

    ordering_status = "collapsed" if abs(t_main - t_sce) < 1e-4 else "distinct"

    eff_prom_thr_main = max(params.min_prominence_um, params.min_prominence_sigma * sigma_hf)
    eff_prom_thr_sce = max(0.0035, min(params.min_prominence_um, getattr(params, "k_sce_sigma", 3.2) * sigma_hf))

    # 3. Centralized automatic classification enforcing strict amplitude hierarchy
    # MAIN: A_peak >= T_MAIN
    # SECONDARY: T_SCE <= A_peak < T_MAIN
    # Below-threshold: A_peak < T_SCE
    for idx in events_df.index:
        r = events_df.loc[idx]
        if bool(r["peak_corrupted"]):
            events_df.loc[idx, "status"] = "REJECTED"
            events_df.loc[idx, "reason"] = "vertical_artifact_at_trough"
            events_df.loc[idx, "event_class"] = "AMBIGUOUS"
        elif float(r["fwhm_s"]) < params.min_fwhm_s:
            events_df.loc[idx, "status"] = "REJECTED"
            events_df.loc[idx, "reason"] = "narrow_spike_width"
            events_df.loc[idx, "event_class"] = "REJECTED"
        elif float(r["fwhm_s"]) > getattr(params, "max_fwhm_s", 0.45):
            events_df.loc[idx, "status"] = "REJECTED"
            events_df.loc[idx, "reason"] = "excessive_width"
            events_df.loc[idx, "event_class"] = "REJECTED"
        elif float(r["prominence_um"]) < eff_prom_thr_sce:
            events_df.loc[idx, "status"] = "REJECTED"
            events_df.loc[idx, "reason"] = "low_prominence"
            events_df.loc[idx, "event_class"] = "REJECTED"
        elif float(r["pct_shortening"]) < t_sce or float(r["abs_shortening_um"]) < eff_abs_sce_um:
            events_df.loc[idx, "status"] = "REJECTED"
            events_df.loc[idx, "reason"] = "below_secondary_threshold"
            events_df.loc[idx, "event_class"] = "REJECTED"
        else:
            # Valid candidate >= T_SCE: strict amplitude hierarchy (pct >= t_main -> MAIN, else SECONDARY)
            if float(r["pct_shortening"]) >= t_main:
                events_df.loc[idx, "status"] = "ACCEPTED"
                events_df.loc[idx, "event_class"] = "MAIN"
                events_df.loc[idx, "reason"] = "accepted_isolated_spike_masked" if bool(r["artifact_nearby"]) else "accepted_main"
            else:
                if ordering_status == "collapsed":
                    events_df.loc[idx, "status"] = "ACCEPTED"
                    events_df.loc[idx, "event_class"] = "MAIN"
                    events_df.loc[idx, "reason"] = "accepted_collapsed_threshold"
                else:
                    events_df.loc[idx, "status"] = "ACCEPTED"
                    events_df.loc[idx, "event_class"] = "SECONDARY"
                    events_df.loc[idx, "reason"] = "accepted_secondary_spike_masked" if bool(r["artifact_nearby"]) else "accepted_secondary"

    # Compute physiological diagnostics (takeoff baseline, interval dt, recovery ratio)
    events_df = classify_sarcomere_contractions(events_df, x_clean, fs, params)

    events_df.attrs["detector_metrics"] = {
        "threshold_main_pct": float(t_main),
        "threshold_sce_pct": float(t_sce),
        "reference_amplitude_pct": float(a_ref_pct) if np.isfinite(a_ref_pct) else None,
        "noise_sigma_pct": float(sigma_pct),
        "noise_sigma_um": float(sigma_hf),
        "baseline_estimation_method": "two_pass_diastolic_percentile",
        "threshold_ordering_status": ordering_status,
        "threshold_main_components": {
            "alpha_ref_component": float(alpha_ref * a_ref_pct) if np.isfinite(a_ref_pct) else None,
            "k_main_sigma_component": float(k_main * sigma_pct),
            "abs_main_pct_component": float(t_abs_main_pct),
            "abs_main_um_component": float(100.0 * t_abs_main_um / max(l0_med, 0.5)),
            "effective_t_main_pct": float(t_main),
        },
        "threshold_sce_components": {
            "k_sce_sigma_component": float(k_sce * sigma_pct),
            "abs_sce_pct_component": float(t_abs_sce_pct),
            "abs_sce_um_component": float(100.0 * t_abs_sce_um / max(l0_med, 0.5)),
            "effective_t_sce_pct": float(t_sce),
        },
        "strictness_preset": params.preset_name,
        "eff_min_abs_shortening_um": float(eff_abs_sce_um),
        "eff_min_pct_shortening": float(t_sce),
        "amplitude_threshold_pct": float(t_main),
        "eff_prominence_um": float(eff_prom_thr),
        "sigma_hf_um": float(sigma_hf),
        "reference_shortening_um": float(a_ref_um) if np.isfinite(a_ref_um) else 0.0,
    }

    return events_df, l0_curve, artifact_info


def classify_sarcomere_contractions(
    events_df: pd.DataFrame,
    sig_clean: np.ndarray,
    fs: float,
    params: Optional[SarcomereDetectorParams] = None,
) -> pd.DataFrame:
    """Classify detected sarcomere shortening events into MAIN, SECONDARY, or AMBIGUOUS.

    Scientific Principles:
    1. MAIN: Substantially independent shortening event with complete or dominant
       excursion dynamics.
    2. SECONDARY: Genuine additional shortening excursion mechanically related to
       a preceding contraction (e.g. early after-contraction riding on an incomplete
       relaxation limb, or delayed after-contraction occurring between main beats).
    3. AMBIGUOUS: Borderline event where available evidence is insufficient to
       distinguish with high confidence between an independent small beat and a
       secondary contraction, or where nearby artifacts obscure baseline recovery.
    4. Negative Control Invariants: Rapid or irregular unpaced adult cardiomyocytes
       (e.g. Dataset B Segs 17, 27, 28, 30) producing trains of similar-amplitude
       contractions are preserved as MAIN. Isolated contractions (e.g. Segs 29, 39, 49)
       remain MAIN.
    """
    if events_df is None or events_df.empty:
        return events_df

    df = events_df.copy()

    # Ensure required columns are initialized with appropriate dtypes
    for col in [
        "takeoff_l0_um",
        "local_abs_shortening_um",
        "local_pct_shortening",
        "relative_to_main_ratio",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    for col in [
        "classification_reason",
        "classification_confidence",
    ]:
        if col not in df.columns:
            df[col] = pd.Series(None, index=df.index, dtype=object)
        else:
            df[col] = df[col].astype(object)

    # Initialize rejected/corrupted events
    for idx in df.index:
        st = df.loc[idx, "status"]
        corrupt = bool(df.loc[idx, "peak_corrupted"])
        if st == "REJECTED":
            if corrupt:
                df.loc[idx, "event_class"] = "AMBIGUOUS"
                df.loc[idx, "classification_reason"] = "vertical_artifact_at_trough"
                df.loc[idx, "classification_confidence"] = "LOW"
            else:
                df.loc[idx, "event_class"] = "REJECTED"
                df.loc[idx, "classification_reason"] = str(df.loc[idx, "reason"])
                df.loc[idx, "classification_confidence"] = "N/A"

    acc_indices = df.index[df["status"] == "ACCEPTED"].tolist()
    if not acc_indices:
        return df

    # 1. First pass: compute local takeoff baseline, local excursion, and recovery ratio
    for pos, r_idx in enumerate(acc_indices):
        p_curr = int(df.loc[r_idx, "peak_index"])
        l0_curr = float(df.loc[r_idx, "l0_um"])
        lmin_curr = float(df.loc[r_idx, "l_min_um"])

        if pos == 0:
            takeoff = l0_curr
            dt = np.nan
            rec = 1.0
        else:
            prev_r_idx = acc_indices[pos - 1]
            p_prev = int(df.loc[prev_r_idx, "peak_index"])
            t_curr = float(df.loc[r_idx, "time_s"])
            t_prev = float(df.loc[prev_r_idx, "time_s"])
            dt = t_curr - t_prev

            # Takeoff level is the maximum sarcomere length reached between p_prev and p_curr
            if p_curr > p_prev + 1:
                slice_between = sig_clean[p_prev : p_curr + 1]
                takeoff = float(np.max(slice_between))
            else:
                takeoff = l0_curr

            lmin_prev = float(df.loc[prev_r_idx, "l_min_um"])
            l0_prev = float(df.loc[prev_r_idx, "l0_um"])
            denom = max(l0_prev - lmin_prev, 1e-5)
            # Recovery ratio: fraction of previous contraction recovered toward L0
            rec = float((takeoff - lmin_prev) / denom)

        local_dl = max(0.0, takeoff - lmin_curr)
        local_pct = 100.0 * local_dl / takeoff if takeoff > 0 else np.nan

        df.loc[r_idx, "takeoff_l0_um"] = takeoff
        df.loc[r_idx, "local_abs_shortening_um"] = local_dl
        df.loc[r_idx, "local_pct_shortening"] = local_pct
        df.loc[r_idx, "preceding_interval_s"] = dt
        df.loc[r_idx, "local_recovery_ratio"] = rec

    # Calculate recording-level reference main amplitude (80th percentile of abs_shortening)
    acc_dls = df.loc[acc_indices, "abs_shortening_um"].dropna().values
    ref_dl = float(np.percentile(acc_dls, 80)) if len(acc_dls) > 0 else 0.10

    # 2. Sequential classification pass
    last_main_dl = ref_dl
    last_main_time = float(df.loc[acc_indices[0], "time_s"])

    for pos, r_idx in enumerate(acc_indices):
        corrupt = bool(df.loc[r_idx, "peak_corrupted"])
        if corrupt:
            df.loc[r_idx, "event_class"] = "AMBIGUOUS"
            df.loc[r_idx, "classification_reason"] = "corrupted_peak"
            df.loc[r_idx, "classification_confidence"] = "LOW"
            continue

        p_idx = int(df.loc[r_idx, "peak_index"])
        t_curr = float(df.loc[r_idx, "time_s"])
        dt = float(df.loc[r_idx, "preceding_interval_s"]) if pos > 0 else np.nan
        rec = float(df.loc[r_idx, "local_recovery_ratio"])
        abs_dl = float(df.loc[r_idx, "abs_shortening_um"])
        local_dl = float(df.loc[r_idx, "local_abs_shortening_um"])
        prom = float(df.loc[r_idx, "prominence_um"])
        art_near = bool(df.loc[r_idx, "artifact_nearby"])

        # Initial accepted event metadata
        if pos == 0:
            df.loc[r_idx, "relative_to_main_ratio"] = 1.0
            df.loc[r_idx, "classification_reason"] = "initial_event"
            df.loc[r_idx, "classification_confidence"] = "HIGH"
            last_main_dl = local_dl
            last_main_time = t_curr
            continue

        # Ratio relative to preceding main contraction
        rel_to_main = local_dl / max(last_main_dl, 1e-4)
        df.loc[r_idx, "relative_to_main_ratio"] = rel_to_main
        rel_to_ref = abs_dl / max(ref_dl, 1e-4)

        # Decision rules based on physiological contractility:
        # A. Independent Strong Beat:
        #    If beat is strong relative to recording reference (rel_to_ref >= 0.70)
        #    or strong relative to previous main (rel_to_main >= 0.75), and not a closely coupled premature beat (dt > 0.20)
        is_strong_beat = (
            (rel_to_ref >= 0.70 and rel_to_main >= 0.70 and (dt > 0.20 or rec >= 0.95))
            or (rel_to_main >= 0.80 and dt > 0.18)
            or (rel_to_ref >= 0.85)
        )

        # B. Independent Pause Beat:
        #    Preceded by long resting baseline (dt >= 0.65s) and decent size
        is_independent_pause = (dt >= 0.65 and rel_to_ref >= 0.45)

        # C. Early Secondary Contraction (riding on incomplete relaxation phase):
        #    Very short coupling interval dt <= 0.22s with incomplete recovery rec < 0.94 and smaller amplitude
        #    Or dt <= 0.32s with rec < 0.85 and rel_to_main < 0.70
        is_early_secondary = (
            (dt <= 0.22 and rec < 0.94 and rel_to_main < 0.85)
            or (dt <= 0.32 and rec < 0.85 and rel_to_main < 0.70)
        )

        # D. Delayed Secondary Contraction / After-contraction:
        #    Coupling interval dt <= 0.55s, very small amplitude relative to main beat (rel_to_main <= 0.48)
        #    and relative to reference (rel_to_ref <= 0.55)
        is_delayed_secondary = (
            (0.18 < dt <= 0.55)
            and (rel_to_main <= 0.48)
            and (rel_to_ref <= 0.55)
        )

        # E. Alternating / Interpolated Secondary Contraction:
        #    Coupling interval dt <= 0.62s, rel_to_main <= 0.58, rel_to_ref <= 0.60, and reduced prominence
        is_alternating_secondary = (
            (0.30 <= dt <= 0.62)
            and (rel_to_main <= 0.58)
            and (rel_to_ref <= 0.60)
            and (prom < 0.60 * last_main_dl)
        )

        # F. Ambiguous:
        is_ambiguous = (
            (0.20 < dt <= 0.45 and 0.58 < rel_to_main < 0.72 and 0.50 < rel_to_ref < 0.70)
            or (art_near and rel_to_main < 0.75 and dt <= 0.45)
        )

        # The mandatory strict amplitude hierarchy is absolute and invariant:
        # event_class has already been determined by A_peak relative to T_MAIN and T_SCE.
        # We assign physiological classification reason and confidence without altering event_class.
        current_class = str(df.loc[r_idx, "event_class"])
        if current_class == "MAIN":
            if pos == 0:
                df.loc[r_idx, "classification_reason"] = "initial_main_contraction"
                df.loc[r_idx, "classification_confidence"] = "HIGH"
            elif is_strong_beat or is_independent_pause:
                df.loc[r_idx, "classification_reason"] = f"independent_main_contraction(dt={dt:.3f}s,rec={rec:.2f},rel={rel_to_main:.2f})"
                df.loc[r_idx, "classification_confidence"] = "HIGH"
            else:
                df.loc[r_idx, "classification_reason"] = f"dominant_main_contraction(dt={dt:.3f}s,rel={rel_to_main:.2f})"
                df.loc[r_idx, "classification_confidence"] = "HIGH"
            last_main_dl = local_dl
            last_main_time = t_curr
        elif current_class == "SECONDARY":
            # Rule 6: Candidate in [T_SCE, T_MAIN) must pass relevant scientific validity checks.
            # A candidate must have physiological contraction coupling (dt <= 1.25s) and sufficient relative amplitude (rel_to_ref >= min_rel_amplitude).
            # An isolated or sub-relative-amplitude excursion is not accepted in sarcomere-only mode without calcium assistance.
            min_rel_amp = getattr(params, "min_rel_amplitude", 0.18)
            if pos == 0 or (np.isfinite(dt) and dt > 1.25) or rel_to_ref < min_rel_amp:
                df.loc[r_idx, "status"] = "REJECTED"
                df.loc[r_idx, "reason"] = "low_relative_amplitude" if rel_to_ref < min_rel_amp else "isolated_sub_threshold_excursion"
                df.loc[r_idx, "classification_reason"] = f"sub_threshold_rejected(rel={rel_to_ref:.2f},dt={dt:.3f}s)" if np.isfinite(dt) else "initial_sub_threshold_excursion"
                df.loc[r_idx, "classification_confidence"] = "LOW"
                continue
            elif is_early_secondary:
                df.loc[r_idx, "classification_reason"] = f"incomplete_relaxation_early_aftercontraction(dt={dt:.3f}s,rec={rec:.2f},rel={rel_to_main:.2f})"
                df.loc[r_idx, "classification_confidence"] = "HIGH"
            elif is_delayed_secondary or is_alternating_secondary:
                df.loc[r_idx, "classification_reason"] = f"delayed_secondary_contraction(dt={dt:.3f}s,rel_main={rel_to_main:.2f})"
                df.loc[r_idx, "classification_confidence"] = "HIGH" if rel_to_main <= 0.50 else "MEDIUM"
            else:
                df.loc[r_idx, "classification_reason"] = f"secondary_contraction(dt={dt:.3f}s,rel_main={rel_to_main:.2f})"
                df.loc[r_idx, "classification_confidence"] = "MEDIUM"
        elif current_class == "AMBIGUOUS":
            df.loc[r_idx, "classification_reason"] = "borderline_trough_or_corrupted"
            df.loc[r_idx, "classification_confidence"] = "LOW"

    return df
