"""
pixel_counter.sarcomere_qc
==========================

Scientific quality control and vertical-line tracking artifact detection for
cardiomyocyte SarcomereLength recordings (IonOptix CytoCypher / CytoSolver).

Key principles:
1. Source Data Integrity: The raw SarcomereLength array is NEVER modified,
   smoothed, or interpolated in place. Artifacts are represented in a separate
   diagnostic boolean mask (`artifact_mask`) and interval list.
2. Physical Mechanism of Vertical Artifacts:
   - IonOptix FFT sarcomere tracking exhibits ceiling/floor rails (~2.2335 um
     and ~1.1168 um) and abrupt 1-to-2 sample tracking discontinuities where
     |dL| exceeds physiological contraction velocity limits (> 0.050 um/sample
     at 250 Hz, i.e., > 12.5 um/s).
   - Genuine rapid or strong contractions evolve smoothly over multiple samples
     with single-sample steps <= 0.043 um/sample.
3. Strictly Binary Segment QC:
   - PASS: Reliable recording with authentic contractions and usable shortening measurements.
   - REJECT: Recording dominated by tracking failure, vertical-line artifacts,
     missing data, or noise preventing complete segment quantification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.signal import savgol_filter


from dataclasses import dataclass


@dataclass(frozen=True)
class SarcomereMeasurementProfile:
    """Configurable physical and instrument parameters for sarcomere length acquisition."""
    fft_rail_min_um: float = 1.28
    fft_rail_max_um: float = 2.18
    spike_dev_vel_um_s: float = 7.0       # 0.028 um/sample at 250 Hz
    spike_step_vel_um_s: float = 8.0      # 0.032 um/sample at 250 Hz
    shelf_jump1_vel_um_s: float = 13.0    # 0.052 um/sample at 250 Hz
    shelf_jump2_vel_um_s: float = 22.5    # 0.090 um/sample at 250 Hz
    shelf_edge_vel_um_s: float = 11.0     # 0.044 um/sample at 250 Hz
    max_shelf_duration_s: float = 0.35


DEFAULT_SARCOMERE_PROFILE = SarcomereMeasurementProfile()


def estimate_sarcomere_noise_sigma(
    sig_raw: np.ndarray,
    fs: float,
    exclude_mask: Optional[np.ndarray] = None,
) -> float:
    """Estimate robust high-frequency noise floor (um) from first differences."""
    x = np.asarray(sig_raw, dtype=float)
    if len(x) < 5:
        return 0.01
    d1 = np.diff(x)
    if exclude_mask is not None and len(exclude_mask) == len(x):
        valid_step = (~exclude_mask[:-1]) & (~exclude_mask[1:]) & np.isfinite(d1)
        d1 = d1[valid_step]
    else:
        d1 = d1[np.isfinite(d1)]
    if len(d1) < 5:
        return 0.01
    mad = float(np.median(np.abs(d1 - np.median(d1))))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return max(sigma, 1e-5)


def detect_sarcomere_artifacts(
    sig_raw: np.ndarray,
    fs: float,
    t0: float = 0.0,
    profile: Optional[SarcomereMeasurementProfile] = None,
) -> Dict[str, Any]:
    """Identify vertical-line tracking artifacts and FFT rail locks without altering raw data.

    Parameters
    ----------
    sig_raw : np.ndarray
        Authentic 1D SarcomereLength signal in micrometers (um).
    fs : float
        Sampling frequency in Hz.
    t0 : float
        Segment start timestamp in seconds.

    Returns
    -------
    Dict[str, Any]
        Diagnostic artifact dictionary containing `artifact_mask`, `rail_mask`,
        `clusters` (sample index pairs), `artifact_intervals_s`, and summary stats.
    """
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    if n < 5 or fs <= 0:
        empty_mask = np.zeros(n, dtype=bool)
        return {
            "artifact_mask": empty_mask,
            "rail_mask": empty_mask.copy(),
            "spike_mask": empty_mask.copy(),
            "shelf_mask": empty_mask.copy(),
            "clusters": [],
            "artifact_intervals_s": [],
            "n_artifact_clusters": 0,
            "n_artifact_samples": 0,
            "artifact_fraction": 0.0,
            "rail_fraction": 0.0,
            "max_artifact_duration_s": 0.0,
            "sigma_hf": 0.01,
            "p99_abs_d1": 0.0,
        }

    if profile is None:
        profile = DEFAULT_SARCOMERE_PROFILE

    dt = 1.0 / float(fs)

    # 1. Hardware / FFT rail lock or non-finite samples (configurable via SarcomereMeasurementProfile)
    rail_mask = (~np.isfinite(x)) | (x >= profile.fft_rail_max_um) | (x <= profile.fft_rail_min_um)

    # 2. Estimate high-frequency noise sigma excluding rail samples
    sigma_hf = estimate_sarcomere_noise_sigma(x, fs, exclude_mask=rail_mask)

    d1 = np.diff(x, prepend=x[0])
    d1_next = np.diff(x, append=x[-1])
    d1 = np.where(np.isfinite(d1), d1, 0.0)
    d1_next = np.where(np.isfinite(d1_next), d1_next, 0.0)

    # Local reference median (~60 ms window) with rail values replaced by global median
    x_clean_init = x.copy()
    finite_nonrail = x[~rail_mask]
    med_global = float(np.median(finite_nonrail)) if len(finite_nonrail) > 0 else 1.80
    x_clean_init[rail_mask] = med_global
    win_local = max(7, int(round(0.06 * fs)) | 1)
    med_local = ndi.median_filter(x_clean_init, size=win_local)

    # 3. Sharp spike reversals scaled by physical velocity (um/s * dt)
    spike_dev_thr = max(profile.spike_dev_vel_um_s * dt, 7.5 * sigma_hf)
    spike_step_thr = max(profile.spike_step_vel_um_s * dt, 8.0 * sigma_hf)
    max_rev_samples = max(2, int(round(0.012 * fs)))
    spike_mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if abs(x[i] - med_local[i]) > spike_dev_thr:
            step_in = d1[i]
            if abs(step_in) > spike_step_thr:
                for k in range(1, min(max_rev_samples + 1, n - i)):
                    step_out = x[i + k] - x[i + k - 1]
                    if step_in * step_out < 0 and abs(step_out) > 0.65 * spike_step_thr:
                        spike_mask[i : i + k] = True
                        break

    # 4. Boxcar / shelf tracking dropouts scaled by physical velocity (um/s * dt)
    x_nospike = x.copy()
    already_bad = rail_mask | spike_mask
    if np.any(already_bad) and not np.all(already_bad):
        good_i = np.where(~already_bad)[0]
        bad_i = np.where(already_bad)[0]
        x_nospike[bad_i] = np.interp(bad_i, good_i, x[good_i])

    d1_ns = np.diff(x_nospike, prepend=x_nospike[0])
    d2_ns = np.zeros(n, dtype=float)
    d2_ns[2:] = x_nospike[2:] - x_nospike[:-2]

    jump_1_thr = max(profile.shelf_jump1_vel_um_s * dt, 11.0 * sigma_hf)
    jump_2_thr = max(profile.shelf_jump2_vel_um_s * dt, 16.0 * sigma_hf)
    edge_vel_thr = profile.shelf_edge_vel_um_s * dt
    opp_edge_thr = 10.0 * dt

    is_wall_start = (np.abs(d1_ns) > jump_1_thr) | (
        (np.abs(d2_ns) > jump_2_thr) & ((np.abs(d1_ns) > edge_vel_thr) | (np.abs(np.roll(d1_ns, 1)) > edge_vel_thr))
    )
    wall_indices = np.where(is_wall_start)[0]
    shelf_mask = np.zeros(n, dtype=bool)
    max_shelf_samples = int(round(profile.max_shelf_duration_s * fs))

    for idx in wall_indices:
        step_dir = np.sign(d1_ns[idx]) if abs(d1_ns[idx]) >= edge_vel_thr else np.sign(d2_ns[idx])
        if step_dir == 0:
            continue
        start_i = max(0, idx - 1) if (abs(d1_ns[idx]) < jump_1_thr and abs(d2_ns[idx]) > jump_2_thr) else idx
        pre_val = float(x_nospike[max(0, start_i - 1)])
        end_search = min(n, idx + max_shelf_samples)
        for j in range(idx + 1, end_search):
            opp_1 = (d1_ns[j] * step_dir < 0) and (abs(d1_ns[j]) > 0.65 * jump_1_thr)
            opp_2 = (
                (j >= 2)
                and (d2_ns[j] * step_dir < 0)
                and (abs(d2_ns[j]) > 0.65 * jump_2_thr)
                and (abs(d1_ns[j]) > opp_edge_thr)
            )
            if opp_1 or opp_2:
                interior = x_nospike[start_i:j]
                if len(interior) > 0 and float(np.median(step_dir * (interior - pre_val))) > 0.045:
                    shelf_mask[start_i:j] = True
                break

    art_mask = rail_mask | spike_mask | shelf_mask

    # Bridge 1-2 sample gaps between artifact samples only if the bridged sample is also displaced
    if np.any(art_mask):
        closed = ndi.binary_closing(art_mask, structure=np.ones(4, dtype=bool))
        art_mask = art_mask | (closed & (np.abs(x - med_local) > max(0.025, 6.0 * sigma_hf)))
        # Expand by 1 sample on either side only if that neighbor itself is displaced from med_local by > 0.035 um
        neighbor_of_art = ndi.binary_dilation(art_mask, structure=np.ones(3, dtype=bool)) & (~art_mask)
        art_mask = art_mask | (neighbor_of_art & (np.abs(x - med_local) > max(0.038, 8.0 * sigma_hf)))

    # Group contiguous or near-contiguous (gap <= 2 samples if displaced) indices into clusters
    bad_idx = np.where(art_mask)[0]
    clusters: List[Tuple[int, int]] = []
    for idx in bad_idx:
        if not clusters or idx - clusters[-1][1] > 2:
            clusters.append((int(idx), int(idx)))
        else:
            gap_slice = x[clusters[-1][1] + 1 : idx]
            gap_med = med_local[clusters[-1][1] + 1 : idx]
            if len(gap_slice) == 0 or np.all(np.abs(gap_slice - gap_med) > 0.025):
                clusters[-1] = (clusters[-1][0], int(idx))
            else:
                clusters.append((int(idx), int(idx)))

    for c0, c1 in clusters:
        art_mask[c0 : c1 + 1] = True

    intervals_s: List[Tuple[float, float]] = [
        (float(t0 + c0 / fs), float(t0 + c1 / fs)) for c0, c1 in clusters
    ]
    max_dur_s = max([(c1 - c0 + 1) / fs for c0, c1 in clusters], default=0.0)
    p99_abs_d1 = float(np.percentile(np.abs(d1[np.isfinite(d1)]), 99.0)) if np.any(np.isfinite(d1)) else 0.0

    # Re-estimate sigma_hf excluding full artifact mask
    sigma_hf_clean = estimate_sarcomere_noise_sigma(x, fs, exclude_mask=art_mask)

    return {
        "artifact_mask": art_mask,
        "rail_mask": rail_mask,
        "spike_mask": spike_mask,
        "shelf_mask": shelf_mask,
        "clusters": clusters,
        "artifact_intervals_s": intervals_s,
        "n_artifact_clusters": len(clusters),
        "n_artifact_samples": int(np.sum(art_mask)),
        "artifact_fraction": float(np.mean(art_mask)),
        "rail_fraction": float(np.mean(rail_mask)),
        "max_artifact_duration_s": float(max_dur_s),
        "sigma_hf": float(sigma_hf_clean),
        "p99_abs_d1": p99_abs_d1,
    }


def compute_waveform_coherence_and_asymmetry(
    sig_raw: np.ndarray,
    fs: float,
    artifact_mask: np.ndarray,
    df_sheet: Optional[pd.DataFrame] = None,
    y_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute downward-vs-upward excursion asymmetry and row-to-row coherence.

    Genuine sarcomere contractions are downward excursions (dn_p98 >> up_p98),
    whereas random tracking noise fluctuates symmetrically up and down (dn_p98 ~ up_p98)
    with poor row-to-row coherence.
    """
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    valid = (~artifact_mask) & np.isfinite(x)
    if np.sum(valid) < max(25, int(0.5 * fs)):
        return {
            "dn_p98_um": 0.0,
            "up_p98_um": 0.0,
            "excursion_asymmetry": 0.0,
            "median_row_corr": 0.0,
        }

    # Linearly interpolate ONLY across masked artifact samples for internal smoothing check
    x_work = x.copy()
    if not np.all(valid):
        good_idx = np.where(valid)[0]
        bad_idx = np.where(~valid)[0]
        x_work[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])

    win_sg = min(21, (len(x_work) // 2) * 2 - 1)
    if win_sg >= 5:
        xs = savgol_filter(x_work, win_sg, 2)
    else:
        xs = x_work

    # Remove very slow baseline drift (> 2.5 s) via rolling median before measuring asymmetry
    drift_win = max(15, int(round(2.0 * fs)) | 1)
    local_med = ndi.median_filter(xs, size=drift_win)
    dev = (xs - local_med)[valid]

    dn_p98 = float(np.percentile(-dev, 98.0))
    up_p98 = float(np.percentile(dev, 98.0))
    asym = dn_p98 / max(up_p98, 1e-5)

    # Row-to-row coherence if transient rows are provided
    med_corr = 0.0
    if df_sheet is not None and y_cols is not None and len(y_cols) >= 20 and len(df_sheet) >= 2:
        try:
            import warnings
            R = df_sheet[y_cols].to_numpy(dtype=float)
            R_clean = np.where((~np.isfinite(R)) | (R >= 2.18) | (R <= 1.28), np.nan, R)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row_med = np.nanmedian(R_clean, axis=0)
            corrs: List[float] = []
            for r in R_clean:
                m = np.isfinite(r) & np.isfinite(row_med)
                if np.sum(m) > 40 and np.std(r[m]) > 1e-6 and np.std(row_med[m]) > 1e-6:
                    c = float(np.corrcoef(r[m], row_med[m])[0, 1])
                    if np.isfinite(c):
                        corrs.append(c)
            if corrs:
                med_corr = float(np.median(corrs))
        except Exception:
            med_corr = 0.0

    return {
        "dn_p98_um": dn_p98,
        "up_p98_um": up_p98,
        "excursion_asymmetry": float(asym),
        "median_row_corr": float(med_corr),
    }


def evaluate_sarcomere_qc(
    sig_raw: np.ndarray,
    fs: float,
    artifact_info: Dict[str, Any],
    events_df: pd.DataFrame,
    df_sheet: Optional[pd.DataFrame] = None,
    y_cols: Optional[List[str]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Evaluate segment-level scientific QC for a SarcomereLength recording.

    Returns
    -------
    Tuple[str, str, Dict[str, Any]]
        `(qc_status, qc_reason, qc_meta)` where `qc_status` is strictly binary:
        - `"PASS"`
        - `"REJECT"`
    """
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    total_dur_s = float(n / fs) if (n > 0 and fs > 0) else 0.0

    if n == 0 or fs <= 0:
        return "REJECT", "no_valid_sarcomere_signal", {
            "total_duration_s": 0.0,
            "analyzable_duration_s": 0.0,
            "rejection_category": "no_valid_sarcomere_signal",
        }

    art_mask = artifact_info.get("artifact_mask", np.zeros(n, dtype=bool))
    valid_mask = (~art_mask) & np.isfinite(x)
    n_valid = int(np.sum(valid_mask))
    analyzable_dur_s = float(n_valid / fs)

    art_frac = float(artifact_info.get("artifact_fraction", 0.0))
    rail_frac = float(artifact_info.get("rail_fraction", 0.0))
    n_clusters = int(artifact_info.get("n_artifact_clusters", 0))
    max_art_dur_s = float(artifact_info.get("max_artifact_duration_s", 0.0))
    sigma_hf = float(artifact_info.get("sigma_hf", 0.01))
    p99_abs_d1 = float(artifact_info.get("p99_abs_d1", 0.0))

    coherence = compute_waveform_coherence_and_asymmetry(
        x, fs, art_mask, df_sheet=df_sheet, y_cols=y_cols
    )
    dn_p98 = coherence["dn_p98_um"]
    up_p98 = coherence["up_p98_um"]
    asym = coherence["excursion_asymmetry"]
    med_row_corr = coherence["median_row_corr"]

    # Event counts
    n_accepted = 0
    n_corrupted = 0
    max_snr = 0.0
    med_snr = 0.0
    if events_df is not None and not events_df.empty:
        acc_df = events_df[events_df["status"] == "ACCEPTED"]
        n_accepted = len(acc_df)
        if "peak_corrupted" in events_df.columns:
            n_corrupted = int(np.sum(events_df["peak_corrupted"].astype(bool)))
        if n_accepted > 0 and "abs_shortening_um" in acc_df.columns:
            valid_amps = acc_df["abs_shortening_um"].dropna().to_numpy(dtype=float)
            if len(valid_amps) > 0:
                max_snr = float(np.max(valid_amps) / max(sigma_hf, 1e-5))
                med_snr = float(np.median(valid_amps) / max(sigma_hf, 1e-5))

    meta: Dict[str, Any] = {
        "total_duration_s": total_dur_s,
        "analyzable_duration_s": analyzable_dur_s,
        "n_valid_samples": n_valid,
        "artifact_fraction": art_frac,
        "rail_fraction": rail_frac,
        "n_artifact_clusters": n_clusters,
        "max_artifact_duration_s": max_art_dur_s,
        "sigma_hf_um": sigma_hf,
        "p99_abs_d1_um": p99_abs_d1,
        "dn_p98_um": dn_p98,
        "up_p98_um": up_p98,
        "excursion_asymmetry": asym,
        "median_row_corr": med_row_corr,
        "n_accepted_raw": n_accepted,
        "n_corrupted_events": n_corrupted,
        "max_snr": max_snr,
        "median_snr": med_snr,
    }

    # 1. Insufficient valid measurements
    if n_valid < int(1.5 * fs) or analyzable_dur_s < 1.5:
        meta["rejection_category"] = "insufficient_valid_measurements"
        return "REJECT", "insufficient_valid_measurements", meta

    # 2. Severe tracking failure / rail lock / dominating vertical lines
    if (
        rail_frac > 0.035
        or art_frac > 0.14
        or max_art_dur_s > 0.80
        or (rail_frac > 0.020 and n_clusters >= 10)
        or (art_frac > 0.08 and n_clusters >= 10 and med_row_corr < 0.70)
    ):
        meta["rejection_category"] = "tracking_artifact_dominated"
        return "REJECT", "tracking_artifact_dominated", meta

    # 3. Excessive high-frequency tracking noise
    if sigma_hf > 0.0080:
        meta["rejection_category"] = "noise_dominated"
        return "REJECT", "noise_dominated", meta

    # 4. Zero accepted contractions: distinguish clean quiescent recording from artifact/noise domination
    if n_accepted == 0:
        if n_corrupted > 0 or art_frac > 0.02 or rail_frac > 0.005:
            meta["rejection_category"] = "tracking_artifact_dominated"
            return "REJECT", "tracking_artifact_dominated", meta
        n_cands_zero = len(events_df) if (events_df is not None and not events_df.empty) else 0
        if sigma_hf <= 0.0012 and dn_p98 <= 0.0040 and n_cands_zero == 0 and n_clusters == 0:
            meta["quiescent"] = True
            return "PASS", "clean_quiescent_no_contractions", meta
        meta["rejection_category"] = "noise_dominated"
        return "REJECT", "noise_dominated", meta

    # 5. Low-SNR / symmetric random-walk noise domination check
    if asym < 1.65 and med_row_corr < 0.75 and (dn_p98 < 0.045 or med_snr < 12.0 or asym < 1.35):
        meta["rejection_category"] = "noise_dominated"
        return "REJECT", "noise_dominated", meta

    if dn_p98 < 0.0055 and med_row_corr < 0.75:
        meta["rejection_category"] = "noise_dominated"
        return "REJECT", "noise_dominated", meta

    if n_accepted >= 3 and dn_p98 < 0.016 and asym < 1.80 and med_row_corr < 0.78:
        meta["rejection_category"] = "noise_dominated"
        return "REJECT", "noise_dominated", meta

    if events_df is not None and not events_df.empty:
        n_noise_rej = int(np.sum(events_df["reason"] == "low_prominence"))
        if n_accepted < 5 and sigma_hf > 0.0020 and n_noise_rej >= n_accepted:
            meta["rejection_category"] = "noise_dominated"
            return "REJECT", "noise_dominated", meta

    # 6. Excessive reconstruction dependence (count-invariant across all recordings)
    n_rec_ivs = len(artifact_info.get("reconstructed_intervals", []))
    if n_accepted > 0 and n_rec_ivs >= 8 and (n_rec_ivs / n_accepted >= 0.75) and rail_frac >= 0.008:
        meta["rejection_category"] = "tracking_artifact_dominated"
        return "REJECT", "excessive_reconstruction_dependence", meta

    # 7. Uncorrupted event coverage check
    if n_corrupted > 0:
        n_clean = n_accepted - n_corrupted
        is_excessive_corruption = (
            art_frac > 0.08
            or rail_frac > 0.025
            or (n_corrupted / max(1, n_accepted) > 0.60)
            or (n_clean == 0)
        )
        if is_excessive_corruption:
            meta["rejection_category"] = "tracking_artifact_dominated"
            return "REJECT", "tracking_artifact_dominated", meta

    return "PASS", "valid_contractions", meta
