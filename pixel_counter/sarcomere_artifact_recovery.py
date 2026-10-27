"""
pixel_counter.sarcomere_artifact_recovery
=========================================

Second-pass, conservative sarcomere-only reconstruction of authentic
mechanical contractions corrupted by narrow vertical tracking artifacts.

Scientific Principles:
1. Zero Modification of Raw Data:
   The original experimental SarcomereLength array L(t) is NEVER modified.
   Reconstruction occurs exclusively within a temporary internal working array
   used for peak localization.
2. Minimal Local Shape-Preserving Reconstruction:
   Only localized, narrow tracking spikes (<= 40 ms / <= 10 samples at 250 Hz)
   bounded by reliable contraction-shaped pre/post samples are interpolated
   using monotonic PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
   to prevent artificial oscillations or overshoots.
3. Zero Mechanical Amplitude Fabrication:
   Contractions whose troughs are interrupted by tracking artifacts have their
   amplitudes reported strictly as NaN. Reconstructed amplitudes are never
   passed off as biological measurements.
4. Amplitude Hierarchy Defense:
   A reconstructed candidate may be classified as MAIN or SECONDARY only if
   the authentic measured shoulder excursion bounding the artifact proves that
   the contraction depth exceeded the corresponding threshold (A_shoulder >= T).
   Otherwise the candidate remains unclassified or rejected.
5. Independent of Calcium:
   Operates strictly on sarcomere mechanics without accessing calcium data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def reconstruct_narrow_tracking_artifacts(
    sig_raw: np.ndarray,
    fs: float,
    artifact_info: Dict[str, Any],
    max_gap_s: float = 0.040,
    t0: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Create a temporary detection-only reconstruction array bridging narrow tracking spikes.

    Parameters
    ----------
    sig_raw : np.ndarray
        Untouched authentic SarcomereLength array (um).
    fs : float
        Sampling frequency in Hz.
    artifact_info : dict
        Artifact metadata containing 'artifact_mask' and 'clusters' from detect_sarcomere_artifacts.
    max_gap_s : float, default 0.040
        Maximum duration (s) of localized artifact eligible for conservative bridging.

    Returns
    -------
    sig_recon : np.ndarray
        Temporary working array for peak localization (sig_raw is untouched).
    recon_meta : dict
        Audit trail of reconstructed intervals and provenance.
    """
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    sig_recon = x.copy()

    art_mask = artifact_info.get("artifact_mask", np.zeros(n, dtype=bool))
    clusters = artifact_info.get("clusters", [])

    valid_mask = (~art_mask) & np.isfinite(x) & (x > 0.5)

    reconstructed_intervals: List[Dict[str, Any]] = []
    unrecoverable_intervals: List[Dict[str, Any]] = []

    # If the recording has widespread tracking failure (> 12% artifacts), abort reconstruction
    art_frac = float(np.mean(art_mask)) if n > 0 else 1.0
    if art_frac > 0.12 or len(clusters) > 40:
        return sig_recon, {
            "reconstructed_intervals": [],
            "unrecoverable_intervals": clusters,
            "reconstructed_samples_count": 0,
            "reconstruction_eligible": False,
            "reason": "excessive_tracking_failure_coverage",
        }

    max_gap_samples = max(1, int(round(max_gap_s * fs)))

    for c0, c1 in clusters:
        cluster_len = c1 - c0 + 1
        dur_s = cluster_len / fs

        if cluster_len > max_gap_samples:
            unrecoverable_intervals.append({
                "start_idx": c0,
                "end_idx": c1,
                    "start_time_s": float(t0 + c0 / fs),
                    "end_time_s": float(t0 + c1 / fs),
                    "duration_ms": float(dur_s * 1000.0),
                "duration_s": dur_s,
                "reason": "duration_exceeds_max_gap",
            })
            continue

        # Look for valid anchor samples immediately before and after the artifact
        left_valid = np.where(valid_mask[:c0])[0]
        right_valid = np.where(valid_mask[c1 + 1:])[0] + (c1 + 1)

        if len(left_valid) < 2 or len(right_valid) < 2:
            unrecoverable_intervals.append({
                "start_idx": c0,
                "end_idx": c1,
                    "start_time_s": float(t0 + c0 / fs),
                    "end_time_s": float(t0 + c1 / fs),
                    "duration_ms": float(dur_s * 1000.0),
                "duration_s": dur_s,
                "reason": "insufficient_boundary_anchors",
            })
            continue

        # Use up to 4 points on either side for local Hermite polynomial
        n_anchor = min(4, len(left_valid), len(right_valid))
        pts_x = np.concatenate([left_valid[-n_anchor:], right_valid[:n_anchor]])
        pts_y = x[pts_x]

        try:
            pchip = PchipInterpolator(pts_x, pts_y)
            interp_range = np.arange(c0, c1 + 1)
            interp_vals = pchip(interp_range)

            # Safety bounds: interpolated values should not run away into negative or infinite space
            min_anchor = np.min(pts_y)
            max_anchor = np.max(pts_y)
            # Allow downward contraction extension up to 0.15 um below anchor minimum, but no upward spikes
            if np.all(np.isfinite(interp_vals)) and np.all(interp_vals >= min_anchor - 0.20) and np.all(interp_vals <= max_anchor + 0.05):
                sig_recon[interp_range] = interp_vals
                reconstructed_intervals.append({
                    "start_idx": c0,
                    "end_idx": c1,
                    "start_time_s": float(t0 + c0 / fs),
                    "end_time_s": float(t0 + c1 / fs),
                    "duration_ms": float(dur_s * 1000.0),
                    "duration_s": dur_s,
                    "samples_count": cluster_len,
                    "anchor_x": pts_x.tolist(),
                    "anchor_y": pts_y.tolist(),
                    "original_vals": x[interp_range].tolist(),
                    "reconstructed_vals": interp_vals.tolist(),
                    "method": "PCHIP_local_hermite",
                })
            else:
                unrecoverable_intervals.append({
                    "start_idx": c0,
                    "end_idx": c1,
                    "start_time_s": float(t0 + c0 / fs),
                    "end_time_s": float(t0 + c1 / fs),
                    "duration_ms": float(dur_s * 1000.0),
                    "duration_s": dur_s,
                    "reason": "interpolated_values_exceeded_safety_bounds",
                })
        except Exception as exc:
            unrecoverable_intervals.append({
                "start_idx": c0,
                "end_idx": c1,
                    "start_time_s": float(t0 + c0 / fs),
                    "end_time_s": float(t0 + c1 / fs),
                    "duration_ms": float(dur_s * 1000.0),
                "duration_s": dur_s,
                "reason": f"interpolation_exception_{exc}",
            })

    total_recon_samples = sum(iv["samples_count"] for iv in reconstructed_intervals)
    return sig_recon, {
        "reconstructed_intervals": reconstructed_intervals,
        "unrecoverable_intervals": unrecoverable_intervals,
        "reconstructed_samples_count": total_recon_samples,
        "reconstruction_eligible": True,
        "n_reconstructed_intervals": len(reconstructed_intervals),
        "n_unrecoverable_intervals": len(unrecoverable_intervals),
    }


def reconcile_reconstructed_events(
    ev_df_recon: pd.DataFrame,
    sig_raw: np.ndarray,
    fs: float,
    artifact_info: Dict[str, Any],
    recon_meta: Dict[str, Any],
    t_main_pct: float,
    t_sce_pct: float,
    l0_curve: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Reconcile second-pass candidate detections with raw measurement integrity.

    Ensures:
    1. Peaks corrupted by artifacts preserve NaN amplitude (zero fabrication).
    2. Event classification (MAIN/SECONDARY) is permitted ONLY if measured
       shoulder excursions bound the contraction depth above threshold.
    3. Completely uncorrupted peaks retain their true measured amplitudes.
    """
    if ev_df_recon is None or ev_df_recon.empty:
        return ev_df_recon

    ev_df = ev_df_recon.copy()
    x = np.asarray(sig_raw, dtype=float)
    n = len(x)

    recon_intervals = recon_meta.get("reconstructed_intervals", [])
    # Build a lookup set of sample indices that underwent reconstruction
    recon_sample_set = set()
    for iv in recon_intervals:
        recon_sample_set.update(range(iv["start_idx"], iv["end_idx"] + 1))

    # Build lookup set of all original artifact samples
    art_mask = artifact_info.get("artifact_mask", np.zeros(n, dtype=bool))
    art_sample_set = set(np.where(art_mask)[0])

    if "display_amplitude" not in ev_df.columns:
        ev_df["display_amplitude"] = ev_df["pct_shortening"]

    for idx in ev_df.index:
        pk_idx = int(ev_df.loc[idx, "peak_index"])
        raw_val = float(x[pk_idx]) if pk_idx < n else np.nan
        # A trough nadir is corrupted if the nadir (+/- 1 sample) was reconstructed,
        # or if the nadir sample itself is an artifact or pegged to rail.
        # Zero amplitude fabrication is preserved for reconstructed troughs.
        pk_neighborhood = {pk_idx - 1, pk_idx, pk_idx + 1}
        is_in_recon = bool(pk_neighborhood & recon_sample_set)
        is_nadir_art = bool(
            pk_idx in art_sample_set
            or not np.isfinite(raw_val)
            or raw_val <= 1.15
            or raw_val >= 2.18
        )
        is_corrupted_nadir = bool(is_in_recon or is_nadir_art)

        if is_corrupted_nadir:
            # Corrupted trough: preserve zero amplitude fabrication
            ev_df.loc[idx, "peak_corrupted"] = True
            ev_df.loc[idx, "has_valid_amplitude"] = False
            ev_df.loc[idx, "recovery_status"] = "RECONSTRUCTED_ARTIFACT"
            
            # Check authentic shoulder samples (within +/- 80 ms)
            w_sh = max(5, int(0.080 * fs))
            pre_valid = [s for s in range(max(0, pk_idx - w_sh), pk_idx) if s not in art_sample_set and np.isfinite(x[s])]
            post_valid = [s for s in range(pk_idx + 1, min(n, pk_idx + w_sh + 1)) if s not in art_sample_set and np.isfinite(x[s])]

            l0_val = float(l0_curve[pk_idx]) if l0_curve is not None and len(l0_curve) > pk_idx else float(np.median(x))
            
            # If genuine mechanical excursion is established on authentic shoulders
            if pre_valid and post_valid:
                # Lowest SarcomereLength observed on intact shoulders
                min_shoulder_l = min(np.min(x[pre_valid]), np.min(x[post_valid]))
                # True contraction minimum is at least as deep as the intact shoulders:
                # L_min <= min_shoulder_l => Shortening_amp >= (L0 - min_shoulder_l) / L0
                bound_short_pct = 100.0 * (l0_val - min_shoulder_l) / max(0.5, l0_val)
                
                # Zero fabrication: physical metric amplitudes remain NaN
                ev_df.loc[idx, "abs_shortening_um"] = np.nan
                ev_df.loc[idx, "pct_shortening"] = np.nan
                # Display amplitude provided for Plotly scatter rendering
                ev_df.loc[idx, "display_amplitude"] = float(bound_short_pct)

                if bound_short_pct >= t_main_pct - 1e-6:
                    ev_df.loc[idx, "status"] = "ACCEPTED"
                    ev_df.loc[idx, "event_class"] = "MAIN"
                    ev_df.loc[idx, "reason"] = "reconstructed_artifact_bounded_main"
                elif bound_short_pct >= t_sce_pct - 1e-6:
                    ev_df.loc[idx, "status"] = "ACCEPTED"
                    ev_df.loc[idx, "event_class"] = "SECONDARY"
                    ev_df.loc[idx, "reason"] = "reconstructed_artifact_bounded_secondary"
                else:
                    ev_df.loc[idx, "status"] = "REJECTED"
                    ev_df.loc[idx, "event_class"] = "REJECTED"
                    ev_df.loc[idx, "reason"] = "reconstructed_shoulder_below_threshold"
            else:
                # No reliable intact shoulders
                ev_df.loc[idx, "abs_shortening_um"] = np.nan
                ev_df.loc[idx, "pct_shortening"] = np.nan
                ev_df.loc[idx, "display_amplitude"] = np.nan
                ev_df.loc[idx, "status"] = "REJECTED"
                ev_df.loc[idx, "event_class"] = "REJECTED"
                ev_df.loc[idx, "reason"] = "unsupported_artifact_trough"
        else:
            # Fully measured authentic contraction
            ev_df.loc[idx, "peak_corrupted"] = False
            ev_df.loc[idx, "has_valid_amplitude"] = True
            ev_df.loc[idx, "recovery_status"] = "DIRECT_MEASUREMENT"
            ev_df.loc[idx, "display_amplitude"] = float(ev_df.loc[idx, "pct_shortening"])
            # Ensure strict amplitude hierarchy on measured values
            amp = ev_df.loc[idx, "pct_shortening"]
            if pd.notna(amp):
                if amp < t_sce_pct - 1e-6:
                    ev_df.loc[idx, "status"] = "REJECTED"
                    ev_df.loc[idx, "event_class"] = "REJECTED"
                    ev_df.loc[idx, "reason"] = "below_secondary_threshold"
                elif amp < t_main_pct - 1e-6:
                    ev_df.loc[idx, "event_class"] = "SECONDARY"
                else:
                    ev_df.loc[idx, "event_class"] = "MAIN"

    return ev_df
