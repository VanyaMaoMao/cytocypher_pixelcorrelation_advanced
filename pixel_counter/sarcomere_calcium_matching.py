"""
pixel_counter.sarcomere_calcium_matching
========================================

Electromechanical event matching and calcium-assisted analysis module
for dual-channel cardiomyocyte recordings (SarcomereLength + RatioMetricCalcium).

Scientific & Signal-Processing Principles:
1. Channel Integrity & True Timestamps:
   Matches authentic events on their original continuous experimental time axes
   without artificial resampling, grid snapping, or curve shifting.
2. Temporal Overlap & Window Gating:
   Restricts matching strictly to the verified temporal overlap interval:
   [max(t0_sarc, t0_cal), min(tend_sarc, tend_cal)].
3. Electromechanical Delay (tau_EMD):
   Calcium activation precedes mechanical shortening (calcium peak occurs prior
   to sarcomere minimum length trough). The characteristic recording delay tau_EMD
   is estimated robustly as the median delay among unambiguous, high-confidence
   event pairs, with explicit configurable bounds to prevent pairing unrelated beats.
4. Monotonic 1-to-1 Matching:
   Uses dynamic programming to solve global optimal non-crossing 1-to-1 bipartite
   matching, ensuring each calcium event matches at most one sarcomere event,
   each sarcomere event matches at most one calcium event, and chronological
   order is strictly preserved.
5. Strict Scientific Guardrails (Zero Fabrication):
   - Calcium NEVER invents mechanical events.
   - Calcium NEVER generates shortening amplitudes when mechanical measurements are
     corrupted by artifacts (NaN is preserved).
   - Unmatched mechanical contractions are retained (not discarded).
   - Unmatched calcium transients are retained in separate diagnostic records.
   - A calcium match does NOT automatically reclassify an event as MAIN (MAIN vs
     SECONDARY remains grounded in mechanical takeoff and relaxation dynamics).
   - Calcium assistance cannot rescue a segment where sarcomere QC is REJECT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SarcomereCalciumMatchingParams:
    """Parameters for electromechanical event matching and calcium assistance."""
    min_delay_s: float = -0.060             # Minimum physiological delay tau = t_sarc - t_cal (seconds)
    max_delay_s: float = 0.240              # Maximum physiological delay tau = t_sarc - t_cal (seconds)
    canonical_delay_window_s: float = 0.095 # Max deviation from tau_EMD for 1-to-1 pairing (seconds)
    high_conf_delay_tol_s: float = 0.035    # Residual tolerance for HIGH matching confidence (seconds)
    medium_conf_delay_tol_s: float = 0.070  # Residual tolerance for MEDIUM matching confidence (seconds)
    min_pairs_for_emd: int = 2              # Min unambiguous pairs to estimate characteristic tau_EMD
    default_prior_emd_s: float = 0.030      # Prior median EMD delay fallback when unresolvable (seconds)
    borderline_prom_factor: float = 0.65    # Fraction of eff_prom_floor allowed for borderline recovery
    borderline_abs_factor: float = 0.65     # Fraction of eff_abs_floor allowed for borderline recovery
    onset_threshold_frac: float = 0.20      # Excursion fraction to localize transient takeoff/onset


def compute_temporal_overlap(
    t0_sarc: float,
    fs_sarc: float,
    n_sarc: int,
    t0_cal: float,
    fs_cal: float,
    n_cal: int,
) -> Tuple[float, float, float]:
    """Compute verified continuous temporal overlap interval between two channels."""
    if n_sarc <= 0 or n_cal <= 0 or fs_sarc <= 0 or fs_cal <= 0:
        return 0.0, 0.0, 0.0
    t_start_s = float(t0_sarc)
    t_end_s = float(t0_sarc + (n_sarc - 1) / fs_sarc)
    t_start_c = float(t0_cal)
    t_end_c = float(t0_cal + (n_cal - 1) / fs_cal)

    t_start = max(t_start_s, t_start_c)
    t_end = min(t_end_s, t_end_c)
    dur = max(0.0, t_end - t_start)
    return t_start, t_end, dur


def estimate_transient_onset(
    sig: np.ndarray,
    fs: float,
    t0: float,
    peak_idx: int,
    baseline_val: float,
    polarity: str = "positive",
    frac: float = 0.20,
    search_window_s: float = 0.35,
) -> float:
    """Estimate physiological onset (takeoff) time of a transient excursion."""
    if peak_idx < 0 or peak_idx >= len(sig) or fs <= 0:
        return np.nan

    n = len(sig)
    search_samples = max(2, int(round(search_window_s * fs)))
    start_idx = max(0, peak_idx - search_samples)
    seg = sig[start_idx : peak_idx + 1]
    if len(seg) < 2:
        return float(t0 + peak_idx / fs)

    if polarity == "positive":
        # Fluorescence: baseline to peak positive excursion
        y_peak = float(sig[peak_idx])
        exc = y_peak - baseline_val
        if exc <= 1e-4:
            return float(t0 + peak_idx / fs)
        target = baseline_val + frac * exc
        # Search backward from peak to find where signal crosses below target
        cross_idx = None
        for i in range(len(seg) - 1, -1, -1):
            if seg[i] <= target:
                cross_idx = start_idx + i
                break
        if cross_idx is not None:
            return float(t0 + cross_idx / fs)
        return float(t0 + start_idx / fs)
    else:
        # Sarcomere: baseline to trough negative excursion
        y_trough = float(sig[peak_idx])
        exc = baseline_val - y_trough
        if exc <= 1e-4:
            return float(t0 + peak_idx / fs)
        target = baseline_val - frac * exc
        cross_idx = None
        for i in range(len(seg) - 1, -1, -1):
            if seg[i] >= target:
                cross_idx = start_idx + i
                break
        if cross_idx is not None:
            return float(t0 + cross_idx / fs)
        return float(t0 + start_idx / fs)


def estimate_electromechanical_delay(
    sarc_acc_df: pd.DataFrame,
    cal_acc_df: pd.DataFrame,
    params: SarcomereCalciumMatchingParams,
) -> Tuple[Optional[float], str]:
    """Robust estimation of characteristic electromechanical delay tau_EMD.

    Finds unambiguous accepted pairs where a single calcium transient falls
    within [min_delay_s, max_delay_s] of an uncorrupted sarcomere contraction.
    """
    if sarc_acc_df.empty or cal_acc_df.empty:
        return None, "UNRESOLVED"

    s_uncorrupted = sarc_acc_df[~sarc_acc_df["peak_corrupted"].astype(bool)]
    if s_uncorrupted.empty:
        return None, "UNRESOLVED"

    s_times = s_uncorrupted["time_s"].to_numpy(dtype=float)
    c_times = cal_acc_df["time_s"].to_numpy(dtype=float)

    candidate_delays: List[float] = []
    for st in s_times:
        diffs = st - c_times
        valid = diffs[(diffs >= params.min_delay_s) & (diffs <= params.max_delay_s)]
        if len(valid) == 1:
            candidate_delays.append(float(valid[0]))

    if len(candidate_delays) >= params.min_pairs_for_emd:
        return float(np.median(candidate_delays)), "ESTIMATED"
    elif len(candidate_delays) == 1:
        return float(candidate_delays[0]), "SINGLE_PAIR"
    else:
        # Fallback to prior default if reasonable candidate pairs exist in broad window
        all_diffs = []
        for st in s_times:
            d = st - c_times
            v = d[(d >= params.min_delay_s) & (d <= params.max_delay_s)]
            all_diffs.extend(v.tolist())
        if len(all_diffs) > 0:
            return float(np.median(all_diffs)), "PRIOR_FALLBACK"
        return params.default_prior_emd_s, "PRIOR_FALLBACK"


def monotonic_bipartite_match(
    s_times: np.ndarray,
    c_times: np.ndarray,
    tau_emd: float,
    max_dev: float = 0.095,
) -> List[Tuple[int, int, float]]:
    """Globally optimal monotonic 1-to-1 matching via Dynamic Programming.

    guarantees:
    - 1-to-1: each sarcomere event matches at most one calcium event.
    - monotonic: chronological order is strictly preserved (no crossing matches).
    - residual minimization: minimizes deviation |(ts - tc) - tau_emd|.

    Returns list of tuples: (s_idx, c_idx, delay_s).
    """
    n_s = len(s_times)
    n_c = len(c_times)
    if n_s == 0 or n_c == 0:
        return []

    INF = 1e9
    cost = np.full((n_s, n_c), INF, dtype=float)
    for i in range(n_s):
        ts = s_times[i]
        for j in range(n_c):
            tc = c_times[j]
            delay = ts - tc
            dev = abs(delay - tau_emd)
            if dev <= max_dev:
                cost[i, j] = dev

    MATCH_REWARD = -1000.0
    dp = np.zeros((n_s + 1, n_c + 1), dtype=float)

    for i in range(1, n_s + 1):
        for j in range(1, n_c + 1):
            # Option 1: s[i-1] unmatched
            best = dp[i - 1, j]
            # Option 2: c[j-1] unmatched
            if dp[i, j - 1] < best:
                best = dp[i, j - 1]
            # Option 3: match s[i-1] with c[j-1]
            if cost[i - 1, j - 1] < INF:
                m_cost = dp[i - 1, j - 1] + MATCH_REWARD + cost[i - 1, j - 1]
                if m_cost < best:
                    best = m_cost
            dp[i, j] = best

    # Backtracking
    matches: List[Tuple[int, int, float]] = []
    i, j = n_s, n_c
    while i > 0 and j > 0:
        c_ij = cost[i - 1, j - 1]
        if c_ij < INF and abs(dp[i, j] - (dp[i - 1, j - 1] + MATCH_REWARD + c_ij)) < 1e-6:
            delay_val = float(s_times[i - 1] - c_times[j - 1])
            matches.append((i - 1, j - 1, delay_val))
            i -= 1
            j -= 1
        elif abs(dp[i, j] - dp[i - 1, j]) < 1e-6:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def evaluate_combined_qc(
    sarc_qc_status: str,
    sarc_qc_reason: str,
    cal_qc_status: str,
    cal_qc_reason: str,
    overlap_dur_s: float,
) -> Dict[str, Any]:
    """Evaluate combined dual-channel quality control status."""
    if overlap_dur_s < 1.0:
        return {
            "combined_qc_status": sarc_qc_status,
            "combined_qc_reason": "insufficient_temporal_overlap",
            "description": f"Continuous temporal overlap ({overlap_dur_s:.2f}s) is below 1.0s threshold.",
        }

    if sarc_qc_status == "PASS" and cal_qc_status == "PASS":
        return {
            "combined_qc_status": "PASS",
            "combined_qc_reason": "both_channels_valid",
            "description": "Both sarcomere shortening and calcium transient recordings are valid.",
        }

    if sarc_qc_status == "PASS" and cal_qc_status == "REJECT":
        return {
            "combined_qc_status": "PASS",
            "combined_qc_reason": "sarcomere_valid_calcium_rejected",
            "description": f"Sarcomere signal is valid, but calcium channel failed QC ({cal_qc_reason}).",
        }

    if sarc_qc_status == "PASS" and cal_qc_status == "UNAVAILABLE":
        return {
            "combined_qc_status": "PASS",
            "combined_qc_reason": "sarcomere_valid_calcium_unavailable",
            "description": "Sarcomere signal is valid; calcium channel data is unavailable.",
        }

    if sarc_qc_status == "REJECT" and cal_qc_status == "PASS":
        return {
            "combined_qc_status": "REJECT",
            "combined_qc_reason": "sarcomere_invalid_calcium_valid",
            "description": "Sarcomere channel failed QC; calcium cannot rescue corrupted mechanical measurements.",
        }

    return {
        "combined_qc_status": "REJECT",
        "combined_qc_reason": "both_channels_invalid",
        "description": f"Both channels failed QC (Sarc: {sarc_qc_reason}, Cal: {cal_qc_reason}).",
    }


def match_sarcomere_and_calcium(
    sarc_events_df: pd.DataFrame,
    cal_events_df: pd.DataFrame,
    fs_sarc: float,
    fs_cal: float,
    t0_sarc: float,
    t0_cal: float,
    n_sarc: int,
    n_cal: int,
    sarc_qc: Tuple[str, str, Dict[str, Any]],
    cal_qc: Tuple[str, str, Dict[str, Any]],
    sig_sarc_raw: Optional[np.ndarray] = None,
    sig_cal_raw: Optional[np.ndarray] = None,
    params: Optional[SarcomereCalciumMatchingParams] = None,
    enable_assistance: bool = False,
    threshold_main_pct: Optional[float] = None,
    threshold_sce_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Execute independent electromechanical event matching without calcium assistance.

    Scientific and Invariant Principles:
    1. Channel Invariance Invariant: Calcium NEVER creates mechanical events, NEVER rescues
       rejected candidates, NEVER alters sarcomere MAIN/SCE classifications, and NEVER
       promotes sarcomere QC status.
    2. Informational Pairing Only: Dynamic programming identifies 1-to-1 pairings between
       independently detected sarcomere and calcium events to estimate electromechanical delay (tau_EMD).
    3. Zero Fabrication: Mechanical amplitudes, timestamps, and classifications are 100% identical
       to the sarcomere-only result.
    """
    if params is None:
        params = SarcomereCalciumMatchingParams()

    sarc_qc_st, sarc_qc_rs, sarc_qc_meta = sarc_qc
    cal_qc_st, cal_qc_rs, cal_qc_meta = cal_qc

    # Compute continuous temporal overlap
    t_start_ov, t_end_ov, overlap_dur = compute_temporal_overlap(
        t0_sarc, fs_sarc, n_sarc, t0_cal, fs_cal, n_cal
    )

    combined_qc = evaluate_combined_qc(
        sarc_qc_st, sarc_qc_rs, cal_qc_st, cal_qc_rs, overlap_dur
    )

    # Initialize enriched output DataFrames
    sarc_df = sarc_events_df.copy() if sarc_events_df is not None and not sarc_events_df.empty else pd.DataFrame()
    cal_df = cal_events_df.copy() if cal_events_df is not None and not cal_events_df.empty else pd.DataFrame()

    # Pre-populate required schema columns on sarcomere events
    for col, default_val in [
        ("calcium_match_status", "UNAVAILABLE"),
        ("matched_calcium_time_s", np.nan),
        ("calcium_amplitude", np.nan),
        ("calcium_mechanical_delay_s", np.nan),
        ("calcium_onset_delay_s", np.nan),
        ("calcium_matching_confidence", "N/A"),
        ("sarcomere_only_classification", None),
        ("assisted_classification", None),
        ("assistance_reason", "independent_analysis"),
        ("event_nature", "MEASURED"),
        ("recovery_status", "MEASURED"),
        ("has_valid_amplitude", True),
    ]:
        if col not in sarc_df.columns:
            sarc_df[col] = pd.Series(default_val, index=sarc_df.index, dtype=object if isinstance(default_val, (str, bool)) else float)

    # Store baseline Sarcomere-Only classifications and status
    if not sarc_df.empty:
        sarc_df["sarcomere_only_classification"] = sarc_df["event_class"]
        sarc_df["assisted_classification"] = sarc_df["event_class"]
        sarc_df["sarcomere_only_status"] = sarc_df["status"]
        for idx in sarc_df.index:
            corrupt = bool(sarc_df.loc[idx, "peak_corrupted"])
            st = str(sarc_df.loc[idx, "status"])
            curr_recov = str(sarc_df.loc[idx, "recovery_status"])
            sarc_df.loc[idx, "has_valid_amplitude"] = (st == "ACCEPTED" and not corrupt)
            if corrupt:
                if curr_recov != "RECONSTRUCTED_ARTIFACT":
                    sarc_df.loc[idx, "recovery_status"] = "CORRUPTED_TROUGH"
                sarc_df.loc[idx, "event_nature"] = "RECONSTRUCTED" if curr_recov == "RECONSTRUCTED_ARTIFACT" else "UNRESOLVED"
            elif st == "ACCEPTED":
                sarc_df.loc[idx, "event_nature"] = "MEASURED"
                sarc_df.loc[idx, "recovery_status"] = "MEASURED"
            else:
                sarc_df.loc[idx, "event_nature"] = "UNRESOLVED"
                sarc_df.loc[idx, "recovery_status"] = "UNRESOLVED"

    sarc_only_copy = sarc_df.copy()

    # Pre-populate calcium match columns
    for col, default_val in [
        ("calcium_match_status", "UNAVAILABLE"),
        ("matched_sarcomere_time_s", np.nan),
        ("matched_sarcomere_index", np.nan),
        ("sarcomere_amplitude_um", np.nan),
        ("recovery_status", "MEASURED"),
    ]:
        if col not in cal_df.columns:
            cal_df[col] = pd.Series(default_val, index=cal_df.index, dtype=object if isinstance(default_val, str) else float)

    # If calcium is invalid/unavailable or no overlap, return early with structured results
    calcium_usable = (cal_qc_st == "PASS") and (overlap_dur >= 1.0) and not cal_df.empty

    if not calcium_usable:
        reason_unavail = "insufficient_temporal_overlap" if overlap_dur < 1.0 else f"calcium_{cal_qc_st.lower()}"
        if not sarc_df.empty:
            sarc_df["calcium_match_status"] = "UNAVAILABLE"
            sarc_df["assistance_reason"] = f"calcium_unavailable_{reason_unavail}"
        if not cal_df.empty:
            cal_df["calcium_match_status"] = "UNAVAILABLE"

        n_acc_s = int(np.sum(sarc_df["status"] == "ACCEPTED")) if not sarc_df.empty else 0
        n_meas_s = int(np.sum((sarc_df["status"] == "ACCEPTED") & sarc_df["has_valid_amplitude"])) if not sarc_df.empty else 0
        return {
            "sarc_events": sarc_df,
            "sarc_only_events": sarc_only_copy,
            "calcium_events": cal_df,
            "matched_pairs": [],
            "unmatched_sarcomere_events": sarc_df[sarc_df["status"] == "ACCEPTED"].copy() if not sarc_df.empty else pd.DataFrame(),
            "unmatched_calcium_events": cal_df[cal_df["status"] == "ACCEPTED"].copy() if not cal_df.empty else pd.DataFrame(),
            "estimated_emd_delay_s": None,
            "emd_delay_status": "UNRESOLVED",
            "n_matched_pairs": 0,
            "n_unmatched_sarc": n_acc_s,
            "n_unmatched_cal": int(np.sum(cal_df["status"] == "ACCEPTED")) if not cal_df.empty else 0,
            "n_directly_measured": n_meas_s,
            "n_calcium_assisted": 0,
            "n_valid_amplitude": n_meas_s,
            "n_lacking_amplitude": n_acc_s - n_meas_s,
            "n_unresolved": int(np.sum(cal_df["status"] == "ACCEPTED")) if not cal_df.empty else 0,
            "n_classifications_changed": 0,
            "n_borderline_recovered": 0,
            "n_artifact_corrupted": int(np.sum(sarc_df["peak_corrupted"].astype(bool))) if not sarc_df.empty else 0,
            "enable_assistance": False,
            "temporal_overlap": {
                "t_start_s": t_start_ov,
                "t_end_s": t_end_ov,
                "overlap_duration_s": overlap_dur,
            },
            "combined_qc": combined_qc,
        }

    # Calcium is valid and analyzable: Proceed with electromechanical delay estimation
    cal_acc = cal_df[cal_df["status"] == "ACCEPTED"].copy()
    sarc_acc = sarc_df[sarc_df["status"] == "ACCEPTED"].copy()

    tau_emd, emd_status = estimate_electromechanical_delay(sarc_acc, cal_acc, params)
    eff_tau_emd = tau_emd if tau_emd is not None else params.default_prior_emd_s

    # Include candidate events in sarcomere pool for pairing
    in_overlap_s = (sarc_df["time_s"] >= t_start_ov - 0.1) & (sarc_df["time_s"] <= t_end_ov + 0.1)
    sarc_pool = sarc_df[in_overlap_s & (sarc_df["status"] == "ACCEPTED")].copy()

    in_overlap_c = (cal_df["time_s"] >= t_start_ov - 0.1) & (cal_df["time_s"] <= t_end_ov + 0.1)
    cal_pool = cal_df[in_overlap_c & (cal_df["status"] == "ACCEPTED")].copy()

    matched_pairs: List[Dict[str, Any]] = []
    matched_s_idx_set = set()
    matched_c_idx_set = set()

    if not sarc_pool.empty and not cal_pool.empty:
        s_pool_indices = sarc_pool.index.to_numpy()
        c_pool_indices = cal_pool.index.to_numpy()

        s_times = sarc_pool["time_s"].to_numpy(dtype=float)
        c_times = cal_pool["time_s"].to_numpy(dtype=float)

        raw_matches = monotonic_bipartite_match(
            s_times, c_times, eff_tau_emd, max_dev=params.canonical_delay_window_s
        )

        for s_i, c_j, delay_s in raw_matches:
            s_row_idx = s_pool_indices[s_i]
            c_row_idx = c_pool_indices[c_j]

            s_ev = sarc_df.loc[s_row_idx]
            c_ev = cal_df.loc[c_row_idx]

            ts = float(s_ev["time_s"])
            tc = float(c_ev["time_s"])
            c_amp = float(c_ev["amplitude"])
            residual = abs(delay_s - eff_tau_emd)

            # Compute onsets if raw traces provided
            onset_delay_s = np.nan
            if sig_sarc_raw is not None and sig_cal_raw is not None:
                t_s_on = estimate_transient_onset(
                    sig_sarc_raw, fs_sarc, t0_sarc, int(s_ev["peak_index"]),
                    float(s_ev["l0_um"]) if pd.notna(s_ev["l0_um"]) else float(sig_sarc_raw[int(s_ev["peak_index"])]),
                    polarity="negative", frac=params.onset_threshold_frac,
                )
                t_c_on = estimate_transient_onset(
                    sig_cal_raw, fs_cal, t0_cal, int(c_ev["peak_index"]),
                    float(c_ev["local_baseline_value"]) if pd.notna(c_ev["local_baseline_value"]) else float(sig_cal_raw[int(c_ev["peak_index"])]),
                    polarity="positive", frac=params.onset_threshold_frac,
                )
                if np.isfinite(t_s_on) and np.isfinite(t_c_on):
                    onset_delay_s = float(t_s_on - t_c_on)

            # Determine confidence
            is_corrupted = bool(s_ev["peak_corrupted"])
            if is_corrupted:
                conf = "MEDIUM" if residual <= params.high_conf_delay_tol_s else "LOW"
            elif residual <= params.high_conf_delay_tol_s and not bool(s_ev["artifact_nearby"]):
                conf = "HIGH"
            elif residual <= params.medium_conf_delay_tol_s:
                conf = "MEDIUM"
            else:
                conf = "LOW"

            # Update sarcomere pairing fields ONLY - NEVER alter mechanical status, class, or amplitude
            sarc_df.loc[s_row_idx, "calcium_match_status"] = "MATCHED"
            sarc_df.loc[s_row_idx, "matched_calcium_time_s"] = tc
            sarc_df.loc[s_row_idx, "calcium_amplitude"] = c_amp
            sarc_df.loc[s_row_idx, "calcium_mechanical_delay_s"] = delay_s
            sarc_df.loc[s_row_idx, "calcium_onset_delay_s"] = onset_delay_s
            sarc_df.loc[s_row_idx, "calcium_matching_confidence"] = conf
            sarc_df.loc[s_row_idx, "assistance_reason"] = "independent_paired"

            # Update calcium pairing fields
            cal_df.loc[c_row_idx, "calcium_match_status"] = "MATCHED"
            cal_df.loc[c_row_idx, "matched_sarcomere_time_s"] = ts
            cal_df.loc[c_row_idx, "matched_sarcomere_index"] = int(s_ev["peak_index"])
            cal_df.loc[c_row_idx, "sarcomere_amplitude_um"] = float(s_ev["abs_shortening_um"]) if pd.notna(s_ev["abs_shortening_um"]) else np.nan

            matched_s_idx_set.add(s_row_idx)
            matched_c_idx_set.add(c_row_idx)

            matched_pairs.append({
                "sarcomere_index": int(s_ev["peak_index"]),
                "sarcomere_time_s": ts,
                "calcium_index": int(c_ev["peak_index"]),
                "calcium_time_s": tc,
                "peak_to_peak_delay_s": delay_s,
                "onset_to_onset_delay_s": onset_delay_s,
                "delay_residual_s": residual,
                "matching_confidence": conf,
                "sarcomere_amplitude_um": float(s_ev["abs_shortening_um"]) if pd.notna(s_ev["abs_shortening_um"]) else np.nan,
                "calcium_amplitude": c_amp,
                "sarcomere_only_class": str(s_ev["event_class"]),
                "assisted_class": str(s_ev["event_class"]),
                "assistance_reason": "independent_paired",
                "event_nature": str(s_ev.get("event_nature", "MEASURED")),
                "recovery_status": str(s_ev.get("recovery_status", "MEASURED")),
            })

    # Flag unmatched sarcomere events
    for idx_s in sarc_df.index:
        if idx_s not in matched_s_idx_set:
            if sarc_df.loc[idx_s, "status"] == "ACCEPTED":
                sarc_df.loc[idx_s, "calcium_match_status"] = "UNMATCHED_MECHANICAL"
                sarc_df.loc[idx_s, "assistance_reason"] = "unmatched_mechanical_contraction"
            else:
                sarc_df.loc[idx_s, "calcium_match_status"] = "UNMATCHED_REJECTED"
                sarc_df.loc[idx_s, "assistance_reason"] = f"unmatched_sarcomere_{sarc_df.loc[idx_s, 'reason']}"

    # Flag unmatched calcium events
    for idx_c in cal_df.index:
        if idx_c not in matched_c_idx_set:
            if cal_df.loc[idx_c, "status"] == "ACCEPTED":
                cal_df.loc[idx_c, "calcium_match_status"] = "UNMATCHED_CALCIUM"
            else:
                cal_df.loc[idx_c, "calcium_match_status"] = "UNMATCHED_REJECTED"

    unmatched_s_df = sarc_df[(sarc_df["status"] == "ACCEPTED") & (sarc_df["calcium_match_status"] == "UNMATCHED_MECHANICAL")].copy()
    unmatched_c_df = cal_df[(cal_df["status"] == "ACCEPTED") & (cal_df["calcium_match_status"] == "UNMATCHED_CALCIUM")].copy()

    n_acc_sarc = int(np.sum(sarc_df["status"] == "ACCEPTED")) if not sarc_df.empty else 0
    n_meas_sarc = int(np.sum((sarc_df["status"] == "ACCEPTED") & sarc_df["has_valid_amplitude"])) if not sarc_df.empty else 0

    return {
        "sarc_events": sarc_df,
        "sarc_only_events": sarc_only_copy,
        "calcium_events": cal_df,
        "matched_pairs": matched_pairs,
        "unmatched_sarcomere_events": unmatched_s_df,
        "unmatched_calcium_events": unmatched_c_df,
        "estimated_emd_delay_s": tau_emd,
        "emd_delay_status": emd_status,
        "n_matched_pairs": len(matched_pairs),
        "n_unmatched_sarc": len(unmatched_s_df),
        "n_unmatched_cal": len(unmatched_c_df),
        "n_directly_measured": n_meas_sarc,
        "n_calcium_assisted": 0,
        "n_valid_amplitude": n_meas_sarc,
        "n_lacking_amplitude": n_acc_sarc - n_meas_sarc,
        "n_unresolved": len(unmatched_c_df),
        "n_classifications_changed": 0,
        "n_borderline_recovered": 0,
        "n_artifact_corrupted": int(np.sum(sarc_df["peak_corrupted"].astype(bool))) if not sarc_df.empty else 0,
        "enable_assistance": False,
        "temporal_overlap": {
            "t_start_s": t_start_ov,
            "t_end_s": t_end_ov,
            "overlap_duration_s": overlap_dur,
        },
        "combined_qc": combined_qc,
    }
