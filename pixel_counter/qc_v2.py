from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import numpy as np
from .config import BeatCounterConfig
from .preprocessing import compute_spike_fraction

def _compute_descriptive_metrics(candidates: Dict[str, Any]) -> Dict[str, float]:
    """Compute IBI CV, amplitude variability, pauses and periodicity."""
    metrics = {
        "ibi_cv": np.nan,
        "amp_cv": np.nan,
        "periodicity": np.nan,
        "max_pause_s": np.nan
    }
    status = candidates.get("status", np.array([], dtype=object))
    valid_cands = (status == "PASS") if len(status) > 0 else np.ones(len(candidates.get("time_s", [])), dtype=bool)

    times = candidates.get("time_s", np.array([]))[valid_cands]
    amps = candidates.get("amplitude", np.array([]))[valid_cands]

    if len(times) >= 3:
        ibis = np.diff(times)
        metrics["max_pause_s"] = float(np.max(ibis))
        if np.mean(ibis) > 0:
            metrics["ibi_cv"] = float(np.std(ibis) / np.mean(ibis))

    if len(amps) >= 2 and np.mean(amps) > 0:
        metrics["amp_cv"] = float(np.std(amps) / np.mean(amps))

    return metrics

def evaluate_qc_v2(
    sig: np.ndarray,
    fs: float,
    candidates: Dict[str, Any],
    meta: Dict[str, Any],
    config: BeatCounterConfig
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Step 18: Unified Quality Control for the experimental v2 pipeline.
    Evaluates technical signal-quality and detector-related uncertainty.
    Distinguishes results into ERROR, REJECT, REVIEW, and PASS.
    Returns:
        status (str): "ERROR", "REJECT", "REVIEW", or "PASS".
        reason (str): Short description of the primary reason.
        qc_meta (Dict): Dictionary containing detailed descriptive metrics and diagnostic information.
    """
    qc_meta: Dict[str, Any] = {}

    # 1. Technical indicators: Sampling frequency and time-axis validity
    if not np.isfinite(fs) or fs <= 0.0:
        qc_meta["fs_valid"] = False
        return "ERROR", "invalid_sampling_frequency", qc_meta
    qc_meta["fs_valid"] = True

    if sig.size == 0:
        qc_meta["empty_signal"] = True
        return "ERROR", "empty_signal", qc_meta

    # 2. Coverage and genuine missing intervals
    valid_mask = np.isfinite(sig)
    valid_count = np.sum(valid_mask)
    coverage_fraction = valid_count / sig.size if sig.size > 0 else 0.0
    qc_meta["coverage_fraction"] = float(coverage_fraction)

    if valid_count == 0:
        return "REJECT", "no_valid_data", qc_meta

    valid_duration = valid_count / fs
    qc_meta["valid_duration_s"] = float(valid_duration)
    if valid_duration < 1.0:
        return "REJECT", "insufficient_valid_data", qc_meta

    # 3. Overlap conflicts
    if meta.get("stitch_overlap_conflicts", 0) > 0:
        qc_meta["overlap_conflicts"] = meta["stitch_overlap_conflicts"]
        return "REJECT", "overlap_conflict", qc_meta


    # 4. Clipping
    # (Requires development-data validation to set a definitive REJECT threshold)
    valid_sig = sig[valid_mask]
    if valid_sig.size > 0:
        max_val = np.nanmax(valid_sig)
        max_count = np.sum(valid_sig == max_val)
        if max_count > 0.05 * valid_sig.size and max_val > 0.1:
             qc_meta["clipping_detected"] = True
             return "REVIEW", "signal_clipping", qc_meta

    # 5. Local noise and temporal resolution
    # (Requires development-data validation to set definitive REJECT thresholds)
    spike_frac = compute_spike_fraction(sig)
    qc_meta["spike_fraction"] = float(spike_frac)

    if spike_frac > 0.85:
        return "REVIEW", "excessive_local_noise", qc_meta

    if fs < 10.0:
         return "REVIEW", "insufficient_temporal_resolution", qc_meta

    # 6. Detector-related uncertainty

    # Both polarities failed
    if meta.get("candidate_selection_reason") == "both_failed":
        return "REVIEW", "both_polarities_failed", qc_meta

    # Ambiguous polarity
    if meta.get("polarity_ambiguity") is True:
        return "REVIEW", "ambiguous_polarity", qc_meta

    # Mixed morphology / unstable candidates
    is_seed = candidates.get("is_seed", np.array([], dtype=bool))
    if len(is_seed) > 0 and np.sum(is_seed) < 2 and len(is_seed) >= 3:
        # Many candidates but very few seeds
        return "REVIEW", "insufficient_reliable_seeds", qc_meta

    # Unresolved neighboring pairs
    status = candidates.get("status", np.array([], dtype=object))
    if np.any(status == "REVIEW"):
        return "REVIEW", "unresolved_neighboring_peaks", qc_meta

    # 7. Sensitivity stability diagnostic
    from .analysis import detect_events_v2

    polarity = meta.get("best_polarity", 1)

    # +15% threshold
    config_high = BeatCounterConfig(**{**config.__dict__, "prom0": config.prom0 * 1.15})
    res_high = detect_events_v2(sig, fs, config_high, polarity=polarity)

    # -15% threshold
    config_low = BeatCounterConfig(**{**config.__dict__, "prom0": config.prom0 * 0.85})
    res_low = detect_events_v2(sig, fs, config_low, polarity=polarity)

    # Compare against accepted events only
    base_mask = (status == "PASS") if len(status) > 0 else np.ones(len(candidates.get("time_s", [])), dtype=bool)
    n_base = int(np.sum(base_mask))

    high_status = res_high.get("status", np.array([], dtype=object))
    high_mask = (high_status == "PASS") if len(high_status) > 0 else np.ones(len(res_high.get("time_s", [])), dtype=bool)
    n_high = int(np.sum(high_mask))

    low_status = res_low.get("status", np.array([], dtype=object))
    low_mask = (low_status == "PASS") if len(low_status) > 0 else np.ones(len(res_low.get("time_s", [])), dtype=bool)
    n_low = int(np.sum(low_mask))

    qc_meta["stability_count_base"] = n_base
    qc_meta["stability_count_high"] = n_high
    qc_meta["stability_count_low"] = n_low

    def match_timestamps(t1, t2, tol=0.01):
        if len(t1) == 0 or len(t2) == 0:
            return 0
        matches = 0
        used_t2 = np.zeros(len(t2), dtype=bool)
        for t in t1:
            diffs = np.abs(t2 - t)
            valid_idx = np.where((diffs <= tol) & (~used_t2))[0]
            if len(valid_idx) > 0:
                best_match = valid_idx[np.argmin(diffs[valid_idx])]
                used_t2[best_match] = True
                matches += 1
        return matches

    base_times = candidates.get("time_s", np.array([]))[base_mask]
    high_times = res_high.get("time_s", np.array([]))[high_mask]
    low_times = res_low.get("time_s", np.array([]))[low_mask]

    high_matches = match_timestamps(base_times, high_times)
    low_matches = match_timestamps(base_times, low_times)

    qc_meta["stability_match_high"] = high_matches
    qc_meta["stability_match_low"] = low_matches

    if n_base >= 3:
        # Both event-count changes and timestamp instability must influence the REVIEW decision
        count_unstable = (n_high < n_base * 0.8) or (n_low > n_base * 1.2)
        match_unstable = (high_matches < n_base * 0.8) or (low_matches < n_base * 0.8)

        if count_unstable or match_unstable:
            qc_meta["unstable_detection"] = True
            return "REVIEW", "sensitivity_instability", qc_meta


    # 8. Descriptive metrics
    descriptive = _compute_descriptive_metrics(candidates)
    qc_meta.update(descriptive)

    return "PASS", "clean_signal", qc_meta
