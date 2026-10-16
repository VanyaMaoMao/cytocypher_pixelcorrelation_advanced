from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .config import BeatCounterConfig
from .preprocessing import compute_spike_fraction
from .v2_signal import valid_blocks


def compute_descriptive_metrics(times: np.ndarray, amplitudes: np.ndarray) -> Dict[str, float]:
    """Descriptive statistics of the ACCEPTED contractions (never used to accept/reject)."""
    metrics = {"ibi_cv": np.nan, "amp_cv": np.nan, "max_pause_s": np.nan}
    times = np.asarray(times, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if times.size >= 3:
        ibis = np.diff(times)
        metrics["max_pause_s"] = float(np.max(ibis))
        if np.mean(ibis) > 0:
            metrics["ibi_cv"] = float(np.std(ibis) / np.mean(ibis))
    if amplitudes.size >= 2 and np.mean(amplitudes) > 0:
        metrics["amp_cv"] = float(np.std(amplitudes) / np.mean(amplitudes))
    return metrics


def evaluate_qc_v2(
    sig: np.ndarray,
    fs: float,
    meta: Dict[str, Any],
    config: BeatCounterConfig,
) -> Tuple[str, str, Dict[str, Any]]:
    """Technical input-validity QC for the experimental v2 pipeline.

    This stage only judges whether the INPUT can be analysed at all; it says nothing about
    individual contractions (those are ACCEPTED/REJECTED by the detector) and has no
    "uncertain" outcome.

    Returns
    -------
    status : "ERROR"  the input is invalid (bad sampling frequency, empty signal)
             "REJECT" the recording is fundamentally unusable (no/too little valid data,
                      conflicting overlapping samples, corrupted spike noise, sub-10 Hz sampling)
             "PASS"   the recording can be analysed
    reason : short machine-readable reason
    qc_meta: technical diagnostics, including non-fatal ``qc_flags`` (e.g. clipping, NaN gaps)
    """
    qc_meta: Dict[str, Any] = {}
    flags: List[str] = []
    qc_meta["qc_flags"] = flags

    # 1. Sampling frequency / time axis
    if not np.isfinite(fs) or fs <= 0.0:
        qc_meta["fs_valid"] = False
        return "ERROR", "invalid_sampling_frequency", qc_meta
    qc_meta["fs_valid"] = True

    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        qc_meta["empty_signal"] = True
        return "ERROR", "empty_signal", qc_meta

    t0 = meta.get("time_origin_s", None)
    if t0 is not None and not np.isfinite(t0):
        return "ERROR", "invalid_time_origin", qc_meta

    # 2. Coverage and genuine missing intervals
    valid_mask = np.isfinite(sig)
    valid_count = int(np.sum(valid_mask))
    qc_meta["coverage_fraction"] = float(valid_count / sig.size)
    if valid_count == 0:
        return "REJECT", "no_valid_data", qc_meta
    qc_meta["valid_duration_s"] = float(valid_count / fs)
    if valid_count / fs < 1.0:
        return "REJECT", "insufficient_valid_data", qc_meta
    n_gaps = len(valid_blocks(sig)) - 1
    qc_meta["nan_gap_count"] = int(max(0, n_gaps))
    if n_gaps > 0:
        flags.append("nan_gaps")

    # 3. Conflicting overlapping samples from the stitching stage
    if meta.get("stitch_overlap_conflicts", 0) > 0:
        qc_meta["overlap_conflicts"] = meta["stitch_overlap_conflicts"]
        return "REJECT", "overlap_conflict", qc_meta

    # 4. Temporal resolution
    if fs < 10.0:
        return "REJECT", "insufficient_temporal_resolution", qc_meta

    # 5. Corrupted local noise
    spike_frac = compute_spike_fraction(sig[valid_mask])
    qc_meta["spike_fraction"] = float(spike_frac)
    if spike_frac > 0.85:
        return "REJECT", "excessive_local_noise", qc_meta

    # 6. Clipping (non-fatal: amplitudes may be compressed but the waveform is analysable)
    valid_sig = sig[valid_mask]
    max_val = np.max(valid_sig)
    if np.sum(valid_sig == max_val) > 0.05 * valid_sig.size and max_val > 0.1:
        qc_meta["clipping_detected"] = True
        flags.append("signal_clipping")

    return "PASS", "clean_signal", qc_meta
