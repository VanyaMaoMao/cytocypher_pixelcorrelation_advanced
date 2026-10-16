"""Experimental v2 pipeline: raw waveform -> polarity -> baseline-relative amplitude -> detection.

One authoritative polarity decision per segment is taken from the ORIGINAL raw waveform
(:func:`pixel_counter.v2_signal.determine_raw_polarity`).  All later stages consume the same
decision and the same baseline-relative amplitude waveform A(t) = polarity * (S_raw - B_raw).

Every coordinate reported (``events``, ``candidates``) is derived from the same arrays that the
review plot draws: ``time_abs_s = t0 + index / fs`` and ``amplitude = A[index]``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import BeatCounterConfig
from .qc_v2 import compute_descriptive_metrics, evaluate_qc_v2
from .v2_detector import ACCEPTED, DetectorParams, detect_contractions, params_from_config
from .v2_signal import compute_amplitude_waveform, determine_raw_polarity


def resolve_polarity(
    sig_raw: np.ndarray,
    fs: float,
    config: BeatCounterConfig,
    polarity_override: Optional[int] = None,
    meta: Optional[Dict] = None,
) -> Dict:
    """Single polarity decision. Supports manual overrides and auditing against automatic prediction.

    Manual override sources:
      1. Explicit argument ``polarity_override`` (+1 or -1).
      2. ``meta['polarity_override']`` or ``meta['force_polarity']``.
      3. ``config.v2_polarity_mode`` ('positive' | 'negative').
    """
    sig_cal = None
    fs_cal = fs
    t0_raw = 0.0
    t0_cal = 0.0
    use_cal = bool(getattr(config, "v2_use_calcium_polarity", False))
    if meta and use_cal:
        sig_cal = meta.get("sig_cal", meta.get("calcium_signal"))
        fs_cal = meta.get("fs_cal", meta.get("calcium_fs", fs))
        t0_raw = float(meta.get("time_origin_s", 0.0))
        t0_cal = float(meta.get("t0_cal", meta.get("calcium_time_origin_s", t0_raw)))

    auto = determine_raw_polarity(
        sig_raw, fs, sig_cal=sig_cal, fs_cal=fs_cal, t0_raw=t0_raw, t0_cal=t0_cal
    )

    ov = polarity_override
    if ov is None and meta:
        ov = meta.get("polarity_override", meta.get("force_polarity"))
    if ov is None:
        mode = str(getattr(config, "v2_polarity_mode", "auto")).lower()
        if mode == "positive":
            ov = 1
        elif mode == "negative":
            ov = -1

    if ov in (-1, 1):
        res = dict(auto)
        res.update(
            best_polarity=int(ov),
            polarity=int(ov),
            polarity_source="manual_override",
            status="OK",
            method="manual_override",
            reason=f"manual_override_{ov:+d}",
            auto_polarity=int(auto.get("polarity", 0)),
            auto_status=str(auto.get("status", "")),
            auto_method=str(auto.get("method", "")),
            auto_reason=str(auto.get("reason", "")),
        )
        return res

    res = dict(auto)
    res.update(
        best_polarity=int(auto.get("polarity", 0)),
        polarity_source="automatic",
        auto_polarity=int(auto.get("polarity", 0)),
        auto_status=str(auto.get("status", "")),
        auto_method=str(auto.get("method", "")),
        auto_reason=str(auto.get("reason", "")),
    )
    return res


def analyze_v2_pipeline(
    sig: np.ndarray,
    fs: float,
    config: BeatCounterConfig,
    t0: float = 0.0,
    meta: Optional[Dict] = None,
    params: Optional[DetectorParams] = None,
    polarity_override: Optional[int] = None,
) -> Dict:
    """Run the complete v2 analysis on an UNINVERTED raw PixelCorrelation waveform.

    Parameters
    ----------
    sig : raw stitched PixelCorrelation (never pre-inverted).
    t0 : absolute time of sample 0 (``meta['time_origin_s']`` of the stitched signal).
    meta : stitching metadata (used by the technical QC).
    params : optional explicit detector strictness parameters (default: from ``config``).
    polarity_override : optional explicit polarity (+1 or -1) overriding automatic prediction.

    Returns dict with keys
      ``metadata``       qc status/reason, polarity, active parameter values, summary numbers
      ``polarity_result``the polarity decision with its evidence
      ``waveform``       raw, baseline_raw, amplitude (all aligned) and absolute time axis
      ``events``         DataFrame with every listed candidate (ACCEPTED / REJECTED + reason)
      ``candidates``     ACCEPTED events as arrays (``time_s`` absolute, ``amplitude``)
    """
    meta = dict(meta or {})
    params = (params or params_from_config(config)).validate()
    sig = np.asarray(sig, dtype=float)
    metadata: Dict = {"time_origin_s": float(t0), "fs": float(fs) if np.isfinite(fs) else fs,
                      "detector_params": {"min_rel_amplitude": params.min_rel_amplitude,
                                          "min_rel_recovery": params.min_rel_recovery,
                                          "min_snr": params.min_snr}}

    status, reason, qc_meta = evaluate_qc_v2(sig, fs, meta, config)
    metadata.update(qc_meta)

    empty_events = detect_contractions(np.array([]), 1.0, params)["events"]
    result = {
        "metadata": metadata,
        "polarity_result": {"best_polarity": 0, "polarity": 0, "status": "NOT_RUN", "method": "none", "reason": ""},
        "waveform": {"raw": sig, "time_s": t0 + np.arange(sig.size) / fs if (np.isfinite(fs) and fs > 0) else np.array([]),
                     "amplitude": np.full(sig.size, np.nan), "baseline_raw": np.full(sig.size, np.nan)},
        "events": empty_events.assign(time_abs_s=[]),
        "candidates": {"candidate_index": np.array([], dtype=int), "time_s": np.array([], dtype=float),
                        "amplitude": np.array([], dtype=float), "status": np.array([], dtype=object)},
    }

    def _finish(st, rs):
        metadata["qc_status"], metadata["qc_reason"] = st, rs
        metadata["n_accepted"] = int(len(result["candidates"]["time_s"]))
        return result

    if status != "PASS":
        return _finish(status, reason)

    pol = resolve_polarity(sig, fs, config, polarity_override=polarity_override, meta=meta)
    result["polarity_result"] = pol
    metadata.update({
        "raw_polarity": int(pol["best_polarity"]),
        "polarity_source": pol.get("polarity_source", "automatic"),
        "polarity_method": pol["method"],
        "polarity_reason": pol["reason"],
        "auto_polarity": pol.get("auto_polarity", pol["best_polarity"]),
        "auto_method": pol.get("auto_method", pol["method"]),
        "auto_reason": pol.get("auto_reason", pol["reason"]),
    })
    if pol["best_polarity"] not in (-1, 1):
        metadata["polarity_status"] = "POLARITY_INPUT_REQUIRED"
        return _finish("REJECT", "polarity_undetermined")

    wf = compute_amplitude_waveform(sig, fs, int(pol["best_polarity"]))
    result["waveform"].update({"amplitude": wf["amplitude"], "baseline_raw": wf["baseline_raw"]})

    det = detect_contractions(wf["amplitude"], fs, params)
    ev = det["events"].copy()
    ev["time_abs_s"] = t0 + ev["time_rel_s"]
    ev["raw_value"] = sig[ev["index"].to_numpy(dtype=int)] if len(ev) else np.array([])
    result["events"] = ev
    acc = ev[ev["status"] == ACCEPTED]
    result["candidates"] = {
        "candidate_index": acc["index"].to_numpy(dtype=int),
        "time_s": acc["time_abs_s"].to_numpy(dtype=float),
        "amplitude": acc["amplitude"].to_numpy(dtype=float),
        "prominence": acc["prominence"].to_numpy(dtype=float),
        "width_s": acc["fwhm_s"].to_numpy(dtype=float),
        "status": np.array([ACCEPTED] * len(acc), dtype=object),
    }
    metadata.update({
        "reference_amplitude": det["reference_amplitude"],
        "amplitude_threshold": det["amplitude_threshold"],
        "noise_sigma": det["noise_sigma"],
        "n_rejected": int((ev["status"] != ACCEPTED).sum()),
    })
    metadata.update(compute_descriptive_metrics(acc["time_abs_s"].to_numpy(), acc["amplitude"].to_numpy()))
    return _finish("PASS", "clean_signal" if len(acc) else "no_accepted_contractions")
