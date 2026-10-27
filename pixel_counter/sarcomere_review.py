"""
pixel_counter.sarcomere_review
==============================

Interactive scientific review utility and HTML generator for cardiomyocyte
Sarcomere Shortening analysis (both SARCOMERE ONLY and SARCOMERE + CALCIUM modes).

Scientific & Architectural Principles:
1. Downward Contraction Polarity:
   Sarcomere length decrease corresponds to contraction. True contraction nadirs
   are local minima of SarcomereLength L(t). Raw micrometer values are never inverted.
2. Standard Multi-Panel Presentation:
   - SARCOMERE ONLY (2 panels):
     * Panel 1: Original untouched SarcomereLength L(t) (um) directly from Excel,
       purely informational, no baseline line, no markers, ignore click-to-add.
     * Panel 2: Normalized shortening % (100 * [L0 - L] / L0) with resting baseline (0)
       and detection threshold. Fully interactive with class shapes (circle MAIN,
       diamond SECONDARY, triangle AMBIGUOUS).
   - SARCOMERE + CALCIUM (3 panels):
     * Panel 1: Original untouched SarcomereLength L(t) (um), informational only.
     * Panel 2: Normalized shortening %, independently interactive.
     * Panel 3: Normalized RatioMetricCalcium (Delta R/R0) transient waveform with
       resting baseline (0) and Stage 1 amplitude threshold, independently interactive.
3. Separation of Concepts:
   - Independent calcium transient analysis vs. calcium assistance for sarcomere recovery.
   - Prominent toggle: [Calcium Assistance: ON | OFF] in header allows instantaneous
     switching between pre-assistance and post-assistance results without reloading.
4. Artifact-Aware Calcium Recovery:
   - When assistance is ON, recovers true contractions obscured by vertical tracking artifacts.
   - Corrupted troughs retain NaN shortening values (never fabricated!).
   - In UI, corrupted trough events are displayed with clear N/A shortening indicators.
5. Interactive Review & Manual Correction:
   - Toggling: GREEN -> RED CROSS -> GREEN; NOTHING -> BLUE -> NOTHING.
   - Snapping: clicks on shortening curve snap to true local minimum of SarcomereLength L(t).
   - Independent calcium editing: clicks on calcium curve snap to local maximum of Delta R/R0.
   - Per-event reclassification (MAIN / SECONDARY / AMBIGUOUS) and per-event strictness.
   - Whole-segment manual exclusion with reason.
   - Save annotations to CSV and Export/Import overrides JSON.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import BeatCounterConfig
from .io_utils import extract_sample_id_from_segment_sheet, load_cytocypher_excel
from .polarity_overrides import compute_file_sha256
from .preprocessing import build_concatenated_signal
from .sarcomere_detector import (
    SARCOMERE_STRICTNESS_PRESETS,
    SarcomereDetectorParams,
)
from .sarcomere_pipeline import (
    analyze_sarcomere_segment_from_excel,
    discover_sarcomere_workbook_channels,
)


def b64_encode_array(arr: Optional[np.ndarray]) -> Optional[str]:
    """Encode numpy array to base64 float32 for compact embedding in HTML."""
    if arr is None or len(arr) == 0:
        return None
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("utf-8")


def _num(x: Any) -> Optional[float]:
    """Convert value to JSON-safe float (NaN/inf -> None)."""
    try:
        val = float(x)
        return val if np.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def _sarcomere_run_payload(sarc_run_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured, JSON-safe run dictionary for sarcomere preset."""
    ev = sarc_run_dict.get("events")
    p: Optional[SarcomereDetectorParams] = sarc_run_dict.get("params")
    qc_meta = sarc_run_dict.get("qc_meta", {})
    det_metrics = sarc_run_dict.get("detector_metrics", {})
    n_ev = len(ev) if ev is not None and not ev.empty else 0

    return {
        "params": {
            "min_abs_shortening_um": p.min_abs_shortening_um if p else 0.0065,
            "min_pct_shortening": p.min_pct_shortening if p else 0.35,
            "min_prominence_sigma": p.min_prominence_sigma if p else 4.0,
            "min_prominence_um": p.min_prominence_um if p else 0.0055,
            "min_rel_amplitude": p.min_rel_amplitude if p else 0.18,
            "min_separation_s": p.min_separation_s if p else 0.11,
        },
        "qc_status": str(sarc_run_dict.get("qc_status", "UNKNOWN")),
        "qc_reason": str(sarc_run_dict.get("qc_reason", "")),
        "analyzable_duration_s": _num(qc_meta.get("analyzable_duration_s")),
        "frequency_hz": _num(sarc_run_dict.get("frequency_hz")),
        "frequency_bpm": _num(sarc_run_dict.get("frequency_bpm")),
        "main_frequency_hz": _num(sarc_run_dict.get("main_frequency_hz")),
        "main_frequency_bpm": _num(sarc_run_dict.get("main_frequency_bpm")),
        "secondary_frequency_hz": _num(sarc_run_dict.get("secondary_frequency_hz")),
        "secondary_frequency_epm": _num(sarc_run_dict.get("secondary_frequency_epm")),
        "noise_sigma": _num(sarc_run_dict.get("noise_sigma", sarc_run_dict.get("noise_sigma_um", qc_meta.get("sigma_hf_um")))),
        "noise_sigma_um": _num(sarc_run_dict.get("noise_sigma", sarc_run_dict.get("noise_sigma_um", qc_meta.get("sigma_hf_um")))),
        "noise_sigma_pct": _num(sarc_run_dict.get("noise_sigma_pct", det_metrics.get("noise_sigma_pct"))),
        "amplitude_threshold": _num(sarc_run_dict.get("threshold_main_pct", sarc_run_dict.get("amplitude_threshold_pct", sarc_run_dict.get("amplitude_threshold")))),
        "amplitude_threshold_pct": _num(sarc_run_dict.get("threshold_main_pct", sarc_run_dict.get("amplitude_threshold_pct", sarc_run_dict.get("amplitude_threshold")))),
        "threshold_main_pct": _num(sarc_run_dict.get("threshold_main_pct", det_metrics.get("threshold_main_pct"))),
        "threshold_sce_pct": _num(sarc_run_dict.get("threshold_sce_pct", det_metrics.get("threshold_sce_pct"))),
        "reference_amplitude_pct": _num(sarc_run_dict.get("reference_amplitude_pct", det_metrics.get("reference_amplitude_pct"))),
        "threshold_ordering_status": str(sarc_run_dict.get("threshold_ordering_status", det_metrics.get("threshold_ordering_status", "distinct"))),
        "threshold_main_components": sarc_run_dict.get("threshold_main_components", det_metrics.get("threshold_main_components", {})),
        "threshold_sce_components": sarc_run_dict.get("threshold_sce_components", det_metrics.get("threshold_sce_components", {})),
        "strictness_preset": str(sarc_run_dict.get("strictness_preset", p.preset_name if p else "default")),
        "baseline_estimation_method": str(det_metrics.get("baseline_estimation_method", "two_pass_diastolic_percentile")),
        "reference_amplitude": _num(sarc_run_dict.get("reference_amplitude_pct", sarc_run_dict.get("mean_pct_shortening"))),
        "mean_l0_um": _num(sarc_run_dict.get("mean_l0_um")),
        "mean_l_min_um": _num(sarc_run_dict.get("mean_l_min_um")),
        "mean_abs_shortening_um": _num(sarc_run_dict.get("mean_abs_shortening_um")),
        "mean_pct_shortening": _num(sarc_run_dict.get("mean_pct_shortening")),
        "n_accepted": int(sarc_run_dict.get("n_accepted", 0)),
        "n_main": int(sarc_run_dict.get("n_main", 0)),
        "n_secondary": int(sarc_run_dict.get("n_secondary", 0)),
        "n_ambiguous": int(sarc_run_dict.get("n_ambiguous", 0)),
        "n_directly_measured": int(sarc_run_dict.get("n_directly_measured", sarc_run_dict.get("n_accepted", 0))),
        "n_calcium_assisted": int(sarc_run_dict.get("n_calcium_assisted", 0)),
        "n_valid_amplitude": int(sarc_run_dict.get("n_valid_amplitude", sarc_run_dict.get("n_accepted", 0))),
        "n_lacking_amplitude": int(sarc_run_dict.get("n_lacking_amplitude", 0)),
        "n_unresolved": int(sarc_run_dict.get("n_unresolved", 0)),
        "events": {
            "index": [int(i) for i in ev["peak_index"]] if n_ev else [],
            "time_s": [float(t) for t in ev["time_s"]] if n_ev else [],
            "amplitude": [_num(v) for v in ev["pct_shortening"]] if n_ev else [],
            "raw_value": [_num(v) for v in ev["l_min_um"]] if n_ev else [],
            "l0_um": [_num(v) for v in ev["l0_um"]] if n_ev else [],
            "l_min_um": [_num(v) for v in ev["l_min_um"]] if n_ev else [],
            "abs_shortening_um": [_num(v) for v in ev["abs_shortening_um"]] if n_ev else [],
            "pct_shortening": [_num(v) for v in ev["pct_shortening"]] if n_ev else [],
            "display_amplitude": [_num(v) for v in (ev["display_amplitude"] if "display_amplitude" in ev.columns else ev["pct_shortening"])] if n_ev else [],
            "takeoff_l0_um": [_num(v) for v in ev["takeoff_l0_um"]] if n_ev else [],
            "local_abs_shortening_um": [_num(v) for v in ev["local_abs_shortening_um"]] if n_ev else [],
            "local_pct_shortening": [_num(v) for v in ev["local_pct_shortening"]] if n_ev else [],
            "event_class": [str(c) for c in ev["event_class"]] if n_ev else [],
            "classification_confidence": [str(c) for c in ev["classification_confidence"]] if n_ev else [],
            "classification_reason": [str(c) for c in ev["classification_reason"]] if n_ev else [],
            "status": [str(s) for s in ev["status"]] if n_ev else [],
            "reason": [str(s) for s in ev["reason"]] if n_ev else [],
            "diagnostic_status": [str(s) for s in (ev["preliminary_status"] if "preliminary_status" in ev.columns else ev["status"])] if n_ev else [],
            "diagnostic_reason": [str(s) for s in (ev["preliminary_reason"] if "preliminary_reason" in ev.columns else ev["reason"])] if n_ev else [],
            "origin": [str(o) for o in ev["origin"]] if n_ev else [],
            "peak_corrupted": [bool(b) for b in ev["peak_corrupted"]] if n_ev else [],
            "artifact_nearby": [bool(b) for b in ev["artifact_nearby"]] if n_ev else [],
            "event_nature": [str(x) for x in ev["event_nature"]] if (n_ev and "event_nature" in ev.columns) else ["MEASURED"] * n_ev,
            "recovery_status": [str(x) for x in ev["recovery_status"]] if (n_ev and "recovery_status" in ev.columns) else ["DIRECT_MEASUREMENT"] * n_ev,
            "has_valid_amplitude": [bool(x) for x in ev["has_valid_amplitude"]] if (n_ev and "has_valid_amplitude" in ev.columns) else [True] * n_ev,
            "calcium_match_status": [str(m) for m in ev["calcium_match_status"]] if (n_ev and "calcium_match_status" in ev.columns) else [],
            "matched_calcium_time_s": [_num(t) for t in ev["matched_calcium_time_s"]] if (n_ev and "matched_calcium_time_s" in ev.columns) else [],
            "calcium_amplitude": [_num(a) for a in ev["calcium_amplitude"]] if (n_ev and "calcium_amplitude" in ev.columns) else [],
            "calcium_mechanical_delay_s": [_num(d) for d in ev["calcium_mechanical_delay_s"]] if (n_ev and "calcium_mechanical_delay_s" in ev.columns) else [],
            "calcium_onset_delay_s": [_num(d) for d in ev["calcium_onset_delay_s"]] if (n_ev and "calcium_onset_delay_s" in ev.columns) else [],
            "calcium_matching_confidence": [str(c) for c in ev["calcium_matching_confidence"]] if (n_ev and "calcium_matching_confidence" in ev.columns) else [],
            "sarcomere_only_classification": [str(c) for c in ev["sarcomere_only_classification"]] if (n_ev and "sarcomere_only_classification" in ev.columns) else [],
            "assisted_classification": [str(c) for c in ev["assisted_classification"]] if (n_ev and "assisted_classification" in ev.columns) else [],
            "assistance_reason": [str(r) for r in ev["assistance_reason"]] if (n_ev and "assistance_reason" in ev.columns) else [],
        },
    }


def _calcium_run_payload(cal_run_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured, JSON-safe run dictionary for calcium preset."""
    ev = cal_run_dict.get("events")
    p = cal_run_dict.get("params")
    qc_meta = cal_run_dict.get("qc_meta", {})
    n_ev = len(ev) if ev is not None and not ev.empty else 0
    return {
        "params": {
            "min_height": getattr(p, "min_height", 0.005),
            "min_norm_height": getattr(p, "min_norm_height", 0.05),
            "min_prominence_sigma": getattr(p, "min_prominence_sigma", 3.0),
            "min_prominence_abs": getattr(p, "min_prominence_abs", 0.005),
            "min_recovery": getattr(p, "min_recovery", 0.25),
            "min_separation_s": getattr(p, "min_separation_s", 0.20),
        },
        "qc_status": str(cal_run_dict.get("qc_status", "UNKNOWN")),
        "qc_reason": str(cal_run_dict.get("qc_reason", "")),
        "analyzable_duration_s": _num(qc_meta.get("analyzable_duration_s")),
        "frequency_hz": _num(cal_run_dict.get("frequency_hz")),
        "frequency_bpm": _num(cal_run_dict.get("frequency_bpm")),
        "noise_sigma": _num(cal_run_dict.get("raw_noise_sigma", qc_meta.get("noise_sigma"))),
        "amplitude_threshold": _num(cal_run_dict.get("stage1_amplitude_criterion", cal_run_dict.get("effective_amplitude_threshold", getattr(p, "min_norm_height", 0.05)))),
        "stage1_amplitude_criterion": _num(cal_run_dict.get("stage1_amplitude_criterion")),
        "reference_amplitude": _num(cal_run_dict.get("reference_amplitude")),
        "effective_prominence": _num(cal_run_dict.get("effective_prominence")),
        "prominence_noise_floor": _num(cal_run_dict.get("prominence_noise_floor")),
        "n_accepted": int(cal_run_dict.get("n_accepted", 0)),
        "n_main": int(cal_run_dict.get("n_main", cal_run_dict.get("n_accepted", 0))),
        "n_secondary": int(cal_run_dict.get("n_secondary", 0)),
        "threshold_main": _num(cal_run_dict.get("threshold_main", cal_run_dict.get("stage1_amplitude_criterion", cal_run_dict.get("effective_amplitude_threshold")))),
        "threshold_sce": _num(cal_run_dict.get("threshold_sce", getattr(p, "min_norm_height_sce", 0.08))),
        "n_stage1_accepted": cal_run_dict.get("n_stage1_accepted"),
        "n_stage2_recovered": cal_run_dict.get("n_stage2_recovered"),
        "events": {
            "index": [int(i) for i in ev["peak_index"]] if n_ev else [],
            "time_s": [float(t) for t in ev["time_s"]] if n_ev else [],
            "amplitude": [_num(a) for a in ev["amplitude"]] if n_ev else [],
            "delta_r": [_num(dr) for dr in ev["delta_r"]] if (n_ev and "delta_r" in ev.columns) else [],
            "raw_value": [_num(r) for r in ev["peak_raw_value"]] if (n_ev and "peak_raw_value" in ev.columns) else [],
            "local_baseline": [_num(b) for b in ev["local_baseline_value"]] if (n_ev and "local_baseline_value" in ev.columns) else [],
            "status": [str(s) for s in ev["status"]] if n_ev else [],
            "event_class": [str(c) for c in ev["event_class"]] if (n_ev and "event_class" in ev.columns) else (["MAIN" if s == "ACCEPTED" else "REJECTED" for s in ev["status"]] if n_ev else []),
            "reason": [str(s) for s in ev["reason"]] if n_ev else [],
            "diagnostic_status": [str(s) for s in ev["status"]] if n_ev else [],
            "diagnostic_reason": [str(s) for s in ev["reason"]] if n_ev else [],
            "detection_stage": [str(ds) for ds in ev["detection_stage"]] if (n_ev and "detection_stage" in ev.columns) else (["stage1" if s == "ACCEPTED" else "none" for s in ev["status"]] if n_ev else []),
        }
    }


def process_sarcomere_workbook(
    file_path: Union[str, Path],
    segments: Optional[List[int]] = None,
    analysis_mode: str = "sarcomere",
    params: Optional[SarcomereDetectorParams] = None,
    enable_assistance: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Execute complete sarcomere analysis across segments of an experimental workbook.

    Parameters
    ----------
    file_path : str or Path
        Path to the experimental Excel workbook.
    segments : Optional[List[int]]
        Optional subset of segment numbers to process.
    analysis_mode : str
        'sarcomere' (SARCOMERE ONLY) or 'sarcomere-calcium' (SARCOMERE + CALCIUM).
    params : Optional[SarcomereDetectorParams]
        Detector strictness parameters.
    enable_assistance : bool
        Default calcium assistance state in 'sarcomere-calcium' mode (default True).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Workbook not found: {file_path}")

    mode_clean = str(analysis_mode).strip().lower()
    if mode_clean not in ("sarcomere", "sarcomere-calcium"):
        raise ValueError(f"Unknown analysis mode '{analysis_mode}'. Must be 'sarcomere' or 'sarcomere-calcium'.")

    ch_info = discover_sarcomere_workbook_channels(p)
    sarc_sheets = ch_info["sarcomere_segments"]
    if not sarc_sheets:
        raise ValueError(f"No SarcomereLength segment sheets found in workbook: {p.name}")

    if segments:
        wanted = set(int(s) for s in segments)
        sarc_sheets = {seg: sheet for seg, sheet in sarc_sheets.items() if seg in wanted}

    print(f"\nProcessing {p.name} [{mode_clean.upper()} mode] across {len(sarc_sheets)} segments...")

    segments_data: List[Dict[str, Any]] = []

    for seg_num, sarc_sheet in sorted(sarc_sheets.items()):
        try:
            if mode_clean == "sarcomere-calcium":
                res = analyze_sarcomere_segment_from_excel(
                    p,
                    sheet_name=sarc_sheet,
                    mode=mode_clean,
                    params=params,
                    enable_assistance=False,
                )
                res_unasst = res
            else:
                res = analyze_sarcomere_segment_from_excel(
                    p,
                    sheet_name=sarc_sheet,
                    mode=mode_clean,
                    params=params,
                    enable_assistance=False,
                )
                res_unasst = res
        except Exception as exc:
            print(f"  [!] Segment {seg_num}: Error during analysis: {exc}")
            continue

        sample_id = extract_sample_id_from_segment_sheet(p, sarc_sheet)
        sig_raw = res.get("sig_raw", np.array([], dtype=float))
        l0_curve = res.get("l0_baseline_um")
        artifact_info = res.get("artifact_info", {})
        art_intervals = artifact_info.get("artifact_intervals_s", [])

        # Delta L(t) = L0(t) - L(t) (positive shortening excursion)
        short_arr = None
        pct_short_arr = None
        if l0_curve is not None and len(l0_curve) == len(sig_raw) and len(sig_raw) > 0:
            short_arr = l0_curve - sig_raw
            pct_short_arr = 100.0 * short_arr / np.maximum(0.1, l0_curve)

        runs = {}
        preset_runs = res.get("preset_runs", {})
        for r_name in ["default", "permissive", "strict"]:
            if r_name in preset_runs:
                runs[r_name] = _sarcomere_run_payload(preset_runs[r_name])
            elif "default" in preset_runs:
                runs[r_name] = _sarcomere_run_payload(preset_runs["default"])

        unassisted_runs = {}
        if mode_clean == "sarcomere-calcium":
            unasst_preset_runs = res_unasst.get("preset_runs", {})
            for r_name in ["default", "permissive", "strict"]:
                if r_name in unasst_preset_runs:
                    unassisted_runs[r_name] = _sarcomere_run_payload(unasst_preset_runs[r_name])
                elif "default" in unasst_preset_runs:
                    unassisted_runs[r_name] = _sarcomere_run_payload(unasst_preset_runs["default"])
        else:
            unassisted_runs = runs

        # Calcium payloads in dual mode
        sig_cal_b64 = None
        baseline_cal_b64 = None
        cal_runs_payload = None
        cal_meta_payload = None
        cal_t0_val = None
        cal_fs_val = None
        cal_len_val = 0

        cal_pipeline_res = res.get("cal_pipeline_res")
        if mode_clean == "sarcomere-calcium" and cal_pipeline_res is not None:
            raw_cal = cal_pipeline_res.get("signal_raw")
            base_cal = cal_pipeline_res.get("baseline")
            sig_cal_b64 = b64_encode_array(raw_cal)
            baseline_cal_b64 = b64_encode_array(base_cal)
            cal_meta = cal_pipeline_res.get("metadata", {})
            cal_meta_payload = cal_meta
            cal_t0_val = float(cal_meta.get("t0", 0.0))
            cal_fs_val = float(cal_meta.get("fs", 250.0))
            cal_len_val = int(cal_meta.get("length", len(raw_cal) if raw_cal is not None else 0))

            c_runs = cal_pipeline_res.get("runs", {})
            cal_runs_payload = {
                "default": _calcium_run_payload(c_runs["default"]),
                "permissive": _calcium_run_payload(c_runs["permissive"]),
                "strict": _calcium_run_payload(c_runs["strict"]),
            }

        cal_events_payload = None
        cal_events = res.get("calcium_events")
        if cal_events is not None and not cal_events.empty:
            cal_events_payload = {
                "index": [int(i) for i in cal_events["peak_index"]],
                "time_s": [float(t) for t in cal_events["time_s"]],
                "amplitude": [_num(a) for a in cal_events["amplitude"]],
                "delta_r": [_num(dr) for dr in cal_events["delta_r"]] if "delta_r" in cal_events.columns else [],
                "raw_value": [_num(r) for r in cal_events["peak_raw_value"]] if "peak_raw_value" in cal_events.columns else [],
                "local_baseline": [_num(b) for b in cal_events["local_baseline_value"]] if "local_baseline_value" in cal_events.columns else [],
                "status": [str(s) for s in cal_events["status"]],
                "calcium_match_status": [str(m) for m in cal_events["calcium_match_status"]] if "calcium_match_status" in cal_events.columns else [],
                "matched_sarcomere_time_s": [_num(t) for t in cal_events["matched_sarcomere_time_s"]] if "matched_sarcomere_time_s" in cal_events.columns else [],
                "sarcomere_amplitude_um": [_num(a) for a in cal_events["sarcomere_amplitude_um"]] if "sarcomere_amplitude_um" in cal_events.columns else [],
            }

        comb_qc = res.get("combined_qc", {})

        segments_data.append({
            "sheet_name": sarc_sheet,
            "segment_number": seg_num,
            "sample_id": str(sample_id) if pd.notna(sample_id) else "",
            "channel": "SarcomereLength",
            "analysis_mode": mode_clean,
            "fs": float(res.get("fs", 250.0)),
            "t0": float(res.get("t0", 0.0)),
            "length": int(len(sig_raw)),
            "qc_status": str(res.get("qc_status", "UNKNOWN")),
            "qc_reason": str(res.get("qc_reason", "")),
            "calcium_qc_status": str(res.get("calcium_qc_status", "UNAVAILABLE")),
            "calcium_qc_reason": str(res.get("calcium_qc_reason", "")),
            "combined_qc": {
                "combined_qc_status": str(comb_qc.get("combined_qc_status", res.get("qc_status", "UNKNOWN"))),
                "combined_qc_reason": str(comb_qc.get("combined_qc_reason", "sarcomere_only_mode")),
                "description": str(comb_qc.get("description", "")),
            },
            "sig_raw_b64": b64_encode_array(sig_raw),
            "l0_baseline_b64": b64_encode_array(l0_curve),
            "shortening_b64": b64_encode_array(short_arr),
            "pct_shortening_b64": b64_encode_array(pct_short_arr),
            "artifact_intervals": [[float(iv[0]), float(iv[1])] for iv in art_intervals],
            "reconstruction": res.get("reconstruction", {}),
            "reconstructed_intervals": [
                [float(iv["start_time_s"]), float(iv["end_time_s"])]
                for iv in res.get("reconstruction", {}).get("reconstructed_intervals", [])
            ],
            "runs": runs,
            "unassisted_runs": unassisted_runs,
            # Dual mode auxiliary data
            "sig_cal_b64": sig_cal_b64,
            "baseline_cal_b64": baseline_cal_b64,
            "t0_cal": cal_t0_val,
            "fs_cal": cal_fs_val,
            "length_cal": cal_len_val,
            "cal_status": "loaded" if sig_cal_b64 else "not_present",
            "cal_runs": cal_runs_payload,
            "cal_metadata": cal_meta_payload,
            "calcium_events": cal_events_payload,
            "estimated_emd_delay_s": _num(res.get("estimated_emd_delay_s")),
            "emd_delay_status": str(res.get("emd_delay_status", "UNAVAILABLE")),
            "n_matched_pairs": int(res.get("n_matched_pairs", 0)),
            "n_unmatched_sarc": int(res.get("n_unmatched_sarc", 0)),
            "n_unmatched_cal": int(res.get("n_unmatched_cal", 0)),
            "sarcomere_only_counts": {
                "n_accepted": int(res.get("sarcomere_only_accepted", 0)),
                "n_main": int(res.get("sarcomere_only_main", 0)),
                "n_secondary": int(res.get("sarcomere_only_secondary", 0)),
                "n_ambiguous": int(res.get("sarcomere_only_ambiguous", 0)),
            },
            "assisted_counts": {
                "n_accepted": int(res.get("assisted_accepted", 0)),
                "n_main": int(res.get("assisted_main", 0)),
                "n_secondary": int(res.get("assisted_secondary", 0)),
                "n_ambiguous": int(res.get("assisted_ambiguous", 0)),
            },
        })

    return segments_data, p.name



# Unified review template is maintained in run_v2_review_utility.HTML_TEMPLATE
from run_v2_review_utility import HTML_TEMPLATE as SARCOMERE_HTML_TEMPLATE


def generate_sarcomere_html(
    segments_data: List[Dict[str, Any]],
    source_workbook: str,
    out_path: Optional[Union[str, Path]] = None,
    default_run: str = "default",
    workbook_path: Optional[Union[str, Path]] = None,
    analysis_mode: str = "sarcomere",
    segment_rejections: Optional[Dict[str, Any]] = None,
    output_html_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Generate a self-contained interactive HTML review report using the unified review interface."""
    target_out = out_path or output_html_path
    if target_out is None:
        raise ValueError("Either out_path or output_html_path must be specified.")
    from run_v2_review_utility import generate_html
    generate_html(
        segments_data=segments_data,
        source_workbook=source_workbook,
        out_path=target_out,
        default_run=default_run,
        workbook_path=workbook_path,
        analysis_mode=analysis_mode,
        segment_rejections=segment_rejections,
    )
    return Path(target_out)
