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



SARCOMERE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cardiomyocyte Analysis — Sarcomere Shortening Review</title>
    <script>__PLOTLY_JS__</script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 18px; color: #212529; background: #fafbfc; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 8px; }
        .badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; color: white; display: inline-block; }
        .badge-sarc { background: #198754; }
        .badge-comb { background: #6f42c1; }
        .info { margin-bottom: 10px; font-size: 13px; color: #495057; font-weight: 500; }
        .controls { display: flex; gap: 12px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; background: #ffffff; padding: 10px 14px; border: 1px solid #dee2e6; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
        .btn-nav { padding: 5px 12px; cursor: pointer; border: 1px solid #ced4da; border-radius: 4px; background: #f8f9fa; font-weight: bold; font-size: 13px; }
        .btn-nav:hover:not(:disabled) { background: #e9ecef; }
        .btn-nav:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-primary { background: #0d6efd; color: white; border-color: #0d6efd; }
        .btn-primary:hover { background: #0b5ed7; }
        .btn-secondary { background: #6c757d; color: white; border-color: #6c757d; }
        .btn-secondary:hover { background: #5c636a; }
        select, input[type="text"] { padding: 5px 8px; border-radius: 4px; border: 1px solid #ced4da; font-size: 13px; }
        .fail { background: #f8d7da; color: #842029; padding: 8px 12px; border: 1px solid #f5c2c7; margin-bottom: 8px; font-weight: bold; border-radius: 4px; font-size: 13px; }
        .params { margin-bottom: 8px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; background: #f1f3f5; padding: 7px 10px; border-radius: 4px; border: 1px solid #e9ecef; color: #343a40; }
        .edit-panel { background: #e7f1ff; border: 1px solid #b6d4fe; padding: 8px 14px; border-radius: 6px; margin-bottom: 8px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-size: 13px; }
        .detail-card { background: #ffffff; border: 1px solid #dee2e6; border-left: 4px solid #0d6efd; padding: 8px 14px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
        .comparison-panel { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 8px 14px; margin-bottom: 8px; font-size: 13px; }
        .summary { margin-top: 24px; border-top: 1px solid #dee2e6; padding-top: 16px; }
        table { border-collapse: collapse; width: 100%; font-size: 12px; background: white; }
        th, td { border: 1px solid #dee2e6; padding: 6px 8px; text-align: left; }
        th { background: #f1f3f5; font-weight: 600; color: #495057; }
        tr:hover { background: #f8f9fa; }
        tr.selected-row { background: #e7f1ff !important; font-weight: bold; }
        #plot { width: 100%; min-height: 540px; background: white; border-radius: 6px; border: 1px solid #dee2e6; }
    </style>
</head>
<body>
    <div class="header">
        <h2 style="margin:0; font-size: 20px; color: #111;">Cardiomyocyte Analysis — Sarcomere Shortening Review</h2>
        <span id="analysis_mode_badge" class="badge">MODE: __ANALYSIS_MODE__</span>
    </div>
    <div class="info">
        Workbook: <span id="source_name" style="font-weight:bold; color:#0d6efd;"></span> &nbsp;|&nbsp;
        SHA256: <span style="font-family:monospace; font-size:11px; background:#e9ecef; padding:2px 5px; border-radius:3px;">__SHA256__</span> &nbsp;|&nbsp;
        Channel: <span style="font-weight:bold; color:#198754;">SarcomereLength (µm)</span>
    </div>

    <!-- Main Navigation & Global Controls -->
    <div class="controls">
        <div style="display:flex; align-items:center; gap:6px;">
            <label for="segment_select" style="font-weight:bold;">Segment:</label>
            <button id="btn_prev" class="btn-nav" type="button" title="Previous segment (Left Arrow or 'p')">&larr; Previous</button>
            <select id="segment_select"></select>
            <button id="btn_next" class="btn-nav" type="button" title="Next segment (Right Arrow or 'n')">Next &rarr;</button>
        </div>
        <label>Detector strictness: <select id="run_select"></select></label>
        <label id="lbl_show_reconstruction" style="margin-left:8px; font-weight:normal; font-size:12px; display:inline-flex; align-items:center; gap:4px; cursor:pointer;" title="Show reconstructed intervals (diagnostic only)">
            <input type="checkbox" id="chk_reconstruction"> Show reconstructed intervals (diagnostic only)
        </label>
        <label><input type="checkbox" id="chk_show_artifacts"> Show detected artifacts (diagnostic)</label>
        <label><input type="checkbox" id="chk_rejected"> Show rejected candidates (diagnostic)</label>
        <label><input type="checkbox" id="chk_pct_short" checked> View % Shortening (ΔL/L₀)</label>
        <div style="background:#fff0f0; padding:4px 8px; border-radius:4px; border:1px solid #f5c2c7; display:inline-flex; align-items:center; gap:6px;">
            <label style="color:#842029; font-weight:bold; cursor:pointer;" title="Exclude entire recording from reportable summary statistics">
                <input type="checkbox" id="chk_segment_rejected"> Reject entire segment
            </label>
            <select id="segment_rejection_reason" style="display:none; font-size:12px; padding:2px 5px;">
                <option value="Severe vertical tracking artifacts">Severe vertical tracking artifacts</option>
                <option value="Noise-dominated / uninterpretable">Noise-dominated / uninterpretable</option>
                <option value="Incorrect recording or alignment">Incorrect recording or alignment</option>
                <option value="Other / researcher decision">Other / researcher decision</option>
            </select>
        </div>
        <button id="btn_save" class="btn-nav">Save annotations (CSV)</button>
        <button id="btn_export_overrides" class="btn-nav btn-primary">Export overrides (JSON)</button>
        <button id="btn_import_trigger" class="btn-nav btn-secondary">Import overrides (JSON)</button>
        <input type="file" id="file_import" accept=".json" style="display:none;">
    </div>

    <!-- Event Inspection & Manual Correction Panel -->
    <div id="peak_strictness_panel" class="edit-panel">
        <span style="font-weight:bold; color:#084298;">Event Correction & Strictness:</span>
        <label style="font-weight:bold;">Selected event:
            <select id="peak_select" style="font-size:12px; padding:3px 6px; max-width:280px;">
                <option value="none">-- Select event --</option>
            </select>
        </label>
        <label style="font-weight:bold;">Contraction class:
            <select id="event_class_select" disabled>
                <option value="MAIN">MAIN (circle)</option>
                <option value="SECONDARY">SECONDARY (diamond)</option>
                <option value="AMBIGUOUS">AMBIGUOUS (triangle)</option>
                <option value="restore">Restore automatic class</option>
            </select>
        </label>
        <label style="font-weight:bold;">Per-event strictness:
            <select id="peak_strictness_select" disabled>
                <option value="inherit">Use global setting</option>
                <option value="permissive">Permissive</option>
                <option value="default">Default</option>
                <option value="strict">Strict</option>
            </select>
        </label>
        <button id="btn_remove_manual" class="btn-nav" style="display:none; color:#dc3545; border-color:#dc3545;" type="button">Delete manual peak</button>
        <span style="font-size:12px; color:#495057; margin-left:auto;">
            <b>Acceptance:</b> <span style="color:#28a745;">&#9679; GREEN</span> = Auto &nbsp;|&nbsp;
            <span style="color:#007bff;">&#9679; BLUE</span> = Manual &nbsp;|&nbsp;
            <span style="color:#dc3545;">&#10006; RED</span> = User-rejected &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Class Shapes:</b> Circle = MAIN &nbsp;|&nbsp; Diamond = SECONDARY &nbsp;|&nbsp; Triangle = AMBIGUOUS
        </span>
    </div>

    <!-- Quantitative Selected Event Details Card -->
    <div id="event_detail_card" class="detail-card" style="display:none;"></div>

    <!-- Side-by-Side Mode Comparison Card (In dual mode) -->
    <div id="comparison_card" class="comparison-panel" style="display:none;"></div>

    <!-- Segment & QC Info Banners -->
    <div id="client_error_banner" class="fail" style="display:none;"></div>
    <div id="segment_info" class="info"></div>
    <div id="fail_box" class="fail" style="display:none;"></div>
    <div id="param_box" class="params"></div>

    <!-- Multi-Panel Plotly Canvas -->
    <div id="plot"></div>

    <!-- Summary Statistics Table across All Segments -->
    <div class="summary">
        <h3 style="margin-top:0; margin-bottom:8px; font-size:15px; color:#333;">Workbook Summary (Click segment row to navigate)</h3>
        <table id="summary_table">
            <thead>
                <tr>
                    <th>Seg</th>
                    <th>Sample ID</th>
                    <th>Duration (s)</th>
                    <th>QC Status</th>
                    <th>QC Reason</th>
                    <th>Accepted</th>
                    <th>MAIN</th>
                    <th>SEC</th>
                    <th>AMB</th>
                    <th>Mean Shortening (%)</th>
                    <th>Contraction Freq (Hz)</th>
                    <th>MAIN Freq (Hz)</th>
                    <th>SEC Freq (Hz)</th>
                    <th>Manual Adds</th>
                    <th>User Rejections</th>
                    <th>Ca2+ Matches</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <script>
        window.addEventListener('error', function(e) {
            console.error("Sarcomere Review Error:", e);
            const eb = document.getElementById("client_error_banner");
            if (eb) {
                eb.style.display = "block";
                eb.innerHTML = `⚠️ <b>JavaScript Error during report initialization:</b> ${e.message || e}`;
            }
        });

        const segments_data = __DATA__;
        const source_workbook = "__SOURCE__";
        const source_sha256 = "__SHA256__";
        const DEFAULT_RUN = "__DEFAULT_RUN__";
        const initialSegmentRejections = __SEGMENT_REJECTIONS__;
        const ANALYSIS_MODE = "__ANALYSIS_MODE__".toLowerCase();

        // Style the mode badge
        const modeBadge = document.getElementById("analysis_mode_badge");
        if (modeBadge) {
            if (ANALYSIS_MODE === "sarcomere-calcium") {
                modeBadge.className = "badge badge-comb";
                modeBadge.innerText = "MODE: SARCOMERE + CALCIUM";
                // Calcium assistance toggle removed (zero assistance)
            } else {
                modeBadge.className = "badge badge-sarc";
                modeBadge.innerText = "MODE: SARCOMERE ONLY";
            }
        }

        let currentSegIdx = 0;
        let currentRun = DEFAULT_RUN;
        let assistanceEnabled = false;       // Assistance permanently removed
        let selectedPeakInfo = null;       // { segIdx, index }
        const userSegmentRejection = {};   // segIdx -> {rejected: bool, reason: str}
        const userManualPeaks = {};        // segIdx -> { [index]: {index, time_s, l0, l_min, abs_short, pct_short, event_class, action} }
        const userClassOverrides = {};     // segIdx -> { [index]: classStr }
        const userPerPeakStrictness = {};  // segIdx -> { [index]: presetStr }
        const userPeakRejections = {};     // segIdx -> Set of rejected indices
        const userCalManualPeaks = {};     // segIdx -> { [index]: {cal_idx, index, time_s, amp, raw} }
        const userCalPeakRejections = {};  // segIdx -> Set of rejected calcium indices
        const decodedSignals = {};

        // Load initial rejections
        if (initialSegmentRejections) {
            segments_data.forEach((seg, idx) => {
                const sNum = String(seg.segment_number);
                if (initialSegmentRejections[sNum] && initialSegmentRejections[sNum].rejected) {
                    userSegmentRejection[idx] = {
                        rejected: true,
                        reason: initialSegmentRejections[sNum].reason || "Researcher decision"
                    };
                }
            });
        }

        function decodeFloat32Array(b64) {
            if (!b64) return null;
            const raw = window.atob(b64);
            const array = new Uint8Array(new ArrayBuffer(raw.length));
            for (let i = 0; i < raw.length; i++) array[i] = raw.charCodeAt(i);
            return new Float32Array(array.buffer);
        }

        function getBaseSignals(segIdx) {
            if (!decodedSignals[segIdx]) {
                const s = segments_data[segIdx];
                decodedSignals[segIdx] = {
                    raw: decodeFloat32Array(s.sig_raw_b64),
                    l0: decodeFloat32Array(s.l0_baseline_b64),
                    short: decodeFloat32Array(s.shortening_b64),
                    pct_short: decodeFloat32Array(s.pct_shortening_b64),
                    cal: decodeFloat32Array(s.sig_cal_b64),
                    cal_base: decodeFloat32Array(s.baseline_cal_b64),
                };
            }
            return decodedSignals[segIdx];
        }

        function timesFor(seg) {
            const fs = seg.fs;
            const t0 = seg.t0;
            const n = seg.length;
            const arr = new Float32Array(n);
            for (let i = 0; i < n; i++) arr[i] = t0 + i / fs;
            return arr;
        }

        function timesForCal(seg) {
            if (!seg.sig_cal_b64 || !seg.length_cal) return null;
            const fs = seg.fs_cal || seg.fs;
            const t0 = (seg.t0_cal !== null && seg.t0_cal !== undefined) ? seg.t0_cal : seg.t0;
            const n = seg.length_cal;
            const arr = new Float32Array(n);
            for (let i = 0; i < n; i++) arr[i] = t0 + i / fs;
            return arr;
        }

        // Build active event array for a segment under selected strictness and assistance
        function getActiveEvents(segIdx, runName) {
            runName = runName || currentRun;
            const seg = segments_data[segIdx];
            const activeRuns = seg.runs;
            const run = (activeRuns && activeRuns[runName]) ? activeRuns[runName] : (activeRuns ? activeRuns["default"] : null);
            if (!run || !run.events || !run.events.index) return [];

            const ev = run.events;
            const fs = seg.fs;
            const w = Math.max(1, Math.round(0.04 * fs));
            const manMap = userManualPeaks[segIdx] || {};
            const clsMap = userClassOverrides[segIdx] || {};
            const ppMap = userPerPeakStrictness[segIdx] || {};
            const rejSet = userPeakRejections[segIdx] || new Set();

            const result = [];

            // 1. Process automatic events from run
            for (let i = 0; i < ev.index.length; i++) {
                const idx = ev.index[i];
                let status = ev.status[i];
                let isManRej = false;

                // Check per-peak strictness
                let ovPreset = "inherit";
                for (const [kIdx, pVal] of Object.entries(ppMap)) {
                    if (Math.abs(parseInt(kIdx) - idx) <= w) {
                        ovPreset = pVal;
                        break;
                    }
                }
                if (ovPreset !== "inherit" && activeRuns && activeRuns[ovPreset]) {
                    const altEv = activeRuns[ovPreset].events;
                    const altK = altEv.index.findIndex(ai => Math.abs(ai - idx) <= w);
                    if (altK >= 0) {
                        status = altEv.status[altK];
                    }
                }

                // Check user rejection toggle
                for (const rIdx of rejSet) {
                    if (Math.abs(rIdx - idx) <= w) {
                        status = "REJECTED";
                        isManRej = true;
                        break;
                    }
                }

                // Check classification override
                let finalClass = ev.event_class[i];
                let isClassOverridden = false;
                for (const [cIdx, cVal] of Object.entries(clsMap)) {
                    if (Math.abs(parseInt(cIdx) - idx) <= w) {
                        finalClass = cVal;
                        isClassOverridden = true;
                        break;
                    }
                }

                result.push({
                    index: idx,
                    time_s: ev.time_s[i],
                    l0_um: ev.l0_um[i],
                    l_min_um: ev.l_min_um[i],
                    abs_shortening_um: ev.abs_shortening_um[i],
                    pct_shortening: ev.pct_shortening[i],
                    takeoff_l0_um: ev.takeoff_l0_um[i],
                    local_abs_shortening_um: ev.local_abs_shortening_um[i],
                    local_pct_shortening: ev.local_pct_shortening[i],
                    event_class: finalClass,
                    auto_event_class: ev.event_class[i],
                    is_class_overridden: isClassOverridden,
                    classification_reason: ev.classification_reason[i],
                    status: status,
                    auto_status: ev.status[i],
                    reason: ev.reason[i],
                    origin: "auto",
                    manually_rejected: isManRej,
                    peak_corrupted: ev.peak_corrupted[i],
                    artifact_nearby: ev.artifact_nearby[i],
                    event_nature: ev.event_nature ? ev.event_nature[i] : "MEASURED",
                    recovery_status: ev.recovery_status ? ev.recovery_status[i] : "DIRECT_MEASUREMENT",
                    has_valid_amplitude: ev.has_valid_amplitude ? ev.has_valid_amplitude[i] : true,
                    calcium_match_status: ev.calcium_match_status ? ev.calcium_match_status[i] : "UNAVAILABLE",
                    matched_calcium_time_s: ev.matched_calcium_time_s ? ev.matched_calcium_time_s[i] : null,
                    calcium_amplitude: ev.calcium_amplitude ? ev.calcium_amplitude[i] : null,
                    calcium_mechanical_delay_s: ev.calcium_mechanical_delay_s ? ev.calcium_mechanical_delay_s[i] : null,
                    calcium_onset_delay_s: ev.calcium_onset_delay_s ? ev.calcium_onset_delay_s[i] : null,
                    calcium_matching_confidence: ev.calcium_matching_confidence ? ev.calcium_matching_confidence[i] : "N/A",
                    sarcomere_only_classification: ev.sarcomere_only_classification ? ev.sarcomere_only_classification[i] : null,
                    assisted_classification: ev.assisted_classification ? ev.assisted_classification[i] : null,
                    assistance_reason: ev.assistance_reason ? ev.assistance_reason[i] : null,
                    per_peak_strictness: ovPreset,
                });
            }

            // 2. Add manual additions
            for (const [mIdxStr, mItem] of Object.entries(manMap)) {
                const mIdx = parseInt(mIdxStr);
                const exists = result.some(r => Math.abs(r.index - mIdx) <= w);
                if (!exists) {
                    let mClass = mItem.event_class || "MAIN";
                    for (const [cIdx, cVal] of Object.entries(clsMap)) {
                        if (Math.abs(parseInt(cIdx) - mIdx) <= w) {
                            mClass = cVal;
                            break;
                        }
                    }
                    result.push({
                        index: mIdx,
                        time_s: mItem.time_s,
                        l0_um: mItem.l0,
                        l_min_um: mItem.l_min,
                        abs_shortening_um: mItem.abs_short,
                        pct_shortening: mItem.pct_short,
                        takeoff_l0_um: mItem.l0,
                        local_abs_shortening_um: mItem.abs_short,
                        local_pct_shortening: mItem.pct_short,
                        event_class: mClass,
                        auto_event_class: mClass,
                        is_class_overridden: (mClass !== (mItem.event_class || "MAIN")),
                        classification_reason: "Manual user addition",
                        status: "ACCEPTED",
                        auto_status: "MISSING",
                        reason: "manual_addition",
                        origin: "manual",
                        manually_rejected: false,
                        peak_corrupted: false,
                        artifact_nearby: false,
                        event_nature: "MEASURED",
                        recovery_status: "DIRECT_MEASUREMENT",
                        has_valid_amplitude: true,
                        calcium_match_status: "UNAVAILABLE",
                        matched_calcium_time_s: null,
                        calcium_amplitude: null,
                        calcium_mechanical_delay_s: null,
                        calcium_onset_delay_s: null,
                        calcium_matching_confidence: "N/A",
                        sarcomere_only_classification: mClass,
                        assisted_classification: mClass,
                        assistance_reason: "Manual user addition",
                        per_peak_strictness: "inherit",
                    });
                }
            }

            result.sort((a, b) => a.index - b.index);
            return result;
        }

        // Build active calcium event array
        function getCalAnn(segIdx, runName) {
            runName = runName || currentRun;
            const seg = segments_data[segIdx];
            if (!seg.cal_runs) return [];
            const r = seg.cal_runs[runName] || seg.cal_runs["default"];
            if (!r || !r.events || !r.events.index) return [];

            const ev = r.events;
            const fs = seg.fs_cal || seg.fs;
            const w = Math.max(1, Math.round(0.04 * fs));
            const rejSet = userCalPeakRejections[segIdx] || new Set();
            const manMap = userCalManualPeaks[segIdx] || {};

            const result = [];
            for (let i = 0; i < ev.index.length; i++) {
                const idx = ev.index[i];
                let status = ev.status[i];
                let isManRej = false;
                for (const rIdx of rejSet) {
                    if (Math.abs(rIdx - idx) <= w) {
                        status = "REJECTED";
                        isManRej = true;
                        break;
                    }
                }
                const evCls = ev.event_class ? ev.event_class[i] : (status === "ACCEPTED" ? "MAIN" : "REJECTED");
                result.push({
                    cal_idx: i + 1,
                    index: idx,
                    time_s: ev.time_s[i],
                    amp: ev.amplitude ? ev.amplitude[i] : null,
                    delta_r: ev.delta_r ? ev.delta_r[i] : null,
                    raw: ev.raw_value ? ev.raw_value[i] : null,
                    base: ev.local_baseline ? ev.local_baseline[i] : null,
                    status: status,
                    auto_status: ev.status[i],
                    event_class: evCls,
                    origin: "auto",
                    manually_rejected: isManRej,
                    reason: ev.reason ? ev.reason[i] : "",
                    detection_stage: ev.detection_stage ? ev.detection_stage[i] : "stage1",
                });
            }

            for (const [mIdxStr, mItem] of Object.entries(manMap)) {
                const mIdx = parseInt(mIdxStr);
                const exists = result.some(item => Math.abs(item.index - mIdx) <= w);
                if (!exists) {
                    result.push({
                        cal_idx: mItem.cal_idx || (result.length + 1),
                        index: mIdx,
                        time_s: mItem.time_s,
                        amp: mItem.amp,
                        delta_r: mItem.delta_r || mItem.amp,
                        raw: mItem.raw,
                        base: mItem.base,
                        status: "ACCEPTED",
                        auto_status: "MISSING",
                        event_class: mItem.event_class || (mItem.amp >= (r.threshold_main || 0.22) ? "MAIN" : "SECONDARY"),
                        origin: "manual",
                        manually_rejected: false,
                        reason: "manual_addition",
                        detection_stage: "manual",
                    });
                }
            }
            result.sort((a, b) => a.index - b.index);
            return result;
        }

        // Compute counts
        function countsFor(segIdx, runName) {
            const seg = segments_data[segIdx];
            const isManualRej = userSegmentRejection[segIdx] && userSegmentRejection[segIdx].rejected;
            const isQCReject = (seg.qc_status !== "PASS");

            const evs = getActiveEvents(segIdx, runName);
            const autoAcc = evs.filter(a => a.origin === "auto" && a.auto_status === "ACCEPTED").length;
            const manualAdds = evs.filter(a => a.origin === "manual" && a.status === "ACCEPTED").length;
            const userRejs = evs.filter(a => a.manually_rejected).length;

            const accEvents = evs.filter(a => a.status === "ACCEPTED");
            const finalAcc = (isManualRej || isQCReject) ? 0 : accEvents.length;
            const nMain = (isManualRej || isQCReject) ? 0 : accEvents.filter(a => a.event_class === "MAIN").length;
            const nSec = (isManualRej || isQCReject) ? 0 : accEvents.filter(a => a.event_class === "SECONDARY").length;
            const nAmb = (isManualRej || isQCReject) ? 0 : accEvents.filter(a => a.event_class === "AMBIGUOUS").length;

            // Shortening metrics (only valid finite values)
            let meanPct = null;
            let meanAbs = null;
            if (finalAcc > 0) {
                const validPct = accEvents.map(a => a.pct_shortening).filter(v => v !== null && isFinite(v));
                const validAbs = accEvents.map(a => a.abs_shortening_um).filter(v => v !== null && isFinite(v));
                if (validPct.length > 0) meanPct = validPct.reduce((s, q) => s + q, 0) / validPct.length;
                if (validAbs.length > 0) meanAbs = validAbs.reduce((s, q) => s + q, 0) / validAbs.length;
            }

            return {
                auto: autoAcc,
                added: manualAdds,
                rej: userRejs,
                final: finalAcc,
                main: nMain,
                sec: nSec,
                amb: nAmb,
                mean_pct: meanPct,
                mean_abs: meanAbs,
                is_rejected: isManualRej || isQCReject,
            };
        }

        function populateSegmentSelector() {
            const sel = document.getElementById("segment_select");
            sel.innerHTML = "";
            segments_data.forEach((seg, i) => {
                const opt = document.createElement("option");
                opt.value = i;
                const c = countsFor(i, currentRun);
                const rejFlag = c.is_rejected ? " [REJECTED]" : "";
                opt.innerText = `Seg ${seg.segment_number} (${seg.sheet_name}) - ${c.final} acc${rejFlag}`;
                sel.appendChild(opt);
            });
            sel.value = currentSegIdx;
        }

        function updateNavButtons() {
            document.getElementById("btn_prev").disabled = (currentSegIdx <= 0);
            document.getElementById("btn_next").disabled = (currentSegIdx >= segments_data.length - 1);
            document.getElementById("segment_select").value = currentSegIdx;
        }

        function updateSummaryTable() {
            const tbody = document.querySelector("#summary_table tbody");
            tbody.innerHTML = "";
            segments_data.forEach((seg, idx) => {
                const tr = document.createElement("tr");
                if (idx === currentSegIdx) tr.className = "selected-row";
                tr.style.cursor = "pointer";
                tr.addEventListener("click", () => {
                    currentSegIdx = idx;
                    selectedPeakInfo = null;
                    renderPlot();
                });

                const c = countsFor(idx, currentRun);
                const dur = (seg.runs && seg.runs[currentRun] && seg.runs[currentRun].analyzable_duration_s) ? seg.runs[currentRun].analyzable_duration_s : (seg.length / seg.fs);
                const freqStr = (c.final > 0 && dur > 0 && !c.is_rejected) ? (c.final / dur).toFixed(2) : "0.00";
                const mainFreqStr = (c.main > 0 && dur > 0 && !c.is_rejected) ? (c.main / dur).toFixed(2) : "0.00";
                const secFreqStr = (c.sec > 0 && dur > 0 && !c.is_rejected) ? (c.sec / dur).toFixed(2) : "0.00";
                const meanPctStr = (c.mean_pct !== null && !c.is_rejected) ? c.mean_pct.toFixed(2) + "%" : "N/A";

                let qcBadge = "";
                const isManRej = userSegmentRejection[idx] && userSegmentRejection[idx].rejected;
                if (isManRej) {
                    qcBadge = `<span style="color:#842029; font-weight:bold;">MANUAL_REJECT</span>`;
                } else if (seg.qc_status === "PASS") {
                    qcBadge = `<span style="color:#198754; font-weight:bold;">PASS</span>`;
                } else {
                    qcBadge = `<span style="color:#dc3545; font-weight:bold;">REJECT</span>`;
                }

                let calMatchStr = "N/A";
                if (ANALYSIS_MODE === "sarcomere-calcium") {
                    calMatchStr = `${seg.n_matched_pairs || 0} matched (${seg.calcium_qc_status || "N/A"})`;
                }

                tr.innerHTML = `
                    <td><b>${seg.segment_number}</b></td>
                    <td>${seg.sample_id || "N/A"}</td>
                    <td>${(seg.length / seg.fs).toFixed(1)}</td>
                    <td>${qcBadge}</td>
                    <td style="max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${isManRej ? userSegmentRejection[idx].reason : seg.qc_reason}">${isManRej ? userSegmentRejection[idx].reason : seg.qc_reason}</td>
                    <td><b>${c.final}</b></td>
                    <td>${c.main}</td>
                    <td>${c.sec}</td>
                    <td>${c.amb}</td>
                    <td>${meanPctStr}</td>
                    <td>${freqStr}</td>
                    <td>${mainFreqStr}</td>
                    <td>${secFreqStr}</td>
                    <td>${c.added}</td>
                    <td>${c.rej}</td>
                    <td>${calMatchStr}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Update selected event detail card
        function updateSelectedPeakCard(evs) {
            const card = document.getElementById("event_detail_card");
            const btnRemove = document.getElementById("btn_remove_manual");
            const selClass = document.getElementById("event_class_select");
            const selStrict = document.getElementById("peak_strictness_select");

            if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) {
                card.style.display = "none";
                btnRemove.style.display = "none";
                selClass.disabled = true;
                selStrict.disabled = true;
                return;
            }

            const seg = segments_data[currentSegIdx];
            const fs = seg.fs;
            const w = Math.max(1, Math.round(0.04 * fs));
            const a = evs.find(e => Math.abs(e.index - selectedPeakInfo.index) <= w);

            if (!a) {
                card.style.display = "none";
                btnRemove.style.display = "none";
                selClass.disabled = true;
                selStrict.disabled = true;
                return;
            }

            card.style.display = "block";
            selClass.disabled = false;
            selStrict.disabled = false;
            selClass.value = a.event_class;
            selStrict.value = a.per_peak_strictness || "inherit";

            if (a.origin === "manual") {
                btnRemove.style.display = "inline-block";
            } else {
                btnRemove.style.display = "none";
            }

            let calInfoHtml = "";
            if (ANALYSIS_MODE === "sarcomere-calcium") {
                const calStatus = a.calcium_match_status || "UNAVAILABLE";
                const calTimeStr = (a.matched_calcium_time_s !== null && a.matched_calcium_time_s !== undefined) ? `${a.matched_calcium_time_s.toFixed(3)} s` : "None";
                const delayStr = (a.calcium_mechanical_delay_s !== null && a.calcium_mechanical_delay_s !== undefined) ? `${(a.calcium_mechanical_delay_s * 1000).toFixed(1)} ms` : "N/A";
                const calAmpStr = (a.calcium_amplitude !== null && a.calcium_amplitude !== undefined) ? `${a.calcium_amplitude.toFixed(4)} ΔR/R₀` : "N/A";
                const confStr = a.calcium_matching_confidence || "N/A";
                const reasonStr = a.assistance_reason || "None";

                calInfoHtml = `
                    <div style="margin-top:6px; padding-top:6px; border-top:1px dashed #b6d4fe; font-size:12px; color:#495057;">
                        <b>Electromechanical Coupling:</b> Match: <b style="color:#6f42c1;">${calStatus}</b> (Ca2+ peak: ${calTimeStr}, delay: ${delayStr}, conf: ${confStr}) |
                        Amp: ${calAmpStr} | Reason: <i>${reasonStr}</i>
                    </div>
                `;
            }

            const lMinStr = (a.l_min_um !== null && a.l_min_um !== undefined) ? `${a.l_min_um.toFixed(4)} µm` : `<span style="color:#dc3545;">NaN (corrupted)</span>`;
            const l0Str = (a.l0_um !== null && a.l0_um !== undefined) ? `${a.l0_um.toFixed(4)} µm` : "N/A";
            const absStr = (a.abs_shortening_um !== null && a.abs_shortening_um !== undefined) ? `${a.abs_shortening_um.toFixed(4)} µm` : `<span style="color:#dc3545;">N/A (corrupted trough)</span>`;
            const pctStr = (a.pct_shortening !== null && a.pct_shortening !== undefined) ? `${a.pct_shortening.toFixed(2)}%` : `<span style="color:#dc3545;">N/A (corrupted trough)</span>`;
            const takeoffStr = (a.takeoff_l0_um !== null && a.takeoff_l0_um !== undefined) ? `${a.takeoff_l0_um.toFixed(4)} µm` : "N/A";
            const locAbsStr = (a.local_abs_shortening_um !== null && a.local_abs_shortening_um !== undefined) ? `${a.local_abs_shortening_um.toFixed(4)} µm` : "N/A";
            const locPctStr = (a.local_pct_shortening !== null && a.local_pct_shortening !== undefined) ? `${a.local_pct_shortening.toFixed(2)}%` : "N/A";

            const statusColor = a.status === "ACCEPTED" ? (a.origin === "manual" ? "#007bff" : "#198754") : "#dc3545";
            const classColor = a.event_class === "MAIN" ? "#198754" : (a.event_class === "SECONDARY" ? "#fd7e14" : "#6f42c1");
            const modifiedTag = a.is_class_overridden ? `<span style="background:#fff3cd; color:#856404; padding:1px 5px; border-radius:3px; font-weight:bold;">USER OVERRIDDEN</span>` : "";

            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-weight:bold; font-size:14px; color:#084298;">
                        Event at t = ${a.time_s.toFixed(3)} s (sample ${a.index})
                    </span>
                    <span>
                        Nature: <b>${a.event_nature || "MEASURED"}</b> &nbsp;|&nbsp;
                        Recovery: <b>${a.recovery_status || "DIRECT_MEASUREMENT"}</b> &nbsp;|&nbsp;
                        Status: <b style="color:${statusColor};">${a.status}</b> (${a.reason || "accepted"}) &nbsp;|&nbsp;
                        Class: <b style="color:${classColor};">${a.event_class}</b> ${modifiedTag} &nbsp;|&nbsp;
                        Origin: <b>${a.origin}</b>
                    </span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:6px; font-size:12px;">
                    <div>Diastolic L₀: <b>${l0Str}</b></div>
                    <div>Trough L_min: <b>${lMinStr}</b></div>
                    <div>Absolute Shortening ΔL: <b>${absStr}</b></div>
                    <div>Percentage Shortening: <b>${pctStr}</b></div>
                    <div>Takeoff L₀: <b>${takeoffStr}</b></div>
                    <div>Local Shortening: <b>${locAbsStr} (${locPctStr})</b></div>
                    <div>Proximity to artifact: <b>${a.artifact_nearby ? "YES" : "No"}</b></div>
                    <div>Auto class: <b>${a.auto_event_class}</b></div>
                </div>
                ${calInfoHtml}
            `;
        }

        // Render side-by-side comparison card in dual mode
        function updateComparisonCard() {
            const card = document.getElementById("comparison_card");
            if (ANALYSIS_MODE !== "sarcomere-calcium") {
                card.style.display = "none";
                return;
            }

            const seg = segments_data[currentSegIdx];
            const sOnly = seg.sarcomere_only_counts || {};
            const asst = seg.assisted_counts || {};
            const dur = (seg.runs && seg.runs[currentRun] && seg.runs[currentRun].analyzable_duration_s) ? seg.runs[currentRun].analyzable_duration_s : (seg.length / seg.fs);

            const sMain = sOnly.n_main || 0;
            const sSec = sOnly.n_secondary || 0;
            const aMain = asst.n_main || 0;
            const aSec = asst.n_secondary || 0;

            const sMainFreq = dur > 0 ? (sMain / dur).toFixed(2) : "N/A";
            const sSecFreq = dur > 0 ? (sSec / dur).toFixed(2) : "N/A";
            const aMainFreq = dur > 0 ? (aMain / dur).toFixed(2) : "N/A";
            const aSecFreq = dur > 0 ? (aSec / dur).toFixed(2) : "N/A";

            const emdStr = (seg.estimated_emd_delay_s !== null && seg.estimated_emd_delay_s !== undefined) ? `${(seg.estimated_emd_delay_s * 1000).toFixed(1)} ms` : "N/A";

            card.style.display = "block";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <b style="color:#495057;">Analysis Mode Comparison (Segment ${seg.segment_number})</b>
                    <span style="font-size:12px; color:#6c757d;">Estimated EMD delay (τ_EMD): <b style="color:#0d6efd;">${emdStr}</b> (${seg.emd_delay_status || "N/A"})</span>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:12px;">
                    <div style="background:white; padding:8px 10px; border-radius:4px; border:1px solid #ced4da;">
                        <b style="color:#198754;">Sarcomere-Only Analysis (Independent)</b><br>
                        Accepted: <b>${sOnly.n_accepted || 0}</b> (MAIN: ${sMain}, SEC: ${sSec})<br>
                        MAIN Rate: <b>${sMainFreq} Hz</b> | SEC Rate: <b>${sSecFreq} Hz</b>
                    </div>
                    <div style="background:white; padding:8px 10px; border-radius:4px; border:1px solid #b6d4fe;">
                        <b style="color:#6f42c1;">Calcium-Assisted Analysis (Recovery Active)</b><br>
                        Accepted: <b>${asst.n_accepted || 0}</b> (MAIN: ${aMain}, SEC: ${aSec})<br>
                        MAIN Rate: <b>${aMainFreq} Hz</b> | SEC Rate: <b>${aSecFreq} Hz</b>
                    </div>
                </div>
            `;
        }

        function populatePeakSelector(evs) {
            const sel = document.getElementById("peak_select");
            sel.innerHTML = '<option value="none">-- Select event --</option>';
            evs.forEach(e => {
                const opt = document.createElement("option");
                opt.value = e.index;
                const dLStr = (e.abs_shortening_um !== null && e.abs_shortening_um !== undefined) ? `${e.abs_shortening_um.toFixed(3)} µm` : "N/A";
                opt.innerText = `t=${e.time_s.toFixed(3)}s [${e.event_class}] (${dLStr}) - ${e.status}`;
                sel.appendChild(opt);
            });
            if (selectedPeakInfo && selectedPeakInfo.segIdx === currentSegIdx) {
                sel.value = selectedPeakInfo.index;
            } else {
                sel.value = "none";
            }
        }

        // Render Multi-Panel Plotly Canvas
        function renderPlot() {
            const seg = segments_data[currentSegIdx];
            updateNavButtons();

            // Segment rejection checkbox
            const chkRej = document.getElementById("chk_segment_rejected");
            const selRejReason = document.getElementById("segment_rejection_reason");
            const curManualRej = userSegmentRejection[currentSegIdx];
            if (curManualRej && curManualRej.rejected) {
                chkRej.checked = true;
                selRejReason.style.display = "inline-block";
                selRejReason.value = curManualRej.reason || "Severe vertical tracking artifacts";
            } else {
                chkRej.checked = false;
                selRejReason.style.display = "none";
            }

            const baseSig = getBaseSignals(currentSegIdx);
            const times = timesFor(seg);
            const isManualRej = (curManualRej && curManualRej.rejected);
            const isQCRej = (seg.qc_status !== "PASS" && seg.qc_status !== "PASS_PARTIAL");

            const activeRuns = seg.runs;
            const run = (activeRuns && activeRuns[currentRun]) ? activeRuns[currentRun] : (activeRuns ? activeRuns["default"] : {params: {}, events: {index:[]}});
            const showRej = document.getElementById("chk_rejected").checked;
            const showPct = document.getElementById("chk_pct_short").checked;
            const showArt = document.getElementById("chk_show_artifacts").checked;

            const evs = getActiveEvents(currentSegIdx, currentRun);
            updateSelectedPeakCard(evs);
            populatePeakSelector(evs);
            updateComparisonCard();
            updateSummaryTable();

            // Info banners
            let qcHeaderTxt = "";
            if (isManualRej) {
                qcHeaderTxt = `<span style="background:#f8d7da; color:#842029; padding:2px 6px; border-radius:3px; font-weight:bold;">MANUAL REJECT</span> <small>(${curManualRej.reason})</small>`;
            } else if (seg.qc_status === "PASS") {
                qcHeaderTxt = `<span style="color:#198754; font-weight:bold;">QC: PASS (${seg.qc_reason})</span>`;
            } else {
                qcHeaderTxt = `<span style="color:#dc3545; font-weight:bold;">QC: REJECT (${seg.qc_reason})</span>`;
            }

            let auxCalTxt = "";
            if (ANALYSIS_MODE === "sarcomere-calcium") {
                const calCol = seg.calcium_qc_status === "PASS" ? "#198754" : "#dc3545";
                auxCalTxt = ` | <span style="font-weight:bold; color:#6f42c1;">Ca2+ QC:</span> <b style="color:${calCol};">${seg.calcium_qc_status}</b> (${seg.calcium_qc_reason})`;
            }

            document.getElementById("segment_info").innerHTML =
                `<b>${seg.sheet_name}</b> | Sample: ${seg.sample_id || "N/A"} | ${seg.length} pts | fs: ${seg.fs} Hz | t₀: ${seg.t0.toFixed(3)}s${auxCalTxt} | ${qcHeaderTxt}`;

            const fb = document.getElementById("fail_box");
            if (isManualRej) {
                fb.className = "fail";
                fb.style.display = "block";
                fb.innerHTML = `⚠️ <b>MANUALLY REJECTED by researcher:</b> ${curManualRej.reason}. Final reportable accepted contractions: <b>0</b>. Uncheck 'Reject entire segment' above to restore this recording.`;
            } else if (isQCRej) {
                fb.className = "fail";
                fb.style.display = "block";
                fb.innerHTML = `⚠️ <b>Segment Quality Control REJECT:</b> ${seg.qc_reason}. Measurements excluded from reportable scientific statistics.`;
            } else {
                fb.style.display = "none";
            }

            // Detector metrics display
            const counts = countsFor(currentSegIdx, currentRun);
            const dur = (run.analyzable_duration_s !== null && run.analyzable_duration_s !== undefined) ? run.analyzable_duration_s : (seg.length / seg.fs);
            const freqStr = (counts.final > 0 && dur > 0 && !counts.is_rejected) ? `${(counts.final / dur).toFixed(2)} Hz (${((counts.final / dur) * 60).toFixed(1)} BPM)` : "N/A";
            const mainFreqStr = (counts.main > 0 && dur > 0 && !counts.is_rejected) ? `${(counts.main / dur).toFixed(2)} Hz` : "N/A";
            const secFreqStr = (counts.sec > 0 && dur > 0 && !counts.is_rejected) ? `${(counts.sec / dur).toFixed(2)} Hz` : "N/A";

            const p = run.params || {};
            let asstMetricsTxt = "";
            const reconCount = (seg.reconstructed_intervals || []).length;
            if (ANALYSIS_MODE === "sarcomere-calcium") {
                asstMetricsTxt = `  |  Dual Channel: Sarcomere QC=${seg.qc_status} (${seg.qc_reason || ""}), Calcium QC=${seg.calcium_qc_status} (${seg.calcium_qc_reason || ""})  |  Recon intervals: ${reconCount}`;
            } else {
                asstMetricsTxt = `  |  Sarcomere QC: ${seg.qc_status} (${seg.qc_reason || ""})  |  Recon intervals: ${reconCount}`;
            }
            document.getElementById("param_box").innerText =
                `Active detector parameters [${currentRun}]: strictness=${currentRun}  |  ` +
                `min_abs_shortening=${p.min_abs_shortening_um || 0.0065} µm  min_pct_shortening=${p.min_pct_shortening || 0.35}%  ` +
                `min_prominence_sigma=${p.min_prominence_sigma || 4.0}  |  ` +
                `Accepted: ${counts.final} (MAIN: ${counts.main}, SEC: ${counts.sec}, AMB: ${counts.amb})  |  ` +
                `Rate: ${freqStr}  MAIN Rate: ${mainFreqStr}  SEC Rate: ${secFreqStr}  valid duration: ${dur.toFixed(2)}s${asstMetricsTxt}`;

            // Multi-Panel Setup
            const hasCal = (ANALYSIS_MODE === "sarcomere-calcium" && baseSig.cal !== null);
            const panels = [
                { key: "raw", title: "Original SarcomereLength L(t) (µm)<br>(informational, un-inverted)", weight: 2 },
                { key: "short", title: showPct ? "Normalized Shortening % (100·[L₀ - L]/L₀)<br>(active analysis panel)" : "Shortening ΔL(t) = L₀ - L (µm)<br>(active analysis panel)", weight: 3 },
            ];
            if (hasCal) {
                panels.push({ key: "cal", title: "Normalized Calcium ΔR/R₀<br>(relative to resting baseline R₀)", weight: 2 });
            }

            const totalW = panels.reduce((s, q) => s + q.weight, 0);
            const gap = 0.05;
            let top = 1.0;
            panels.forEach((q, i) => {
                const h = (1.0 - gap * (panels.length - 1)) * q.weight / totalW;
                q.domain = [top - h, top];
                top -= (h + gap);
                q.ax = "y" + (i === 0 ? "" : (i + 1));
            });
            const P = {};
            panels.forEach(q => P[q.key] = q);

            const traces = [];

            // Panel 1: Original untouched SarcomereLength L(t) (informational only)
            traces.push({
                x: times, y: baseSig.raw, mode: "lines", name: "SarcomereLength (µm)",
                line: { color: "#1f77b4", width: 1.2 },
                yaxis: P.raw.ax, xaxis: "x", hoverinfo: "x+y", connectgaps: false
            });

            // Panel 2: Normalized Shortening % or Delta L
            const shortY = showPct ? baseSig.pct_short : baseSig.short;
            const thrVal = showPct ? p.min_pct_shortening : p.min_abs_shortening_um;
            if (shortY) {
                traces.push({
                    x: times, y: shortY, mode: "lines", name: showPct ? "% Shortening" : "Shortening ΔL (µm)",
                    line: { color: "#212529", width: 1.2 },
                    yaxis: P.short.ax, xaxis: "x", hoverinfo: "x+y", connectgaps: false
                });
                traces.push({
                    x: [times[0], times[times.length - 1]], y: [0, 0], mode: "lines",
                    name: "Resting baseline (0)", line: { color: "#adb5bd", width: 1 },
                    yaxis: P.short.ax, xaxis: "x", hoverinfo: "skip"
                });
                if (thrVal !== undefined) {
                    traces.push({
                        x: [times[0], times[times.length - 1]], y: [thrVal, thrVal], mode: "lines",
                        name: "Detection threshold", line: { color: "#fd7e14", dash: "dash", width: 1.2 },
                        yaxis: P.short.ax, xaxis: "x", hoverinfo: "skip"
                    });
                }
            }

            // Panel 3: Normalized Calcium ΔR/R₀
            if (hasCal && P.cal) {
                const timesCal = timesForCal(seg) || times;
                traces.push({
                    x: timesCal, y: baseSig.cal, mode: "lines", name: "Normalized calcium ΔR/R₀",
                    line: { color: "#b02a37", width: 1.2 },
                    yaxis: P.cal.ax, xaxis: "x", hoverinfo: "x+y", connectgaps: false
                });
                traces.push({
                    x: [timesCal[0], timesCal[timesCal.length - 1]], y: [0, 0], mode: "lines",
                    name: "Resting baseline (ΔR/R₀ = 0)", line: { color: "#adb5bd", width: 1 },
                    yaxis: P.cal.ax, xaxis: "x", hoverinfo: "skip"
                });

                const calRun = (seg.cal_runs && seg.cal_runs[currentRun]) ? seg.cal_runs[currentRun] : (seg.cal_runs ? seg.cal_runs["default"] : null);
                if (calRun) {
                    const tMainCal = calRun.threshold_main || calRun.stage1_amplitude_criterion || calRun.amplitude_threshold;
                    if (tMainCal !== undefined && tMainCal !== null) {
                        traces.push({
                            x: [timesCal[0], timesCal[timesCal.length - 1]], y: [tMainCal, tMainCal], mode: "lines",
                            name: "Calcium MAIN threshold", line: { color: "#fd7e14", dash: "dash", width: 1.2 },
                            yaxis: P.cal.ax, xaxis: "x", hoverinfo: "skip"
                        });
                    }
                    const tSceCal = calRun.threshold_sce || (calRun.params ? calRun.params.min_norm_height_sce : 0.08);
                    if (tSceCal !== undefined && tSceCal !== null) {
                        traces.push({
                            x: [timesCal[0], timesCal[timesCal.length - 1]], y: [tSceCal, tSceCal], mode: "lines",
                            name: "Calcium SECONDARY threshold", line: { color: "#6f42c1", dash: "dot", width: 1.2 },
                            yaxis: P.cal.ax, xaxis: "x", hoverinfo: "skip"
                        });
                    }
                    const calList = getCalAnn(currentSegIdx, currentRun);
                    const calQC = calRun.qc_status || "PASS";
                    const isCalQCFail = (calQC !== "PASS");

                    if (isCalQCFail) {
                        if (showRej) {
                            const calDiag = calList.filter(a => a.status === "ACCEPTED" || a.auto_status === "ACCEPTED");
                            if (calDiag.length > 0) {
                                traces.push({
                                    x: calDiag.map(a => a.time_s), y: calDiag.map(a => a.amp || 0.0),
                                    customdata: calDiag.map(a => "cal_" + a.index),
                                    text: calDiag.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>EXCLUDED (Calcium QC ${calQC}: ${calRun.qc_reason})`),
                                    hoverinfo: "text", mode: "markers", name: "Calcium diagnostic candidates",
                                    marker: { color: "#888888", size: 8, symbol: "circle-open" }, opacity: 0.65,
                                    yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                                });
                            }
                        }
                    } else {
                        const calAutoMain = calList.filter(a => a.status === "ACCEPTED" && a.origin !== "manual" && a.event_class === "MAIN");
                        const calAutoSec = calList.filter(a => a.status === "ACCEPTED" && a.origin !== "manual" && a.event_class === "SECONDARY");
                        const calManMain = calList.filter(a => a.status === "ACCEPTED" && a.origin === "manual" && a.event_class !== "SECONDARY");
                        const calManSec = calList.filter(a => a.status === "ACCEPTED" && a.origin === "manual" && a.event_class === "SECONDARY");
                        const calManRej = calList.filter(a => a.status === "REJECTED" && a.manually_rejected);

                        if (calAutoMain.length > 0) {
                            traces.push({
                                x: calAutoMain.map(a => a.time_s), y: calAutoMain.map(a => a.amp || 0.0),
                                customdata: calAutoMain.map(a => "cal_" + a.index),
                                text: calAutoMain.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>Calcium transient [MAIN]`),
                                hoverinfo: "text", mode: "markers", name: `Calcium MAIN (${calAutoMain.length})`,
                                marker: { color: "#28a745", size: 9, symbol: "circle" },
                                yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                            });
                        }
                        if (calAutoSec.length > 0) {
                            traces.push({
                                x: calAutoSec.map(a => a.time_s), y: calAutoSec.map(a => a.amp || 0.0),
                                customdata: calAutoSec.map(a => "cal_" + a.index),
                                text: calAutoSec.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>Calcium transient [SECONDARY]`),
                                hoverinfo: "text", mode: "markers", name: `Calcium SCE (${calAutoSec.length})`,
                                marker: { color: "#6f42c1", size: 9, symbol: "diamond" },
                                yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                            });
                        }
                        if (calManMain.length > 0) {
                            traces.push({
                                x: calManMain.map(a => a.time_s), y: calManMain.map(a => a.amp || 0.0),
                                customdata: calManMain.map(a => "cal_" + a.index),
                                text: calManMain.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>Manual calcium MAIN (click to delete)`),
                                hoverinfo: "text", mode: "markers", name: `Calcium manual MAIN (${calManMain.length})`,
                                marker: { color: "#007bff", size: 9, symbol: "circle", line: { color: "#004085", width: 1.5 } },
                                yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                            });
                        }
                        if (calManSec.length > 0) {
                            traces.push({
                                x: calManSec.map(a => a.time_s), y: calManSec.map(a => a.amp || 0.0),
                                customdata: calManSec.map(a => "cal_" + a.index),
                                text: calManSec.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>Manual calcium SCE (click to delete)`),
                                hoverinfo: "text", mode: "markers", name: `Calcium manual SCE (${calManSec.length})`,
                                marker: { color: "#007bff", size: 9, symbol: "diamond", line: { color: "#004085", width: 1.5 } },
                                yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                            });
                        }
                        if (calManRej.length > 0) {
                            traces.push({
                                x: calManRej.map(a => a.time_s), y: calManRej.map(a => a.amp || 0.0),
                                customdata: calManRej.map(a => "cal_" + a.index),
                                text: calManRej.map(a => `t=${a.time_s.toFixed(3)}s<br>ΔR/R₀=${a.amp !== null ? a.amp.toFixed(4) : "N/A"}<br>Rejected calcium transient (click to restore)`),
                                hoverinfo: "text", mode: "markers", name: `Calcium rejected (${calManRej.length})`,
                                marker: { color: "#dc3545", size: 9, symbol: "x", line: { color: "#a71d2a", width: 2 } },
                                yaxis: P.cal.ax, xaxis: "x", type: "scatter"
                            });
                        }
                    }
                }
            }

            // Marker traces on Panel 2 (Shortening) ONLY! Panel 1 has ZERO markers.
            function addContractionMarkers(filterFn, style, labelName, showLegend) {
                const subset = evs.filter(filterFn);
                if (subset.length === 0) return;

                const xs = [];
                const ys = [];
                const cds = [];
                const texts = [];

                subset.forEach(e => {
                    xs.push(e.time_s);
                    cds.push(e.index);
                    const isCorrupted = (e.pct_shortening === null || e.abs_shortening_um === null || !e.has_valid_amplitude);
                    const yPos = isCorrupted ? (thrVal || 0.0) : (showPct ? e.pct_shortening : e.abs_shortening_um);
                    ys.push(yPos);

                    const dLStr = (e.abs_shortening_um !== null && e.abs_shortening_um !== undefined) ? `${e.abs_shortening_um.toFixed(4)} µm` : "N/A (corrupted trough)";
                    const pctStr = (e.pct_shortening !== null && e.pct_shortening !== undefined) ? `${e.pct_shortening.toFixed(2)}%` : "N/A (corrupted trough)";
                    const natStr = e.event_nature ? ` [${e.event_nature}]` : "";
                    const asstStr = e.assistance_reason ? `<br>Assistance: ${e.assistance_reason}` : "";
                    const calStr = (e.calcium_match_status && e.calcium_match_status !== "UNAVAILABLE") ? `<br>Ca2+ match: ${e.calcium_match_status}` : "";
                    texts.push(`t=${e.time_s.toFixed(3)}s<br>ΔL=${dLStr} (${pctStr})<br>${e.status} [${e.event_class}]${natStr}${asstStr}${calStr}`);
                });

                traces.push({
                    x: xs, y: ys, customdata: cds, text: texts,
                    hoverinfo: "text", mode: "markers", name: labelName,
                    marker: style, yaxis: P.short.ax, xaxis: "x", type: "scatter",
                    showlegend: showLegend
                });
            }

            const classSymbolMap = {
                "MAIN": "circle",
                "SECONDARY": "diamond",
                "AMBIGUOUS": "triangle-up",
            };

            if (!isQCRej && !isManualRej) {
                // 1. Automatically accepted MAIN events (circle)
                addContractionMarkers(
                    e => e.status === "ACCEPTED" && e.origin !== "manual" && e.event_class === "MAIN",
                    { color: "#28a745", size: 10, symbol: "circle" },
                    "Accepted MAIN (circle)", true
                );

                // 2. Automatically accepted SECONDARY events (diamond)
                addContractionMarkers(
                    e => e.status === "ACCEPTED" && e.origin !== "manual" && e.event_class === "SECONDARY",
                    { color: "#28a745", size: 10, symbol: "diamond" },
                    "Accepted SECONDARY (diamond)", true
                );

                // 3. Automatically accepted AMBIGUOUS events (triangle)
                addContractionMarkers(
                    e => e.status === "ACCEPTED" && e.origin !== "manual" && e.event_class === "AMBIGUOUS",
                    { color: "#28a745", size: 10, symbol: "triangle-up" },
                    "Accepted AMBIGUOUS (triangle)", true
                );

                // 4. Manually added events (BLUE, shape matching class)
                ["MAIN", "SECONDARY", "AMBIGUOUS"].forEach(cls => {
                    addContractionMarkers(
                        e => e.status === "ACCEPTED" && e.origin === "manual" && e.event_class === cls,
                        { color: "#007bff", size: 11, symbol: classSymbolMap[cls], line: { color: "#004085", width: 1.5 } },
                        `Manual ${cls} (blue)`, true
                    );
                });

                // 5. Manually rejected automatic events (RED CROSS)
                addContractionMarkers(
                    e => e.status === "REJECTED" && e.manually_rejected,
                    { color: "#dc3545", size: 11, symbol: "x", line: { color: "#a71d2a", width: 2 } },
                    "Rejected automatic peak (click to restore)", true
                );

                // 6. Diagnostic excluded candidates
                if (showRej) {
                    addContractionMarkers(
                        e => e.status === "REJECTED" && !e.manually_rejected,
                        { color: "#888888", size: 8, symbol: "circle-open" },
                        "Excluded candidate (diagnostic)", true
                    );
                }
            } else {
                // QC Rejected or Manually Excluded segment: show diagnostic gray markers only
                addContractionMarkers(
                    e => (e.auto_status === "ACCEPTED" || e.status === "ACCEPTED"),
                    { color: "#888888", size: 8, symbol: "circle-open" },
                    "EXCLUDED candidates (segment rejected)", true
                );
            }

            // Highlight selected peak
            if (selectedPeakInfo && selectedPeakInfo.segIdx === currentSegIdx) {
                const fs = seg.fs;
                const w = Math.max(1, Math.round(0.04 * fs));
                const sel = evs.find(e => Math.abs(e.index - selectedPeakInfo.index) <= w);
                if (sel) {
                    const isCorrupted = (sel.pct_shortening === null || sel.abs_shortening_um === null || !sel.has_valid_amplitude);
                    const yPos = isCorrupted ? (thrVal || 0.0) : (showPct ? sel.pct_shortening : sel.abs_shortening_um);
                    traces.push({
                        x: [sel.time_s], y: [yPos],
                        mode: "markers", name: "SELECTED event",
                        marker: { color: "rgba(0,0,0,0)", size: 18, symbol: "circle", line: { color: "#ff9800", width: 3 } },
                        yaxis: P.short.ax, xaxis: "x", hoverinfo: "skip"
                    });
                }
            }

            // Artifact shaded intervals as Plotly layout shapes (ONLY if checked)
            const layoutShapes = [];
            if (showArt && seg.artifact_intervals && seg.artifact_intervals.length > 0) {
                seg.artifact_intervals.forEach(iv => {
                    layoutShapes.push({
                        type: "rect",
                        xref: "x", yref: "paper",
                        x0: iv[0], x1: iv[1],
                        y0: 0, y1: 1,
                        fillcolor: "rgba(220, 53, 69, 0.12)",
                        line: { color: "rgba(220, 53, 69, 0.35)", width: 1, dash: "dot" },
                    });
                });
            }

            const titleTxt = `${seg.sheet_name} — accepted contractions: ${counts.final} (MAIN: ${counts.main}, SEC: ${counts.sec}) — strictness: ${currentRun}`;

            const layout = {
                title: titleTxt,
                hovermode: "closest",
                height: Math.max(540, 240 * panels.length + 120),
                uirevision: currentSegIdx + "|" + currentRun + "|" + showPct + "|" + assistanceEnabled,
                xaxis: { title: "Time (s)", anchor: "y" + (panels.length === 1 ? "" : panels.length), domain: [0, 1] },
                legend: { orientation: "h", y: -0.12, x: 0.0 },
                shapes: layoutShapes,
                margin: { l: 60, r: 30, t: 50, b: 60 },
            };

            panels.forEach(q => {
                const o = { domain: q.domain, anchor: "x", title: q.title };
                if (q.key === "short") {
                    o.zeroline = true;
                }
                layout["yaxis" + (q.ax === "y" ? "" : q.ax.replace("y", ""))] = o;
            });

            Plotly.react("plot", traces, layout);

            // Click listener for peak snapping, toggling, and selection
            const plotDiv = document.getElementById("plot");
            plotDiv.removeAllListeners && plotDiv.removeAllListeners("plotly_click");
            plotDiv.on("plotly_click", function(data) {
                const pt = data.points[0];
                const name = pt.data.name || "";
                const xClick = pt.x;

                // 1. Ignore clicks on Panel 1 (raw SarcomereLength)
                if (pt.data.yaxis === P.raw.ax || name.includes("SarcomereLength")) {
                    return;
                }

                // 2. Calcium panel interaction (Panel 3)
                if (P.cal && (pt.data.yaxis === P.cal.ax || name.startsWith("Normalized calcium") || name.startsWith("RatioMetricCalcium") || name.startsWith("Calcium "))) {
                    if (!seg.cal_runs) return;
                    const fsCal = seg.fs_cal || seg.fs;
                    const t0Cal = (seg.t0_cal !== null && seg.t0_cal !== undefined) ? seg.t0_cal : seg.t0;
                    const wCal = Math.max(1, Math.round(0.04 * fsCal));
                    const calList = getCalAnn(currentSegIdx, currentRun);

                    // Clicked on a calcium marker
                    if (pt.customdata && typeof pt.customdata === "string" && pt.customdata.startsWith("cal_")) {
                        const clickedCalIdx = parseInt(pt.customdata.replace("cal_", ""));
                        const targetItem = calList.find(c => Math.abs(c.index - clickedCalIdx) <= wCal);
                        if (targetItem) {
                            if (targetItem.origin === "manual") {
                                if (userCalManualPeaks[currentSegIdx]) {
                                    delete userCalManualPeaks[currentSegIdx][targetItem.index];
                                }
                            } else if (targetItem.manually_rejected) {
                                if (userCalPeakRejections[currentSegIdx]) {
                                    userCalPeakRejections[currentSegIdx].delete(targetItem.index);
                                }
                            } else {
                                if (!userCalPeakRejections[currentSegIdx]) userCalPeakRejections[currentSegIdx] = new Set();
                                userCalPeakRejections[currentSegIdx].add(targetItem.index);
                            }
                            renderPlot();
                            return;
                        }
                    }

                    // Clicked on the calcium line trace -> snap to local max and add manual peak
                    if (name.includes("Normalized calcium") && baseSig.cal) {
                        const approxIdx = Math.max(0, Math.min(baseSig.cal.length - 1, Math.round((xClick - t0Cal) * fsCal)));
                        const i0 = Math.max(0, approxIdx - wCal);
                        const i1 = Math.min(baseSig.cal.length - 1, approxIdx + wCal);
                        let bestIdx = approxIdx;
                        let maxVal = -Infinity;
                        for (let j = i0; j <= i1; j++) {
                            if (baseSig.cal[j] > maxVal) {
                                maxVal = baseSig.cal[j];
                                bestIdx = j;
                            }
                        }
                        const bestTime = t0Cal + bestIdx / fsCal;
                        if (!userCalManualPeaks[currentSegIdx]) userCalManualPeaks[currentSegIdx] = {};
                        userCalManualPeaks[currentSegIdx][bestIdx] = {
                            index: bestIdx,
                            time_s: bestTime,
                            amp: maxVal,
                            raw: maxVal,
                            origin: "manual",
                        };
                        renderPlot();
                        return;
                    }
                    return;
                }

                // 3. Shortening panel interaction (Panel 2)
                const fs = seg.fs;
                const t0 = seg.t0;
                const w = Math.max(1, Math.round(0.04 * fs));

                // Clicked on an existing contraction marker
                if (pt.customdata !== undefined && pt.customdata !== null && typeof pt.customdata === "number") {
                    const clickedIndex = parseInt(pt.customdata);
                    selectedPeakInfo = { segIdx: currentSegIdx, index: clickedIndex };

                    // Toggle acceptance:
                    if (userPeakRejections[currentSegIdx] && userPeakRejections[currentSegIdx].has(clickedIndex)) {
                        userPeakRejections[currentSegIdx].delete(clickedIndex);
                    } else if (userManualPeaks[currentSegIdx] && userManualPeaks[currentSegIdx][clickedIndex]) {
                        delete userManualPeaks[currentSegIdx][clickedIndex];
                        selectedPeakInfo = null;
                    } else {
                        if (!userPeakRejections[currentSegIdx]) userPeakRejections[currentSegIdx] = new Set();
                        userPeakRejections[currentSegIdx].add(clickedIndex);
                    }

                    renderPlot();
                    return;
                }

                // Clicked on Shortening curve -> snap to local minimum of authentic SarcomereLength L(t)
                if (name.includes("Shortening") && baseSig.raw) {
                    const approxIdx = Math.max(0, Math.min(seg.length - 1, Math.round((xClick - t0) * fs)));
                    const i0 = Math.max(0, approxIdx - w);
                    const i1 = Math.min(seg.length - 1, approxIdx + w);
                    let bestIdx = approxIdx;
                    let minVal = Infinity;
                    for (let j = i0; j <= i1; j++) {
                        if (baseSig.raw[j] < minVal) {
                            minVal = baseSig.raw[j];
                            bestIdx = j;
                        }
                    }

                    const bestTime = t0 + bestIdx / fs;
                    const l0Val = baseSig.l0 ? baseSig.l0[bestIdx] : (minVal + 0.15);
                    const absShort = Math.max(0, l0Val - minVal);
                    const pctShort = (l0Val > 0) ? (100.0 * absShort / l0Val) : 0.0;

                    if (!userManualPeaks[currentSegIdx]) userManualPeaks[currentSegIdx] = {};
                    userManualPeaks[currentSegIdx][bestIdx] = {
                        index: bestIdx,
                        time_s: bestTime,
                        l0: l0Val,
                        l_min: minVal,
                        abs_short: absShort,
                        pct_short: pctShort,
                        event_class: "MAIN",
                    };
                    selectedPeakInfo = { segIdx: currentSegIdx, index: bestIdx };
                    renderPlot();
                    return;
                }
            });
        }

        // Setup user interface listeners
        function setupUI() {
            document.getElementById("source_name").innerText = source_workbook;

            // Strictness selector
            const runSel = document.getElementById("run_select");
            ["default", "permissive", "strict"].forEach(r => {
                const opt = document.createElement("option");
                opt.value = r;
                opt.innerText = r.charAt(0).toUpperCase() + r.slice(1);
                runSel.appendChild(opt);
            });
            runSel.value = currentRun;
            runSel.addEventListener("change", e => {
                currentRun = e.target.value;
                selectedPeakInfo = null;
                populateSegmentSelector();
                renderPlot();
            });

            // Calcium Assistance toggle (dual mode)
            const btnAsst = document.getElementById("btn_toggle_assistance");
            if (btnAsst) {
                btnAsst.addEventListener("click", () => {
                    assistanceEnabled = !assistanceEnabled;
                    btnAsst.innerText = assistanceEnabled ? "ON" : "OFF";
                    btnAsst.style.background = assistanceEnabled ? "#0d6efd" : "#6c757d";
                    selectedPeakInfo = null;
                    populateSegmentSelector();
                    renderPlot();
                });
            }

            // Navigation
            populateSegmentSelector();
            document.getElementById("segment_select").addEventListener("change", e => {
                currentSegIdx = parseInt(e.target.value);
                selectedPeakInfo = null;
                renderPlot();
            });
            document.getElementById("btn_prev").addEventListener("click", () => {
                if (currentSegIdx > 0) {
                    currentSegIdx--;
                    selectedPeakInfo = null;
                    renderPlot();
                }
            });
            document.getElementById("btn_next").addEventListener("click", () => {
                if (currentSegIdx < segments_data.length - 1) {
                    currentSegIdx++;
                    selectedPeakInfo = null;
                    renderPlot();
                }
            });

            // Keyboard navigation
            window.addEventListener("keydown", e => {
                if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
                if (e.key === "ArrowLeft" || e.key === "p" || e.key === "P") {
                    if (currentSegIdx > 0) {
                        currentSegIdx--;
                        selectedPeakInfo = null;
                        renderPlot();
                    }
                } else if (e.key === "ArrowRight" || e.key === "n" || e.key === "N") {
                    if (currentSegIdx < segments_data.length - 1) {
                        currentSegIdx++;
                        selectedPeakInfo = null;
                        renderPlot();
                    }
                }
            });

            // Checkbox listeners
            document.getElementById("chk_show_artifacts").addEventListener("change", renderPlot);
            document.getElementById("chk_rejected").addEventListener("change", renderPlot);
            document.getElementById("chk_pct_short").addEventListener("change", renderPlot);

            // Segment rejection checkbox
            const chkRej = document.getElementById("chk_segment_rejected");
            const selRejReason = document.getElementById("segment_rejection_reason");
            chkRej.addEventListener("change", e => {
                const rej = e.target.checked;
                selRejReason.style.display = rej ? "inline-block" : "none";
                if (rej) {
                    userSegmentRejection[currentSegIdx] = { rejected: true, reason: selRejReason.value };
                } else {
                    delete userSegmentRejection[currentSegIdx];
                }
                populateSegmentSelector();
                renderPlot();
            });
            const chkReconInit = document.getElementById("chk_reconstruction");
            if (chkReconInit) {
                chkReconInit.addEventListener("change", () => {
                    renderPlot();
                });
            }
            selRejReason.addEventListener("change", e => {
                if (userSegmentRejection[currentSegIdx]) {
                    userSegmentRejection[currentSegIdx].reason = e.target.value;
                    renderPlot();
                }
            });

            // Peak dropdown selection
            document.getElementById("peak_select").addEventListener("change", e => {
                const val = e.target.value;
                if (val === "none") {
                    selectedPeakInfo = null;
                } else {
                    selectedPeakInfo = { segIdx: currentSegIdx, index: parseInt(val) };
                }
                renderPlot();
            });

            // Class reclassification override
            document.getElementById("event_class_select").addEventListener("change", e => {
                if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) return;
                const newClass = e.target.value;
                const targetIdx = selectedPeakInfo.index;
                const seg = segments_data[currentSegIdx];
                const w = Math.max(1, Math.round(0.04 * seg.fs));

                if (!userClassOverrides[currentSegIdx]) userClassOverrides[currentSegIdx] = {};

                if (newClass === "restore") {
                    Object.keys(userClassOverrides[currentSegIdx]).forEach(k => {
                        if (Math.abs(parseInt(k) - targetIdx) <= w) {
                            delete userClassOverrides[currentSegIdx][k];
                        }
                    });
                } else {
                    userClassOverrides[currentSegIdx][targetIdx] = newClass;
                }
                renderPlot();
            });

            // Per-peak strictness override
            document.getElementById("peak_strictness_select").addEventListener("change", e => {
                if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) return;
                const preset = e.target.value;
                const targetIdx = selectedPeakInfo.index;
                const seg = segments_data[currentSegIdx];
                const w = Math.max(1, Math.round(0.04 * seg.fs));

                if (!userPerPeakStrictness[currentSegIdx]) userPerPeakStrictness[currentSegIdx] = {};

                if (preset === "inherit") {
                    Object.keys(userPerPeakStrictness[currentSegIdx]).forEach(k => {
                        if (Math.abs(parseInt(k) - targetIdx) <= w) {
                            delete userPerPeakStrictness[currentSegIdx][k];
                        }
                    });
                } else {
                    userPerPeakStrictness[currentSegIdx][targetIdx] = preset;
                }
                renderPlot();
            });

            // Delete manual peak button
            document.getElementById("btn_remove_manual").addEventListener("click", () => {
                if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) return;
                const targetIdx = selectedPeakInfo.index;
                const seg = segments_data[currentSegIdx];
                const w = Math.max(1, Math.round(0.04 * seg.fs));
                if (userManualPeaks[currentSegIdx]) {
                    Object.keys(userManualPeaks[currentSegIdx]).forEach(k => {
                        if (Math.abs(parseInt(k) - targetIdx) <= w) {
                            delete userManualPeaks[currentSegIdx][k];
                        }
                    });
                }
                selectedPeakInfo = null;
                renderPlot();
            });

            // Export annotations to CSV
            document.getElementById("btn_save").addEventListener("click", saveAnnotations);

            // Export overrides to JSON
            document.getElementById("btn_export_overrides").addEventListener("click", exportOverrides);

            // Import overrides trigger
            const btnImp = document.getElementById("btn_import_trigger");
            const fileImp = document.getElementById("file_import");
            btnImp.addEventListener("click", () => fileImp.click());
            fileImp.addEventListener("change", importOverrides);

            renderPlot();
        }

        // Export verified annotations as CSV
        function saveAnnotations() {
            const rows = [
                ["workbook", "segment_number", "peak_index", "time_s", "l0_um", "l_min_um",
                 "abs_shortening_um", "pct_shortening", "takeoff_l0_um", "local_abs_shortening_um",
                 "local_pct_shortening", "event_class", "event_nature", "recovery_status", "status",
                 "origin", "is_user_modified", "calcium_match_status", "calcium_mechanical_delay_ms",
                 "calcium_assistance_enabled"]
            ];

            segments_data.forEach((seg, idx) => {
                const isManRej = userSegmentRejection[idx] && userSegmentRejection[idx].rejected;
                const isQCRej = (seg.qc_status !== "PASS" && seg.qc_status !== "PASS_PARTIAL");
                if (isManRej || isQCRej) return;

                const evs = getActiveEvents(idx, currentRun);
                evs.filter(e => e.status === "ACCEPTED").forEach(e => {
                    const delayMs = (e.calcium_mechanical_delay_s !== null && e.calcium_mechanical_delay_s !== undefined) ? (e.calcium_mechanical_delay_s * 1000).toFixed(1) : "";
                    rows.push([
                        source_workbook,
                        seg.segment_number,
                        e.index,
                        e.time_s.toFixed(4),
                        e.l0_um !== null ? e.l0_um.toFixed(4) : "",
                        e.l_min_um !== null ? e.l_min_um.toFixed(4) : "",
                        e.abs_shortening_um !== null ? e.abs_shortening_um.toFixed(4) : "",
                        e.pct_shortening !== null ? e.pct_shortening.toFixed(2) : "",
                        e.takeoff_l0_um !== null ? e.takeoff_l0_um.toFixed(4) : "",
                        e.local_abs_shortening_um !== null ? e.local_abs_shortening_um.toFixed(4) : "",
                        e.local_pct_shortening !== null ? e.local_pct_shortening.toFixed(2) : "",
                        e.event_class,
                        e.event_nature || "MEASURED",
                        e.recovery_status || "DIRECT_MEASUREMENT",
                        e.status,
                        e.origin,
                        e.is_class_overridden ? "true" : "false",
                        e.calcium_match_status || "",
                        delayMs,
                        assistanceEnabled ? "true" : "false",
                    ]);
                });
            });

            const csvContent = "data:text/csv;charset=utf-8," + rows.map(r => r.join(",")).join("\n");
            const link = document.createElement("a");
            link.setAttribute("href", encodeURI(csvContent));
            link.setAttribute("download", `${source_workbook.replace(/\.xlsx$/i, "")}_sarcomere_annotations.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        // Export user overrides and rejections as JSON sidecar
        function exportOverrides() {
            const rejectionsExport = {};
            Object.entries(userSegmentRejection).forEach(([idx, val]) => {
                if (val && val.rejected) {
                    const sNum = segments_data[parseInt(idx)].segment_number;
                    rejectionsExport[String(sNum)] = { rejected: true, reason: val.reason };
                }
            });

            const manualPeaksExport = {};
            Object.entries(userManualPeaks).forEach(([idx, map]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                const peaks = Object.values(map);
                if (peaks.length > 0) {
                    manualPeaksExport[String(sNum)] = peaks;
                }
            });

            const rejectionsPeaksExport = {};
            Object.entries(userPeakRejections).forEach(([idx, set]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                if (set.size > 0) {
                    rejectionsPeaksExport[String(sNum)] = Array.from(set);
                }
            });

            const classOverridesExport = {};
            Object.entries(userClassOverrides).forEach(([idx, map]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                if (Object.keys(map).length > 0) {
                    classOverridesExport[String(sNum)] = map;
                }
            });

            const strictnessExport = {};
            Object.entries(userPerPeakStrictness).forEach(([idx, map]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                if (Object.keys(map).length > 0) {
                    strictnessExport[String(sNum)] = map;
                }
            });

            const calManualExport = {};
            Object.entries(userCalManualPeaks).forEach(([idx, map]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                const pks = Object.values(map);
                if (pks.length > 0) calManualExport[String(sNum)] = pks;
            });

            const calRejExport = {};
            Object.entries(userCalPeakRejections).forEach(([idx, set]) => {
                const sNum = segments_data[parseInt(idx)].segment_number;
                if (set.size > 0) calRejExport[String(sNum)] = Array.from(set);
            });

            const exportObj = {
                format_version: "2.1",
                workbook: source_workbook,
                workbook_sha256: source_sha256,
                analysis_mode: ANALYSIS_MODE,
                calcium_assistance_enabled: assistanceEnabled,
                segment_rejections: rejectionsExport,
                manual_peaks: manualPeaksExport,
                peak_rejections: rejectionsPeaksExport,
                class_overrides: classOverridesExport,
                per_peak_strictness: strictnessExport,
                calcium_manual_peaks: calManualExport,
                calcium_peak_rejections: calRejExport,
            };

            const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj, null, 2));
            const link = document.createElement("a");
            link.setAttribute("href", jsonStr);
            link.setAttribute("download", `${source_workbook.replace(/\.xlsx$/i, "")}_sarcomere_overrides.json`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        // Import overrides from JSON sidecar
        function importOverrides(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                try {
                    const data = JSON.parse(evt.target.result);

                    if (data.calcium_assistance_enabled !== undefined) {
                        assistanceEnabled = Boolean(data.calcium_assistance_enabled);
                        const btnAsst = document.getElementById("btn_toggle_assistance");
                        if (btnAsst) {
                            btnAsst.innerText = assistanceEnabled ? "ON" : "OFF";
                            btnAsst.style.background = assistanceEnabled ? "#0d6efd" : "#6c757d";
                        }
                    }

                    // Restore segment rejections
                    if (data.segment_rejections) {
                        Object.entries(data.segment_rejections).forEach(([sNum, val]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) userSegmentRejection[idx] = val;
                        });
                    }

                    // Restore manual peaks
                    if (data.manual_peaks) {
                        Object.entries(data.manual_peaks).forEach(([sNum, pList]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) {
                                userManualPeaks[idx] = {};
                                pList.forEach(p => { userManualPeaks[idx][p.index] = p; });
                            }
                        });
                    }

                    // Restore peak rejections
                    if (data.peak_rejections) {
                        Object.entries(data.peak_rejections).forEach(([sNum, rList]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) userPeakRejections[idx] = new Set(rList);
                        });
                    }

                    // Restore class overrides
                    if (data.class_overrides) {
                        Object.entries(data.class_overrides).forEach(([sNum, cMap]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) userClassOverrides[idx] = cMap;
                        });
                    }

                    // Restore per-peak strictness
                    if (data.per_peak_strictness) {
                        Object.entries(data.per_peak_strictness).forEach(([sNum, ppMap]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) userPerPeakStrictness[idx] = ppMap;
                        });
                    }

                    // Restore calcium manual & rejections
                    if (data.calcium_manual_peaks) {
                        Object.entries(data.calcium_manual_peaks).forEach(([sNum, pList]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) {
                                userCalManualPeaks[idx] = {};
                                pList.forEach(p => { userCalManualPeaks[idx][p.index] = p; });
                            }
                        });
                    }
                    if (data.calcium_peak_rejections) {
                        Object.entries(data.calcium_peak_rejections).forEach(([sNum, rList]) => {
                            const idx = segments_data.findIndex(s => String(s.segment_number) === String(sNum));
                            if (idx >= 0) userCalPeakRejections[idx] = new Set(rList);
                        });
                    }

                    alert(`Successfully imported overrides for ${source_workbook}`);
                    populateSegmentSelector();
                    renderPlot();
                } catch (err) {
                    alert(`Failed to import JSON: ${err}`);
                }
            };
            reader.readAsText(file);
        }

        // Initialize on DOM load
        document.addEventListener("DOMContentLoaded", setupUI);
    </script>
</body>
</html>
"""


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
