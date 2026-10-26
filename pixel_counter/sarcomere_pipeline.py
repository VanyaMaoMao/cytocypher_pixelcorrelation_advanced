"""
pixel_counter.sarcomere_pipeline
================================

End-to-end workbook channel discovery, authentic segment reconstruction,
and multi-preset analysis pipeline for cardiomyocyte SarcomereLength recordings
with independent Sarcomere-Only and Calcium-Assisted dual-channel modes.

Scientific & Architectural Principles:
1. Channel Separation: Only `SarcomereLength Segment N` sheets are used for
   sarcomere shortening analysis. `EdgeLength` sheets are discovered for
   metadata auditing only and are NEVER substituted for `SarcomereLength`.
2. Authentic Time-Series Reconstruction: Uses `load_cytocypher_excel` and
   `build_concatenated_signal(..., force_invert=False)` to reconstruct the
   exact experimental time axis and raw micrometer measurements without
   polarity inversion or synthetic interpolation.
3. Explicit Mode Selection:
   - `sarcomere`: Independent sarcomere shortening detection, baseline estimation,
     artifact masking, and MAIN/SECONDARY/AMBIGUOUS classification with zero calcium influence.
   - `sarcomere-calcium`: Foundational sarcomere analysis supplemented by independent
     calcium transient detection (`analyze_calcium_pipeline`) and monotonic 1-to-1
     electromechanical event matching (`match_sarcomere_and_calcium`).
4. Invariance & Zero Fabrication:
   - Running `sarcomere` mode produces 100% identical results whether calcium data is
     present or absent.
   - Calcium never invents mechanical contractions, never manufactures shortening
     amplitudes for corrupted troughs (`NaN` preserved), and never overrides
     sarcomere QC rejections.
5. QC Precedence: Segments rejected by QC (`qc_status == "REJECT"`) report
   0 reportable accepted events and `None` summary frequencies/amplitudes so
   corrupted recordings never silently pollute scientific summary tables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .calcium_detector import (
    CALCIUM_STRICTNESS_PRESETS,
    CalciumDetectorParams,
    analyze_calcium_pipeline,
)
from .config import BeatCounterConfig
from .io_utils import load_cytocypher_excel
from .preprocessing import build_concatenated_signal
from .sarcomere_calcium_matching import (
    SarcomereCalciumMatchingParams,
    match_sarcomere_and_calcium,
)
from .sarcomere_artifact_recovery import (
    reconcile_reconstructed_events,
    reconstruct_narrow_tracking_artifacts,
)
from .sarcomere_detector import (
    SARCOMERE_STRICTNESS_PRESETS,
    SarcomereDetectorParams,
    detect_sarcomere_contractions,
)
from .sarcomere_qc import detect_sarcomere_artifacts, evaluate_sarcomere_qc


def _extract_segment_number(sheet_name: str) -> Optional[int]:
    m = re.search(r"Segment\s+(\d+)", str(sheet_name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def discover_sarcomere_workbook_channels(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    """Inspect an IonOptix CytoCypher/CytoSolver Excel workbook and catalog channels."""
    p = Path(file_path)
    xl = pd.ExcelFile(p)
    sheet_names = list(xl.sheet_names)

    sarcomere_sheets: Dict[int, str] = {}
    calcium_sheets: Dict[int, str] = {}
    edgelength_sheets: Dict[int, str] = {}
    other_sheets: List[str] = []

    for s in sheet_names:
        s_clean = s.strip()
        seg_num = _extract_segment_number(s_clean)
        if s_clean.startswith("SarcomereLength") and seg_num is not None:
            sarcomere_sheets[seg_num] = s
        elif s_clean.startswith("RatioMetricCalcium") and seg_num is not None:
            calcium_sheets[seg_num] = s
        elif s_clean.startswith("EdgeLength") and seg_num is not None:
            edgelength_sheets[seg_num] = s
        else:
            other_sheets.append(s)

    return {
        "file_path": str(p),
        "file_name": p.name,
        "sarcomere_segments": dict(sorted(sarcomere_sheets.items())),
        "calcium_segments": dict(sorted(calcium_sheets.items())),
        "edgelength_segments_ignored": dict(sorted(edgelength_sheets.items())),
        "other_sheets": other_sheets,
        "n_sarcomere_segments": len(sarcomere_sheets),
        "n_calcium_segments": len(calcium_sheets),
        "n_edgelength_segments": len(edgelength_sheets),
    }


def load_sarcomere_segment(
    file_path: Union[str, Path],
    sheet_name: str,
    config: Optional[BeatCounterConfig] = None,
) -> Dict[str, Any]:
    """Load and reconstruct an authentic SarcomereLength segment without polarity inversion.

    Raises ValueError if `EdgeLength` is passed as `sheet_name`.
    """
    if "edgelength" in str(sheet_name).lower():
        raise ValueError(
            f"Refusing to load EdgeLength sheet '{sheet_name}' as SarcomereLength. "
            "EdgeLength must never be substituted for SarcomereLength."
        )

    if config is None:
        config = BeatCounterConfig()

    try:
        df_sheet, y_cols, fs, t0_sheet = load_cytocypher_excel(file_path, sheet_name=sheet_name)
        sig_raw, seg_df, concat_meta = build_concatenated_signal(
            df_sheet, y_cols, fs, config, force_invert=False
        )
        t0 = float(concat_meta.get("time_origin_s", t0_sheet))
        sig_arr = np.asarray(sig_raw, dtype=float)
        time_axis_s = t0 + np.arange(len(sig_arr), dtype=float) / float(fs)
        return {
            "load_ok": True,
            "load_error": None,
            "sheet_name": sheet_name,
            "segment_number": _extract_segment_number(sheet_name),
            "sig_raw": sig_arr,
            "time_axis_s": time_axis_s,
            "fs": float(fs),
            "t0": t0,
            "n_samples": len(sig_arr),
            "duration_s": float(len(sig_arr) / fs) if fs > 0 else 0.0,
            "units": "um",
            "df_sheet": df_sheet,
            "y_cols": y_cols,
            "seg_df": seg_df,
            "concat_meta": concat_meta,
        }
    except Exception as exc:
        return {
            "load_ok": False,
            "load_error": str(exc),
            "sheet_name": sheet_name,
            "segment_number": _extract_segment_number(sheet_name),
            "sig_raw": np.array([], dtype=float),
            "time_axis_s": np.array([], dtype=float),
            "fs": 0.0,
            "t0": 0.0,
            "n_samples": 0,
            "duration_s": 0.0,
            "units": "um",
            "df_sheet": None,
            "y_cols": [],
            "seg_df": None,
            "concat_meta": {},
        }


def analyze_sarcomere_pipeline(
    sig_raw: np.ndarray,
    fs: float,
    t0: float = 0.0,
    df_sheet: Optional[pd.DataFrame] = None,
    y_cols: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    params: Optional[SarcomereDetectorParams] = None,
    mode: str = "sarcomere",
    cal_sig_raw: Optional[np.ndarray] = None,
    cal_fs: Optional[float] = None,
    cal_t0: Optional[float] = None,
    cal_params: Optional[CalciumDetectorParams] = None,
    matching_params: Optional[SarcomereCalciumMatchingParams] = None,
    enable_assistance: bool = True,
) -> Dict[str, Any]:
    """Run complete sarcomere shortening detection and QC across all strictness presets.

    Supports two explicitly selectable modes:
    - `"sarcomere"`: Sarcomere shortening analysis without calcium data.
    - `"sarcomere-calcium"`: Sarcomere shortening analysis with calcium-assisted
      event validation and classification support.
    """
    mode_clean = str(mode).strip().lower()
    if mode_clean not in ("sarcomere", "sarcomere-calcium"):
        raise ValueError(f"Unknown analysis mode '{mode}'. Must be 'sarcomere' or 'sarcomere-calcium'.")

    if params is None:
        params = SARCOMERE_STRICTNESS_PRESETS["default"]

    x = np.asarray(sig_raw, dtype=float)
    n = len(x)
    total_dur_s = float(n / fs) if (n > 0 and fs > 0) else 0.0

    # Compute artifact diagnostics once on the raw signal
    artifact_info = detect_sarcomere_artifacts(x, fs, t0=t0)

    # Conservative second-pass artifact reconstruction on internal detection array
    # sig_raw (x) remains 100% untouched; reconstruction occurs on temporary sig_recon
    sig_recon, recon_meta = reconstruct_narrow_tracking_artifacts(x, fs, artifact_info)
    artifact_info["reconstructed_intervals"] = recon_meta.get("reconstructed_intervals", [])
    artifact_info["n_reconstructed_intervals"] = len(artifact_info["reconstructed_intervals"])

    # In sarcomere-calcium mode, run independent calcium pipeline across its presets
    cal_pipeline_res: Optional[Dict[str, Any]] = None
    if mode_clean == "sarcomere-calcium":
        if cal_sig_raw is not None and len(cal_sig_raw) > 0 and cal_fs is not None and cal_fs > 0:
            c_t0 = float(cal_t0 if cal_t0 is not None else 0.0)
            if cal_params is None:
                from pixel_counter.calcium_detector import CalciumDetectorParams
                cal_params = CalciumDetectorParams(is_adult_cardiomyocyte=True)
            cal_pipeline_res = analyze_calcium_pipeline(
                cal_sig_raw, cal_fs, t0=c_t0, params=cal_params
            )

    runs: Dict[str, Dict[str, Any]] = {}
    primary_events_df: Optional[pd.DataFrame] = None
    primary_l0: Optional[np.ndarray] = None
    primary_match_res: Optional[Dict[str, Any]] = None

    for preset_name in ["default", "permissive", "strict"]:
        p = SARCOMERE_STRICTNESS_PRESETS[preset_name]
        ev_df_recon, l0_curve, _ = detect_sarcomere_contractions(
            sig_recon, fs, t0=t0, params=p, artifact_info=artifact_info
        )
        det_metrics = getattr(ev_df_recon, "attrs", {}).get("detector_metrics", {})
        t_main_val = det_metrics.get("threshold_main_pct", 0.55)
        t_sce_val = det_metrics.get("threshold_sce_pct", 0.35)

        # Reconcile reconstructed candidates with raw measurement integrity:
        # Preserves NaN amplitude for corrupted troughs (zero fabrication)
        # Bounds MAIN/SCE classification strictly by authentic shoulder excursions
        ev_df = reconcile_reconstructed_events(
            ev_df_recon,
            x,
            fs,
            artifact_info,
            recon_meta,
            t_main_pct=t_main_val,
            t_sce_pct=t_sce_val,
            l0_curve=l0_curve,
        )
        ev_df.attrs["detector_metrics"] = det_metrics

        qc_st, qc_rs, qc_meta = evaluate_sarcomere_qc(
            x, fs, artifact_info, ev_df, df_sheet=df_sheet, y_cols=y_cols
        )

        # Baseline Sarcomere-Only counts
        acc_df_pre = ev_df[ev_df["status"] == "ACCEPTED"] if not ev_df.empty else pd.DataFrame()
        n_acc_sarc_only = len(acc_df_pre)
        n_main_sarc_only = int(np.sum(acc_df_pre["event_class"] == "MAIN")) if not acc_df_pre.empty else 0
        n_sec_sarc_only = int(np.sum(acc_df_pre["event_class"] == "SECONDARY")) if not acc_df_pre.empty else 0
        n_amb_sarc_only = int(np.sum(acc_df_pre["event_class"] == "AMBIGUOUS")) if not acc_df_pre.empty else 0
        n_corrupted = int(np.sum(ev_df["peak_corrupted"].astype(bool))) if not ev_df.empty else 0

        # Execute calcium matching if in sarcomere-calcium mode
        match_info: Dict[str, Any] = {}
        if mode_clean == "sarcomere-calcium":
            if cal_pipeline_res is not None:
                cal_preset_run = cal_pipeline_res["runs"][preset_name]
                cal_ev = cal_preset_run["events"]
                cal_qc_tuple = (cal_preset_run["qc_status"], cal_preset_run["qc_reason"], cal_preset_run["qc_meta"])
                c_fs_val = float(cal_fs if cal_fs else fs)
                c_t0_val = float(cal_t0 if cal_t0 is not None else t0)
                n_c_val = len(cal_sig_raw) if cal_sig_raw is not None else 0
            else:
                cal_preset_run = {"qc_status": "UNAVAILABLE", "qc_reason": "no_calcium_signal", "qc_meta": {}}
                cal_ev = pd.DataFrame()
                cal_qc_tuple = ("UNAVAILABLE", "no_calcium_signal", {})
                c_fs_val = float(fs)
                c_t0_val = float(t0)
                n_c_val = 0

            match_info = match_sarcomere_and_calcium(
                sarc_events_df=ev_df,
                cal_events_df=cal_ev,
                fs_sarc=fs,
                fs_cal=c_fs_val,
                t0_sarc=t0,
                t0_cal=c_t0_val,
                n_sarc=n,
                n_cal=n_c_val,
                sarc_qc=(qc_st, qc_rs, qc_meta),
                cal_qc=cal_qc_tuple,
                sig_sarc_raw=x,
                sig_cal_raw=cal_sig_raw,
                params=matching_params,
                enable_assistance=enable_assistance,
                threshold_main_pct=det_metrics.get("threshold_main_pct"),
                threshold_sce_pct=det_metrics.get("threshold_sce_pct"),
            )
            ev_df = match_info["sarc_events"]
        else:
            # Mode A: SARCOMERE ONLY
            # Schema columns populated with UNAVAILABLE/identity values so structure is identical
            if not ev_df.empty:
                ev_df["calcium_match_status"] = "UNAVAILABLE"
                ev_df["matched_calcium_time_s"] = np.nan
                ev_df["calcium_amplitude"] = np.nan
                ev_df["calcium_mechanical_delay_s"] = np.nan
                ev_df["calcium_onset_delay_s"] = np.nan
                ev_df["calcium_matching_confidence"] = "N/A"
                ev_df["sarcomere_only_classification"] = ev_df["event_class"]
                ev_df["assisted_classification"] = ev_df["event_class"]
                ev_df["assistance_reason"] = "sarcomere_only_mode"
            match_info = {
                "sarc_events": ev_df,
                "calcium_events": pd.DataFrame(),
                "matched_pairs": [],
                "unmatched_sarcomere_events": ev_df[ev_df["status"] == "ACCEPTED"].copy() if not ev_df.empty else pd.DataFrame(),
                "unmatched_calcium_events": pd.DataFrame(),
                "estimated_emd_delay_s": None,
                "emd_delay_status": "UNAVAILABLE",
                "n_matched_pairs": 0,
                "n_unmatched_sarc": n_acc_sarc_only,
                "n_unmatched_cal": 0,
                "n_classifications_changed": 0,
                "n_borderline_recovered": 0,
                "n_artifact_corrupted": n_corrupted,
                "combined_qc": {
                    "combined_qc_status": qc_st,
                    "combined_qc_reason": "sarcomere_only_mode",
                    "description": "Sarcomere-Only analysis mode executed without calcium channel.",
                },
            }

        # Evaluated accepted events (after assistance if enabled)
        acc_df = ev_df[ev_df["status"] == "ACCEPTED"] if not ev_df.empty else pd.DataFrame()
        n_acc_raw = len(acc_df)
        n_main_raw = int(np.sum(acc_df["event_class"] == "MAIN")) if not acc_df.empty else 0
        n_sec_raw = int(np.sum(acc_df["event_class"] == "SECONDARY")) if not acc_df.empty else 0
        n_amb_raw = int(np.sum(acc_df["event_class"] == "AMBIGUOUS")) if not acc_df.empty else 0

        analyzable_dur = float(qc_meta.get("analyzable_duration_s", total_dur_s))

        # QC Gating Precedence: strictly binary PASS or REJECT from sarcomere QC alone.
        # Zero calcium promotion: calcium never overrides mechanical QC.
        eff_qc_st = qc_st
        eff_qc_rs = qc_rs

        is_reportable = (eff_qc_st == "PASS")
        if not ev_df.empty:
            ev_df = ev_df.copy()
            ev_df["preliminary_status"] = ev_df["status"].copy()
            ev_df["preliminary_reason"] = ev_df["reason"].copy()
            if not is_reportable:
                acc_mask = (ev_df["status"] == "ACCEPTED")
                if np.any(acc_mask):
                    ev_df.loc[acc_mask, "reason"] = f"qc_rejected_{eff_qc_rs}"
                    ev_df.loc[acc_mask, "status"] = "REJECTED"
        if is_reportable and n_acc_raw > 0 and analyzable_dur > 0:
            reportable_acc = n_acc_raw
            reportable_main = n_main_raw
            reportable_sec = n_sec_raw
            reportable_amb = n_amb_raw
            freq_hz = float(reportable_acc / analyzable_dur)
            freq_bpm = float(freq_hz * 60.0)
            main_freq_hz = float(reportable_main / analyzable_dur)
            main_freq_bpm = float(main_freq_hz * 60.0)
            sec_freq_hz = float(reportable_sec / analyzable_dur)
            sec_freq_epm = float(sec_freq_hz * 60.0)
        elif is_reportable and n_acc_raw == 0:
            reportable_acc = 0
            reportable_main = 0
            reportable_sec = 0
            reportable_amb = 0
            freq_hz = 0.0
            freq_bpm = 0.0
            main_freq_hz = 0.0
            main_freq_bpm = 0.0
            sec_freq_hz = 0.0
            sec_freq_epm = 0.0
        else:
            reportable_acc = 0
            reportable_main = 0
            reportable_sec = 0
            reportable_amb = 0
            freq_hz = None
            freq_bpm = None
            main_freq_hz = None
            main_freq_bpm = None
            sec_freq_hz = None
            sec_freq_epm = None

        # Quantitative shortening summary metrics on reportable accepted events
        mean_l0_um = None
        mean_l_min_um = None
        mean_abs_short_um = None
        mean_pct_short = None
        median_pct_short = None
        mean_main_abs_short_um = None
        mean_main_pct_short = None
        mean_sec_local_abs_short_um = None
        mean_sec_local_pct_short = None
        mean_sec_abs_short_um = None
        mean_sec_pct_short = None

        if is_reportable and not acc_df.empty:
            valid_acc = acc_df.dropna(subset=["l0_um", "l_min_um", "abs_shortening_um", "pct_shortening"])
            if not valid_acc.empty:
                mean_l0_um = float(valid_acc["l0_um"].mean())
                mean_l_min_um = float(valid_acc["l_min_um"].mean())
                mean_abs_short_um = float(valid_acc["abs_shortening_um"].mean())
                mean_pct_short = float(valid_acc["pct_shortening"].mean())
                median_pct_short = float(valid_acc["pct_shortening"].median())

            main_acc = acc_df[acc_df["event_class"] == "MAIN"]
            sec_acc = acc_df[acc_df["event_class"] == "SECONDARY"]
            if not main_acc.empty:
                main_vals = main_acc["abs_shortening_um"].dropna()
                main_pct_vals = main_acc["pct_shortening"].dropna()
                mean_main_abs_short_um = float(main_vals.mean()) if len(main_vals) > 0 else None
                mean_main_pct_short = float(main_pct_vals.mean()) if len(main_pct_vals) > 0 else None

            if not sec_acc.empty:
                sec_loc_vals = sec_acc["local_abs_shortening_um"].dropna()
                sec_loc_pct = sec_acc["local_pct_shortening"].dropna()
                sec_abs_vals = sec_acc["abs_shortening_um"].dropna()
                sec_pct_vals = sec_acc["pct_shortening"].dropna()
                mean_sec_local_abs_short_um = float(sec_loc_vals.mean()) if len(sec_loc_vals) > 0 else None
                mean_sec_local_pct_short = float(sec_loc_pct.mean()) if len(sec_loc_pct) > 0 else None
                mean_sec_abs_short_um = float(sec_abs_vals.mean()) if len(sec_abs_vals) > 0 else None
                mean_sec_pct_short = float(sec_pct_vals.mean()) if len(sec_pct_vals) > 0 else None

        det_metrics = getattr(ev_df, "attrs", {}).get("detector_metrics", {})

        runs[preset_name] = {
            "events": ev_df,
            "qc_status": eff_qc_st,
            "qc_reason": eff_qc_rs,
            "qc_meta": qc_meta,
            "n_accepted": reportable_acc,
            "n_accepted_raw": n_acc_raw,
            "n_main": reportable_main,
            "n_secondary": reportable_sec,
            "n_ambiguous": reportable_amb,
            "n_corrupted_events": n_corrupted,
            "sarcomere_only_accepted": n_acc_sarc_only if is_reportable else 0,
            "sarcomere_only_main": n_main_sarc_only if is_reportable else 0,
            "sarcomere_only_secondary": n_sec_sarc_only if is_reportable else 0,
            "sarcomere_only_ambiguous": n_amb_sarc_only if is_reportable else 0,
            "assisted_accepted": reportable_acc,
            "assisted_main": reportable_main,
            "assisted_secondary": reportable_sec,
            "assisted_ambiguous": reportable_amb,
            "frequency_hz": freq_hz,
            "frequency_bpm": freq_bpm,
            "main_frequency_hz": main_freq_hz,
            "main_frequency_bpm": main_freq_bpm,
            "secondary_frequency_hz": sec_freq_hz,
            "secondary_frequency_epm": sec_freq_epm,
            "mean_l0_um": mean_l0_um,
            "mean_l_min_um": mean_l_min_um,
            "mean_abs_shortening_um": mean_abs_short_um,
            "mean_pct_shortening": mean_pct_short,
            "median_pct_shortening": median_pct_short,
            "mean_main_abs_shortening_um": mean_main_abs_short_um,
            "mean_main_pct_shortening": mean_main_pct_short,
            "mean_secondary_local_abs_shortening_um": mean_sec_local_abs_short_um,
            "mean_secondary_local_pct_shortening": mean_sec_local_pct_short,
            "mean_secondary_abs_shortening_um": mean_sec_abs_short_um,
            "mean_secondary_pct_shortening": mean_sec_pct_short,
            "detector_metrics": det_metrics,
            "amplitude_threshold": float(det_metrics.get("threshold_main_pct", det_metrics.get("amplitude_threshold_pct", p.min_pct_shortening if p else 0.55))),
            "threshold_main_pct": float(det_metrics.get("threshold_main_pct", det_metrics.get("amplitude_threshold_pct", p.min_pct_shortening if p else 0.55))),
            "threshold_sce_pct": float(det_metrics.get("threshold_sce_pct", getattr(p, "min_pct_shortening_sce", 0.35))),
            "reference_amplitude_pct": float(det_metrics.get("reference_amplitude_pct", 0.0)) if pd.notna(det_metrics.get("reference_amplitude_pct")) else None,
            "noise_sigma_pct": float(det_metrics.get("noise_sigma_pct", 0.0)),
            "threshold_ordering_status": str(det_metrics.get("threshold_ordering_status", "distinct")),
            "threshold_main_components": det_metrics.get("threshold_main_components", {}),
            "threshold_sce_components": det_metrics.get("threshold_sce_components", {}),
            "strictness_preset": str(getattr(p, "preset_name", preset_name)),
            "noise_sigma": float(artifact_info.get("sigma_hf", 0.01)),
            "params": p,
            "matching": match_info,
            "n_directly_measured": match_info.get("n_directly_measured", reportable_acc),
            "n_calcium_assisted": match_info.get("n_calcium_assisted", 0),
            "n_valid_amplitude": match_info.get("n_valid_amplitude", reportable_acc),
            "n_lacking_amplitude": match_info.get("n_lacking_amplitude", 0),
            "n_unresolved": match_info.get("n_unresolved", 0),
            "sarc_only_events": match_info.get("sarc_only_events"),
        }

        if preset_name == params.preset_name or primary_events_df is None:
            primary_events_df = ev_df
            primary_l0 = l0_curve
            primary_match_res = match_info

    default_run = runs[params.preset_name]

    ibi_mean_s = None
    ibi_cv = None
    if default_run["qc_status"] == "PASS" and default_run["n_accepted"] > 1:
        acc_times = (
            primary_events_df[primary_events_df["status"] == "ACCEPTED"]["time_s"].to_numpy(dtype=float)
            if primary_events_df is not None
            else np.array([])
        )
        if len(acc_times) > 1:
            ibis = np.diff(acc_times)
            ibi_mean_s = float(np.mean(ibis))
            if ibi_mean_s > 0:
                ibi_cv = float(np.std(ibis) / ibi_mean_s)

    # Determine calcium QC status to report
    cal_qc_reported = "UNAVAILABLE"
    cal_reason_reported = "sarcomere_only_mode" if mode_clean == "sarcomere" else "no_calcium_data"
    if cal_pipeline_res is not None:
        cal_qc_reported = str(cal_pipeline_res["qc_status"])
        cal_reason_reported = str(cal_pipeline_res["qc_reason"])

    combined_qc_dict = (
        primary_match_res["combined_qc"]
        if primary_match_res is not None and "combined_qc" in primary_match_res
        else {
            "combined_qc_status": default_run["qc_status"],
            "combined_qc_reason": "sarcomere_only_mode",
            "description": "Sarcomere-Only analysis mode executed.",
        }
    )

    return {
        "mode": mode_clean,
        "events": primary_events_df,
        "l0_baseline_um": primary_l0,
        "sig_raw": x,
        "fs": float(fs),
        "t0": float(t0),
        "duration_s": total_dur_s,
        "analyzable_duration_s": float(default_run["qc_meta"].get("analyzable_duration_s", total_dur_s)),
        "artifact_info": artifact_info,
        "artifact_mask": artifact_info["artifact_mask"],
        "reconstruction": recon_meta,
        "reconstructed_intervals": recon_meta.get("reconstructed_intervals", []),
        "n_reconstructed_intervals": recon_meta.get("n_reconstructed_intervals", 0),
        "reconstructed_fraction": recon_meta.get("reconstructed_fraction", 0.0),
        "noise_sigma_um": float(artifact_info["sigma_hf"]),
        "qc_status": default_run["qc_status"],
        "qc_reason": default_run["qc_reason"],
        "qc_meta": default_run["qc_meta"],
        "calcium_qc_status": cal_qc_reported,
        "calcium_qc_reason": cal_reason_reported,
        "combined_qc": combined_qc_dict,
        "n_accepted": default_run["n_accepted"],
        "n_accepted_raw": default_run["n_accepted_raw"],
        "n_main": default_run["n_main"],
        "n_secondary": default_run["n_secondary"],
        "n_ambiguous": default_run["n_ambiguous"],
        "n_corrupted_events": default_run["n_corrupted_events"],
        "sarcomere_only_accepted": default_run["sarcomere_only_accepted"],
        "sarcomere_only_main": default_run["sarcomere_only_main"],
        "sarcomere_only_secondary": default_run["sarcomere_only_secondary"],
        "sarcomere_only_ambiguous": default_run["sarcomere_only_ambiguous"],
        "assisted_accepted": default_run["assisted_accepted"],
        "assisted_main": default_run["assisted_main"],
        "assisted_secondary": default_run["assisted_secondary"],
        "assisted_ambiguous": default_run["assisted_ambiguous"],
        "calcium_events": primary_match_res.get("calcium_events") if primary_match_res else None,
        "matched_pairs": primary_match_res.get("matched_pairs", []) if primary_match_res else [],
        "unmatched_sarcomere_events": primary_match_res.get("unmatched_sarcomere_events") if primary_match_res else None,
        "unmatched_calcium_events": primary_match_res.get("unmatched_calcium_events") if primary_match_res else None,
        "estimated_emd_delay_s": primary_match_res.get("estimated_emd_delay_s") if primary_match_res else None,
        "emd_delay_status": primary_match_res.get("emd_delay_status", "UNAVAILABLE") if primary_match_res else "UNAVAILABLE",
        "n_matched_pairs": primary_match_res.get("n_matched_pairs", 0) if primary_match_res else 0,
        "n_unmatched_sarc": primary_match_res.get("n_unmatched_sarc", 0) if primary_match_res else 0,
        "n_unmatched_cal": primary_match_res.get("n_unmatched_cal", 0) if primary_match_res else 0,
        "n_classifications_changed": primary_match_res.get("n_classifications_changed", 0) if primary_match_res else 0,
        "n_borderline_recovered": primary_match_res.get("n_borderline_recovered", 0) if primary_match_res else 0,
        "frequency_hz": default_run["frequency_hz"],
        "frequency_bpm": default_run["frequency_bpm"],
        "main_frequency_hz": default_run["main_frequency_hz"],
        "main_frequency_bpm": default_run["main_frequency_bpm"],
        "secondary_frequency_hz": default_run["secondary_frequency_hz"],
        "secondary_frequency_epm": default_run["secondary_frequency_epm"],
        "mean_l0_um": default_run["mean_l0_um"],
        "mean_l_min_um": default_run["mean_l_min_um"],
        "mean_abs_shortening_um": default_run["mean_abs_shortening_um"],
        "mean_pct_shortening": default_run["mean_pct_shortening"],
        "median_pct_shortening": default_run["median_pct_shortening"],
        "mean_main_abs_shortening_um": default_run["mean_main_abs_shortening_um"],
        "mean_main_pct_shortening": default_run["mean_main_pct_shortening"],
        "mean_secondary_local_abs_shortening_um": default_run["mean_secondary_local_abs_shortening_um"],
        "mean_secondary_local_pct_shortening": default_run["mean_secondary_local_pct_shortening"],
        "mean_secondary_abs_shortening_um": default_run["mean_secondary_abs_shortening_um"],
        "mean_secondary_pct_shortening": default_run["mean_secondary_pct_shortening"],
        "amplitude_threshold": default_run["amplitude_threshold"],
        "threshold_main_pct": default_run["threshold_main_pct"],
        "threshold_sce_pct": default_run["threshold_sce_pct"],
        "reference_amplitude_pct": default_run["reference_amplitude_pct"],
        "noise_sigma_pct": default_run["noise_sigma_pct"],
        "threshold_ordering_status": default_run["threshold_ordering_status"],
        "threshold_main_components": default_run["threshold_main_components"],
        "threshold_sce_components": default_run["threshold_sce_components"],
        "ibi_mean_s": ibi_mean_s,
        "ibi_cv": ibi_cv,
        "preset_runs": runs,
        "cal_pipeline_res": cal_pipeline_res,
        "meta": meta or {},
    }


def analyze_sarcomere_segment_from_excel(
    file_path: Union[str, Path],
    sheet_name: str,
    mode: str = "sarcomere",
    cal_sheet_name: Optional[str] = None,
    config: Optional[BeatCounterConfig] = None,
    params: Optional[SarcomereDetectorParams] = None,
    cal_params: Optional[CalciumDetectorParams] = None,
    matching_params: Optional[SarcomereCalciumMatchingParams] = None,
    enable_assistance: bool = True,
) -> Dict[str, Any]:
    """Load a SarcomereLength sheet from Excel and run the full sarcomere pipeline.

    Parameters
    ----------
    file_path : str or Path
        Path to the experimental Excel workbook.
    sheet_name : str
        Name of the SarcomereLength sheet (e.g. 'SarcomereLength Segment 33').
    mode : str
        'sarcomere' (SARCOMERE ONLY) or 'sarcomere-calcium' (SARCOMERE + CALCIUM).
    cal_sheet_name : Optional[str]
        Explicit name of the RatioMetricCalcium sheet. If None and mode='sarcomere-calcium',
        automatically discovers 'RatioMetricCalcium Segment N' matching the segment number.
    """
    if config is None:
        config = BeatCounterConfig()

    loaded = load_sarcomere_segment(file_path, sheet_name, config=config)
    if not loaded["load_ok"]:
        res = analyze_sarcomere_pipeline(
            np.array([], dtype=float),
            fs=0.0,
            t0=0.0,
            meta={"sheet_name": sheet_name, "load_error": loaded["load_error"]},
            params=params,
            mode=mode,
        )
        res["load_ok"] = False
        res["load_error"] = loaded["load_error"]
        res["sheet_name"] = sheet_name
        res["segment_number"] = loaded["segment_number"]
        return res

    cal_sig_raw: Optional[np.ndarray] = None
    cal_fs: Optional[float] = None
    cal_t0: Optional[float] = None
    resolved_cal_sheet: Optional[str] = None

    mode_clean = str(mode).strip().lower()
    if mode_clean == "sarcomere-calcium":
        if cal_sheet_name is None:
            seg_num = loaded["segment_number"]
            if seg_num is not None:
                p = Path(file_path)
                xl = pd.ExcelFile(p)
                target_prefix = f"RatioMetricCalcium Segment {seg_num}".lower()
                for s in xl.sheet_names:
                    if s.strip().lower() == target_prefix:
                        resolved_cal_sheet = s
                        break
        else:
            resolved_cal_sheet = cal_sheet_name

        if resolved_cal_sheet is not None:
            try:
                df_c, y_c, fs_c, t0_c = load_cytocypher_excel(file_path, sheet_name=resolved_cal_sheet)
                sig_c, _, meta_c = build_concatenated_signal(
                    df_c, y_c, fs_c, config, force_invert=False
                )
                cal_sig_raw = np.asarray(sig_c, dtype=float)
                cal_fs = float(fs_c)
                cal_t0 = float(meta_c.get("time_origin_s", t0_c))
            except Exception:
                cal_sig_raw = None
                cal_fs = None
                cal_t0 = None

    res = analyze_sarcomere_pipeline(
        loaded["sig_raw"],
        loaded["fs"],
        t0=loaded["t0"],
        df_sheet=loaded["df_sheet"],
        y_cols=loaded["y_cols"],
        meta=loaded["concat_meta"],
        params=params,
        mode=mode_clean,
        cal_sig_raw=cal_sig_raw,
        cal_fs=cal_fs,
        cal_t0=cal_t0,
        cal_params=cal_params,
        matching_params=matching_params,
        enable_assistance=enable_assistance,
    )
    res["load_ok"] = True
    res["load_error"] = None
    res["sheet_name"] = sheet_name
    res["cal_sheet_name"] = resolved_cal_sheet
    res["segment_number"] = loaded["segment_number"]
    res["time_axis_s"] = loaded["time_axis_s"]
    return res
