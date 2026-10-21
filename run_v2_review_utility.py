"""Experimental v2 review utility.

Generates a single self-contained HTML file for reviewing the v2 contraction detector.

Data flow (single source of truth)
----------------------------------
Excel -> raw stitched PixelCorrelation (NEVER pre-inverted) -> analyze_v2_pipeline
       -> {raw polarity, baseline B_raw, amplitude A = polarity*(S_raw-B_raw), events}
       -> HTML.  The plot, the embedded event table, the review state and the exported CSV all
       use the same arrays and the same absolute timestamps (t0 + index/fs).

Strictness parameters are chosen when the HTML is generated (see ``--help``); the HTML embeds the
real detector output for the permissive / default / strict presets (plus a "custom" run if the
command-line values differ from the defaults) and lets the reviewer switch between them.
"""
import argparse
import base64
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pixel_counter.analysis import analyze_v2_pipeline
from pixel_counter.config import BeatCounterConfig
from pixel_counter.io_utils import extract_sample_id_from_segment_sheet, load_cytocypher_excel
from pixel_counter.polarity_overrides import (
    compute_file_sha256,
    default_overrides_path,
    load_polarity_overrides,
    load_segment_rejections,
    parse_polarity_override_str,
    save_polarity_overrides,
)
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.v2_detector import STRICTNESS_PRESETS, DetectorParams
from pixel_counter.v2_signal import determine_raw_polarity


def b64_encode_array(arr) -> str:
    """Encode numpy array to base64 float32 for embedding in HTML."""
    return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode("utf-8")


def _num(x):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _events_payload(v2_res):
    ev = v2_res["events"]
    diag_ev = v2_res.get("diagnostic_events", ev)
    return {
        "index": [int(i) for i in ev["index"]],
        "time_s": [float(t) for t in ev["time_abs_s"]],
        "amplitude": [_num(a) for a in ev["amplitude"]],
        "raw_value": [_num(a) for a in ev["raw_value"]] if "raw_value" in ev else [None] * len(ev),
        "status": [str(s) for s in ev["status"]],
        "reason": [str(s) for s in ev["reason"]],
        "diagnostic_status": [str(s) for s in diag_ev["status"]] if len(diag_ev) == len(ev) else [str(s) for s in ev["status"]],
        "diagnostic_reason": [str(s) for s in diag_ev["reason"]] if len(diag_ev) == len(ev) else [str(s) for s in ev["reason"]],
    }


def _run_payload(v2_res):
    md = v2_res["metadata"]
    return {
        "params": md.get("detector_params", {}),
        "separation_strictness": _num(md.get("separation_strictness")),
        "amplitude_strictness": _num(md.get("amplitude_strictness")),
        "qc_status": md.get("qc_status", "UNKNOWN"),
        "qc_reason": md.get("qc_reason", ""),
        "qc_layer1_status": md.get("qc_layer1_status", md.get("qc_status", "UNKNOWN")),
        "qc_layer1_reason": md.get("qc_layer1_reason", ""),
        "qc_layer2_status": md.get("qc_layer2_status", "PASS"),
        "qc_layer2_reason": md.get("qc_layer2_reason", ""),
        "signal_quality_category": md.get("signal_quality_category", ""),
        "analyzable_duration_s": _num(md.get("analyzable_duration_s")),
        "analyzable_fraction": _num(md.get("analyzable_fraction")),
        "excluded_intervals_s": md.get("excluded_intervals_s", []),
        "contraction_frequency_hz": _num(md.get("contraction_frequency_hz")),
        "reference_amplitude": _num(md.get("reference_amplitude")),
        "amplitude_threshold": _num(md.get("amplitude_threshold")),
        "noise_sigma": _num(md.get("noise_sigma")),
        "events": _events_payload(v2_res),
    }


def process_workbook(
    file_path: str,
    params_by_name=None,
    config=None,
    segments=None,
    polarity_overrides: Optional[Dict[int, int]] = None,
    inspect_only: bool = False,
):
    """Run the v2 pipeline on every PixelCorrelation segment of ``file_path``.

    Implements the two-stage polarity workflow:
      Stage 1: Determine polarity for every segment, verify evidence, log summary table.
      Stage 2: Execute full contraction analysis with effective polarity and precalculate
               both orientations (+1 and -1) for interactive offline review.

    params_by_name     : dict name -> DetectorParams (default: the strictness presets)
    config             : BeatCounterConfig
    segments           : optional iterable of segment numbers to restrict the run
    polarity_overrides : optional dict segment_number -> (+1 or -1) manual orientation
    inspect_only       : if True, runs Stage 1 and returns without generating full output
    """
    wb_path = Path(file_path)
    if not wb_path.exists():
        raise FileNotFoundError(f"Workbook not found: {file_path}")

    params_by_name = params_by_name or dict(STRICTNESS_PRESETS)
    config = config or BeatCounterConfig()
    polarity_overrides = dict(polarity_overrides or {})
    print(f"Loading {file_path}...")
    xl = pd.ExcelFile(wb_path)

    corr_sheets = [s for s in xl.sheet_names if "PixelCorrelation" in s and "Segment" in s]
    if not corr_sheets:
        raise ValueError(f"No PixelCorrelation segment sheets found in workbook: {wb_path.name}")

    def get_seg_num(s):
        try:
            return int(s.split("Segment")[-1].strip())
        except ValueError:
            return 9999

    corr_sheets.sort(key=get_seg_num)
    if segments:
        wanted = set(int(s) for s in segments)
        corr_sheets = [s for s in corr_sheets if get_seg_num(s) in wanted]

    use_ca = bool(getattr(config, "v2_use_calcium_polarity", False))
    mode_str = "Calcium-Assisted (ECC Timing Enabled)" if use_ca else "PixelCorrelation-Only (Default)"
    print(f"\n{'='*84}")
    print(f"STAGE 1 — Physiological Polarity Determination & Verification: {Path(file_path).name}")
    print(f"Mode:    {mode_str}")
    print(f"{'='*84}")
    print(f"{'Segment':<9} {'Auto Pol':<10} {'Status':<14} {'Method / Reason':<38} {'Override':<10} {'Effective':<10}")
    print(f"{'-'*84}")

    loaded_raw = []
    n_auto_ok = 0
    n_auto_undet = 0
    n_overrides = 0

    for i, corr_sheet in enumerate(corr_sheets):
        seg_num = get_seg_num(corr_sheet)
        df_corr, y_cols_corr, fs, _ = load_cytocypher_excel(file_path, sheet_name=corr_sheet)
        sample_id = extract_sample_id_from_segment_sheet(file_path, corr_sheet)
        sig_raw, _seg_meta, meta = build_concatenated_signal(df_corr, y_cols_corr, fs, config, force_invert=False)
        t0 = float(meta.get("time_origin_s", 0.0))

        def load_aux_channel(kind):
            sheet = corr_sheet.replace("PixelCorrelation", kind)
            if sheet not in xl.sheet_names:
                return None
            try:
                df_o, y_o, fs_o, _ = load_cytocypher_excel(file_path, sheet_name=sheet)
                sig_o, _, meta_o = build_concatenated_signal(df_o, y_o, fs_o, config, force_invert=False)
                t0_o = float(meta_o.get("time_origin_s", t0))
                return {
                    "signal": sig_o,
                    "fs": float(fs_o),
                    "t0": float(t0_o),
                    "length": int(len(sig_o)),
                }
            except Exception as e:
                return None

        aux_cal = load_aux_channel("RatioMetricCalcium")
        aux_int = load_aux_channel("PixelIntensity")
        sig_cal = aux_cal["signal"] if aux_cal is not None else None
        sig_int = aux_int["signal"] if aux_int is not None else None
        meta["sig_cal"] = sig_cal
        meta["sig_int"] = sig_int

        cal_for_polarity = sig_cal if use_ca else None
        fs_cal_pol = aux_cal["fs"] if aux_cal is not None else fs
        t0_cal_pol = aux_cal["t0"] if aux_cal is not None else t0
        auto_pol = determine_raw_polarity(sig_raw, fs, sig_cal=cal_for_polarity, fs_cal=fs_cal_pol, t0_raw=t0, t0_cal=t0_cal_pol)
        ov = polarity_overrides.get(seg_num)
        ov_str = f"{ov:+d}" if ov in (-1, 1) else "AUTO"
        eff = ov if ov in (-1, 1) else auto_pol["polarity"]
        eff_str = f"{eff:+d}" if eff in (-1, 1) else "UNDETERMINED"

        if auto_pol["status"] == "OK":
            n_auto_ok += 1
        else:
            n_auto_undet += 1
        if ov in (-1, 1):
            n_overrides += 1

        meth_reason = f"{auto_pol['method']}: {auto_pol['reason']}"[:36]
        print(f"Seg {seg_num:<5d} {auto_pol['polarity']:+2d}         {auto_pol['status']:<14} {meth_reason:<38} {ov_str:<10} {eff_str:<10}")

        loaded_raw.append({
            "seg_num": seg_num,
            "corr_sheet": corr_sheet,
            "sample_id": sample_id,
            "sig_raw": sig_raw,
            "fs": fs,
            "meta": meta,
            "t0": t0,
            "auto_pol": auto_pol,
            "ov": ov,
            "eff": eff,
            "aux_cal": aux_cal,
            "aux_int": aux_int,
        })

    print(f"{'-'*84}")
    print(f"Stage 1 Summary: Total segments={len(corr_sheets)}, Auto Determined={n_auto_ok}, "
          f"Auto Undetermined={n_auto_undet}, Active Overrides={n_overrides}")
    unresolved = [item["seg_num"] for item in loaded_raw if item["eff"] not in (-1, 1)]
    if unresolved:
        print(f"[!] UNRESOLVED POLARITY (INPUT REQUIRED) on Segments: {unresolved}")
        print("    Use --override-polarity 'SEG=-1' or --overrides <file.json> to supply orientation.")
    else:
        print("[OK] All processed segments have an effective physiological orientation.")
    print(f"{'='*84}\n")

    if inspect_only:
        return [], Path(file_path).name

    print(f"STAGE 2 — Full Contraction Detection & Amplitude Analysis...")
    segments_data = []
    for item in loaded_raw:
        seg_num = item["seg_num"]
        corr_sheet = item["corr_sheet"]
        sample_id = item["sample_id"]
        sig_raw = item["sig_raw"]
        fs = item["fs"]
        meta = item["meta"]
        t0 = item["t0"]
        auto_pol = item["auto_pol"]
        ov = item["ov"]
        eff = item["eff"]

        runs = {}
        r_runs_eff = {}
        for name, p in params_by_name.items():
            res = analyze_v2_pipeline(sig_raw, fs, config, t0=t0, meta=meta, params=p,
                                      polarity_override=eff if eff in (-1, 1) else None)
            runs[name] = _run_payload(res)
            r_runs_eff[name] = res
        pref_eff_name = "custom" if "custom" in r_runs_eff else ("default" if "default" in r_runs_eff else list(r_runs_eff.keys())[0])
        first = r_runs_eff[pref_eff_name]
        wf = first["waveform"]
        pol = first["polarity_result"]
        has_amp = bool(np.isfinite(wf["amplitude"]).any())

        # Precompute runs for both orientations (+1 and -1) for interactive offline review
        orientations = {}
        for fixed_pol, fixed_key in [(1, "1"), (-1, "-1")]:
            fixed_runs = {}
            r_runs_raw = {}
            for name, p in params_by_name.items():
                r_fix = analyze_v2_pipeline(sig_raw, fs, config, t0=t0, meta=meta, params=p,
                                            polarity_override=fixed_pol)
                fixed_runs[name] = _run_payload(r_fix)
                r_runs_raw[name] = r_fix
            pref_name = "custom" if "custom" in r_runs_raw else ("default" if "default" in r_runs_raw else list(r_runs_raw.keys())[0])
            first_fixed = r_runs_raw[pref_name]
            wf_fix = first_fixed["waveform"]
            orientations[fixed_key] = {
                "amplitude_b64": b64_encode_array(wf_fix["amplitude"]),
                "baseline_raw_b64": b64_encode_array(wf_fix["baseline_raw"]),
                "runs": fixed_runs,
                "qc_status": first_fixed["metadata"].get("qc_status", "PASS"),
                "qc_reason": first_fixed["metadata"].get("qc_reason", ""),
                "n_accepted": int(len(first_fixed["candidates"]["time_s"])),
            }

        aux_int = item.get("aux_int")
        aux_cal = item.get("aux_cal")
        sig_int = aux_int["signal"] if aux_int else None
        sig_cal = aux_cal["signal"] if aux_cal else None

        segments_data.append({
            "sheet_name": corr_sheet,
            "segment_number": seg_num,
            "sample_id": str(sample_id) if pd.notna(sample_id) else "",
            "fs": float(fs),
            "t0": t0,
            "length": int(len(sig_raw)),
            "qc_status": first["metadata"].get("qc_status", "UNKNOWN"),
            "qc_reason": first["metadata"].get("qc_reason", ""),
            "raw_polarity": int(pol.get("best_polarity", 0)),
            "polarity_source": str(pol.get("polarity_source", "automatic")),
            "polarity_method": str(pol.get("method", "")),
            "polarity_reason": str(pol.get("reason", "")),
            "auto_polarity": int(auto_pol.get("polarity", 0)),
            "auto_method": str(auto_pol.get("method", "")),
            "auto_reason": str(auto_pol.get("reason", "")),
            "effective_polarity": int(eff) if eff in (-1, 1) else 0,
            "has_amplitude": has_amp,
            "sig_raw_b64": b64_encode_array(sig_raw),
            "baseline_raw_b64": b64_encode_array(wf["baseline_raw"]) if has_amp else None,
            "amplitude_b64": b64_encode_array(wf["amplitude"]) if has_amp else None,
            "sig_int_b64": b64_encode_array(sig_int) if sig_int is not None and len(sig_int) else None,
            "t0_int": aux_int["t0"] if aux_int else None,
            "fs_int": aux_int["fs"] if aux_int else None,
            "length_int": aux_int["length"] if aux_int else 0,
            "sig_cal_b64": b64_encode_array(sig_cal) if sig_cal is not None and len(sig_cal) else None,
            "t0_cal": aux_cal["t0"] if aux_cal else None,
            "fs_cal": aux_cal["fs"] if aux_cal else None,
            "length_cal": aux_cal["length"] if aux_cal else 0,
            "runs": runs,
            "orientations": orientations,
        })

    return segments_data, Path(file_path).name


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cytocypher v2 - Contraction Review</title>
    <script>__PLOTLY_JS__</script>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .controls { display: flex; gap: 15px; margin-bottom: 12px; align-items: center; flex-wrap: wrap;}
        #plot { width: 100%; }
        .info { margin-bottom: 8px; font-weight: bold; }
        .params { margin-bottom: 8px; font-family: monospace; font-size: 13px; background: #f4f4f4; padding: 6px; }
        .fail { background: #fde8e8; color: #900; padding: 8px; border: 1px solid #c66; margin-bottom: 8px; font-weight: bold; border-radius: 4px; }
        .polarity-banner { background: #fff3cd; color: #856404; padding: 10px; border: 1px solid #ffeeba; margin-bottom: 8px; font-weight: bold; border-radius: 4px; }
        button { padding: 8px 14px; cursor: pointer; border: 1px solid #ccc; border-radius: 3px; background: #f8f9fa; }
        button:hover { background: #e2e6ea; }
        .btn-nav { padding: 5px 10px; cursor: pointer; border: 1px solid #aaa; border-radius: 3px; background: #f0f0f0; font-weight: bold; font-size: 13px; }
        .btn-nav:disabled { opacity: 0.4; cursor: not-allowed; background: #e0e0e0; }
        .btn-nav:hover:not(:disabled) { background: #dcdcdc; }
        .btn-primary { background: #007bff; color: white; border-color: #007bff; }
        .btn-primary:hover { background: #0069d9; }
        .summary { margin-top: 30px; border-top: 1px solid #ccc; padding-top: 20px; }
        table { border-collapse: collapse; width: 100%; font-size: 13px; }
        th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
        th { background: #eee; }
        select { padding: 5px 8px; border-radius: 3px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h2>Cytocypher v2 Contraction Review</h2>
    <div class="info">Source: <span id="source_name"></span> &nbsp;|&nbsp; Polarity Mode: <span id="polarity_mode_badge" style="background:#e8f0fe; color:#1a73e8; padding:3px 8px; border-radius:4px; font-weight:bold;">__POLARITY_MODE__</span></div>

    <div class="controls">
        <div style="display:flex; align-items:center; gap:6px;">
            <label for="segment_select" style="font-weight:bold;">Segment:</label>
            <button id="btn_prev" class="btn-nav" type="button" title="Previous segment (Left Arrow or 'p')">&larr; Previous</button>
            <select id="segment_select"></select>
            <button id="btn_next" class="btn-nav" type="button" title="Next segment (Right Arrow or 'n')">Next &rarr;</button>
        </div>
        <label>Polarity: <select id="polarity_select"></select></label>
        <label>Detector strictness: <select id="run_select"></select></label>
        <label><input type="checkbox" id="chk_rejected"> Show rejected candidates (diagnostic)</label>
        <label><input type="checkbox" id="chk_prelim_diagnostic"> Show preliminary candidates (diagnostic only)</label>
        <label><input type="checkbox" id="chk_raw"> Show raw PixelCorrelation</label>
        <label style="background:#eee; padding:5px; border-radius:3px;">
            <input type="checkbox" id="chk_reviewed"> Mark segment as reviewed</label>
        <div style="background:#fff0f0; padding:5px 8px; border-radius:4px; border:1px solid #f5c2c7; display:inline-flex; align-items:center; gap:6px;">
            <label style="color:#842029; font-weight:bold; cursor:pointer;" title="Reject entire recording from final counts and exports">
                <input type="checkbox" id="chk_segment_rejected"> Reject entire segment
            </label>
            <select id="segment_rejection_reason" style="display:none; font-size:12px; padding:3px 6px;">
                <option value="Poor signal / excessive noise">Poor signal / excessive noise</option>
                <option value="Tracking or motion artifact">Tracking or motion artifact</option>
                <option value="Incorrect recording">Incorrect recording</option>
                <option value="Other / researcher decision">Other / researcher decision</option>
            </select>
        </div>
        <button id="btn_save">Save annotations (CSV)</button>
        <button id="btn_export_overrides" class="btn-primary">Export overrides & rejections (JSON)</button>
    </div>
    <div id="peak_strictness_panel" style="background:#eef6ff; border:1px solid #b6d4fe; padding:7px 12px; border-radius:4px; margin-bottom:8px; display:flex; align-items:center; gap:14px; flex-wrap:wrap; font-size:13px;">
        <span style="font-weight:bold; color:#084298;">Per-Peak Detector Strictness:</span>
        <label style="font-weight:bold;">Peak to edit:
            <select id="peak_select" style="font-size:12px; padding:3px 6px; max-width:260px;">
                <option value="none">-- Select peak --</option>
            </select>
        </label>
        <span id="selected_peak_label" style="font-family:monospace; color:#333;">No peak selected (click any marker or choose from dropdown)</span>
        <label style="font-weight:bold;">Per-peak strictness:
            <select id="peak_strictness_select" disabled>
                <option value="inherit">Use global setting</option>
                <option value="permissive">Permissive</option>
                <option value="default">Default</option>
                <option value="strict">Strict</option>
            </select>
        </label>
        <span style="font-size:12px; color:#555;">
            Legend: <b style="color:#28a745;">&#9679; GREEN</b> = Automatically accepted &nbsp;|&nbsp;
            <b style="color:#007bff;">&#9679; BLUE</b> = Manually added &nbsp;|&nbsp;
            <b style="color:#dc3545;">&#10006; RED</b> = Rejected automatic peak
        </span>
    </div>
    <div style="margin-bottom:8px;">
        <b>Review:</b> click a <b style="color:#28a745;">GREEN</b> (automatically accepted) peak to reject it into a visible <b style="color:#dc3545;">RED CROSS</b>;
        click the same <b style="color:#dc3545;">RED CROSS</b> again to restore it to <b style="color:#28a745;">GREEN</b>;
        click on an unmarked part of the amplitude trace to add a missed contraction in <b style="color:#007bff;">BLUE</b> (snapped to the local maximum);
        click that same <b style="color:#007bff;">BLUE</b> peak again to delete it completely.
        Use <b>Per-peak strictness</b> above to evaluate an individual peak under <i>Permissive</i>, <i>Default</i>, or <i>Strict</i> detector criteria, or return to <i>Use global setting</i>.
        Use the <b>Polarity</b> dropdown to inspect or override physiological orientation (+1 or -1) without re-running Python code.
    </div>

    <div id="client_error_banner" class="fail" style="display:none; margin: 15px 0; font-size: 14px; background: #fff0f0; border: 2px solid #dc3545; color: #721c24; padding: 12px; border-radius: 4px;"></div>
    <div id="segment_info" class="info"></div>
    <div id="fail_box" class="fail" style="display:none"></div>
    <div id="param_box" class="params"></div>
    <div id="plot"></div>

    <div class="summary">
        <h3>Summary (selected strictness)</h3>
        <table id="summary_table">
            <thead><tr>
                <th>Segment</th><th>Sample ID</th><th>QC</th><th>Polarity</th><th>Reviewed</th>
                <th>Accepted (auto)</th><th>Manual additions</th><th>Manually rejected</th><th>Final count</th>
            </tr></thead>
            <tbody></tbody>
        </table>
    </div>

    <script>
        window.addEventListener('error', function (e) {
            console.error("Review UI Error:", e);
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

        let currentSegIdx = 0;
        let currentRun = DEFAULT_RUN;
        let selectedPeakInfo = null;     // { segIdx, polKey, index }
        const userPolarityOverride = {}; // key: segIdx -> 1, -1, or "auto"
        const userSegmentRejection = {}; // key: segIdx -> {rejected: bool, reason: str}
        const annotations = {};          // key `${seg}|${polKey}|${run}` -> array of event records
        const perPeakStrictness = {};    // key `${seg}|${polKey}` -> { [index]: "inherit"|"strict"|"default"|"permissive" }
        const manualPeaksByPol = {};     // key `${seg}|${polKey}` -> { [index]: {index, time_s, amp, raw, action: "ADDED"|"REJECTED"|"ADDED_REJECTED"} }
        const segmentReviewed = {};
        const decodedSignals = {};
        const decodedOrientations = {};

        if (initialSegmentRejections) {
            segments_data.forEach((seg, idx) => {
                const sNum = String(seg.segment_number);
                if (initialSegmentRejections[sNum] && initialSegmentRejections[sNum].rejected) {
                    userSegmentRejection[idx] = {
                        rejected: true,
                        reason: initialSegmentRejections[sNum].reason || "Other / researcher decision"
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
                    inten: s.sig_int_b64 ? decodeFloat32Array(s.sig_int_b64) : null,
                    cal: s.sig_cal_b64 ? decodeFloat32Array(s.sig_cal_b64) : null,
                };
            }
            return decodedSignals[segIdx];
        }

        function getOrientationArrays(segIdx, orientKey) {
            const cacheKey = segIdx + "_" + orientKey;
            if (!decodedOrientations[cacheKey]) {
                const seg = segments_data[segIdx];
                const o = seg.orientations ? seg.orientations[orientKey] : null;
                if (o) {
                    decodedOrientations[cacheKey] = {
                        amp: decodeFloat32Array(o.amplitude_b64),
                        base: decodeFloat32Array(o.baseline_raw_b64),
                    };
                } else {
                    decodedOrientations[cacheKey] = { amp: null, base: null };
                }
            }
            return decodedOrientations[cacheKey];
        }

        function timesFor(seg) {
            const t = new Float64Array(seg.length);
            for (let i = 0; i < seg.length; i++) t[i] = seg.t0 + i / seg.fs;
            return t;
        }

        function timesForCal(seg) {
            if (!seg.t0_cal || !seg.fs_cal || !seg.length_cal) return timesFor(seg);
            const t = new Float64Array(seg.length_cal);
            for (let i = 0; i < seg.length_cal; i++) t[i] = seg.t0_cal + i / seg.fs_cal;
            return t;
        }

        function timesForInt(seg) {
            if (!seg.t0_int || !seg.fs_int || !seg.length_int) return timesFor(seg);
            const t = new Float64Array(seg.length_int);
            for (let i = 0; i < seg.length_int; i++) t[i] = seg.t0_int + i / seg.fs_int;
            return t;
        }

        function getActivePolarity(segIdx) {
            const seg = segments_data[segIdx];
            if (userPolarityOverride[segIdx] !== undefined) {
                const val = userPolarityOverride[segIdx];
                if (val === 1 || val === -1) {
                    return { pol: val, source: "manual_override", isOverride: true };
                }
            }
            if (seg.polarity_source === "manual_override" && (seg.effective_polarity === 1 || seg.effective_polarity === -1)) {
                return { pol: seg.effective_polarity, source: "manual_override", isOverride: true };
            }
            return { pol: seg.auto_polarity, source: "automatic", isOverride: false };
        }

        function getRunEvents(segIdx, polKey, runName) {
            const seg = segments_data[segIdx];
            if ((polKey === "1" || polKey === "-1") && seg.orientations && seg.orientations[polKey] && seg.orientations[polKey].runs[runName]) {
                return seg.orientations[polKey].runs[runName].events;
            }
            if (seg.runs && seg.runs[runName]) {
                return seg.runs[runName].events;
            }
            return {index: [], time_s: [], amplitude: [], raw_value: [], status: [], reason: [], diagnostic_status: [], diagnostic_reason: []};
        }

        function findCandidateInRun(segIdx, polKey, runName, sampleIdx, tolSamples) {
            const ev = getRunEvents(segIdx, polKey, runName);
            let bestK = -1, bestDist = Infinity;
            for (let k = 0; k < ev.index.length; k++) {
                const d = Math.abs(ev.index[k] - sampleIdx);
                if (d <= tolSamples && d < bestDist) {
                    bestDist = d;
                    bestK = k;
                }
            }
            if (bestK < 0) return null;
            return {
                index: ev.index[bestK],
                time_s: ev.time_s[bestK],
                amp: ev.amplitude[bestK],
                raw: ev.raw_value[bestK],
                status: ev.status[bestK],
                reason: ev.reason[bestK] || ""
            };
        }

        function syncPeakState(segIdx, run, polKey) {
            const key = segIdx + "|" + polKey + "|" + run;
            const anns = annotations[key];
            if (!anns) return;
            const seg = segments_data[segIdx];
            const w = Math.max(1, Math.round(0.04 * seg.fs));
            const minSep = Math.max(2, Math.round(0.12 * seg.fs));
            const polMapKey = segIdx + "|" + polKey;
            const ppMap = perPeakStrictness[polMapKey] || {};
            const manMap = manualPeaksByPol[polMapKey] || {};

            // 1. Reset auto events to baseline run state, then apply per-peak strictness overrides
            for (let i = anns.length - 1; i >= 0; i--) {
                const a = anns[i];
                a.manually_rejected = false;
                if (a.original_status === "MISSING") {
                    const hasMan = Object.values(manMap).some(m => (m.action === "ADDED" || m.action === "ADDED_REJECTED") && Math.abs(m.index - a.index) <= w);
                    const hasPp = Object.keys(ppMap).some(k => ppMap[k] !== "inherit" && Math.abs(parseInt(k) - a.index) <= w);
                    if (!hasMan && !hasPp) {
                        anns.splice(i, 1);
                        continue;
                    }
                }
                let ovPreset = "inherit";
                for (const [kIdx, pVal] of Object.entries(ppMap)) {
                    if (Math.abs(parseInt(kIdx) - a.index) <= w) {
                        ovPreset = pVal;
                        break;
                    }
                }
                a.per_peak_strictness = ovPreset || "inherit";
                if (a.per_peak_strictness !== "inherit") {
                    const cand = findCandidateInRun(segIdx, polKey, a.per_peak_strictness, a.index, w);
                    a.origin = "auto";
                    if (cand) {
                        a.status = cand.status;
                        a.reason = cand.reason || (cand.status === "ACCEPTED" ? "" : `rejected_under_${a.per_peak_strictness}`);
                        a.amp = cand.amp;
                        a.raw = cand.raw;
                    } else {
                        a.status = "REJECTED";
                        a.reason = `not_accepted_under_${a.per_peak_strictness}`;
                    }
                } else if (a.original_status !== "MISSING") {
                    a.status = a.original_status;
                    a.reason = a.original_reason || "";
                    a.origin = "auto";
                }
            }

            // 2. Enforce minimum cycle separation (deduplication) when per-peak strictness accepts a peak
            const accList = anns.filter(a => a.status === "ACCEPTED" && a.origin === "auto");
            for (let i = 0; i < accList.length; i++) {
                for (let j = i + 1; j < accList.length; j++) {
                    const a = accList[i], b = accList[j];
                    if (a.status === "ACCEPTED" && b.status === "ACCEPTED" && Math.abs(a.index - b.index) < minSep) {
                        if ((b.amp || 0) > (a.amp || 0)) {
                            a.status = "REJECTED";
                            a.reason = "duplicate_peak_within_cycle";
                        } else {
                            b.status = "REJECTED";
                            b.reason = "duplicate_peak_within_cycle";
                        }
                    }
                }
            }

            // 3. Apply persistent manual additions / manual rejections across runs
            for (const mRec of Object.values(manMap)) {
                const existing = anns.find(a => Math.abs(a.index - mRec.index) <= w);
                if (mRec.action === "ADDED") {
                    if (existing) {
                        if (!existing.per_peak_strictness || existing.per_peak_strictness === "inherit") {
                            existing.status = "ACCEPTED";
                            existing.origin = "manual";
                            existing.manually_rejected = false;
                        }
                    } else {
                        anns.push({
                            index: mRec.index,
                            time_s: mRec.time_s,
                            amp: mRec.amp,
                            raw: mRec.raw,
                            status: "ACCEPTED",
                            original_status: "MISSING",
                            original_reason: "",
                            reason: "",
                            origin: "manual",
                            manually_rejected: false,
                            per_peak_strictness: "inherit"
                        });
                    }
                } else if (mRec.action === "REJECTED") {
                    if (existing && (!existing.per_peak_strictness || existing.per_peak_strictness === "inherit")) {
                        existing.status = "REJECTED";
                        existing.origin = "auto";
                        existing.manually_rejected = true;
                        existing.reason = "manual_rejection";
                    }
                }
            }
            anns.sort((a, b) => a.index - b.index);
        }

        function getAnn(segIdx, run, polKey) {
            const key = segIdx + "|" + polKey + "|" + run;
            if (!annotations[key]) {
                const seg = segments_data[segIdx];
                const w = Math.max(1, Math.round(0.04 * seg.fs));
                const ev = getRunEvents(segIdx, polKey, run);
                const diagStatuses = ev.diagnostic_status || ev.status || [];
                const diagReasons = ev.diagnostic_reason || ev.reason || [];
                const list = ev.index.map((idx, i) => ({
                    index: idx, time_s: ev.time_s[i], amp: ev.amplitude[i], raw: ev.raw_value[i],
                    status: ev.status[i], original_status: ev.status[i], original_reason: ev.reason[i] || "",
                    reason: ev.reason[i] || "", origin: "auto", manually_rejected: false, per_peak_strictness: "inherit",
                    diagnostic_status: diagStatuses[i] || ev.status[i],
                    diagnostic_reason: diagReasons[i] || ev.reason[i] || ""
                }));
                // Also include any candidate from 'permissive' not already in list so it can be selected for per-peak strictness
                const permEv = getRunEvents(segIdx, polKey, "permissive");
                if (permEv && permEv.index) {
                    const permDiagSt = permEv.diagnostic_status || permEv.status || [];
                    permEv.index.forEach((pIdx, i) => {
                        if (!list.some(a => Math.abs(a.index - pIdx) <= w)) {
                            list.push({
                                index: pIdx, time_s: permEv.time_s[i], amp: permEv.amplitude[i], raw: permEv.raw_value[i],
                                status: "REJECTED", original_status: "REJECTED",
                                original_reason: permEv.reason[i] || "below_global_strictness_threshold",
                                reason: permEv.reason[i] || "below_global_strictness_threshold",
                                origin: "auto", manually_rejected: false, per_peak_strictness: "inherit",
                                diagnostic_status: permDiagSt[i] || "REJECTED",
                                diagnostic_reason: permEv.reason[i] || "below_global_strictness_threshold"
                            });
                        }
                    });
                }
                list.sort((a, b) => a.index - b.index);
                annotations[key] = list;
            }
            syncPeakState(segIdx, run, polKey);
            return annotations[key];
        }

        function updateNavButtons() {
            const btnPrev = document.getElementById("btn_prev");
            const btnNext = document.getElementById("btn_next");
            if (btnPrev) btnPrev.disabled = (currentSegIdx <= 0);
            if (btnNext) btnNext.disabled = (currentSegIdx >= segments_data.length - 1);
        }

        function getBaseRunQCStatus(segIdx, activePol, runKey) {
            const seg = segments_data[segIdx];
            runKey = runKey || currentRun;
            if (activePol !== 1 && activePol !== -1) {
                return { status: "REJECT", reason: seg.auto_reason || "polarity_undetermined" };
            }
            const polKey = String(activePol);
            const oData = seg.orientations ? seg.orientations[polKey] : null;
            if (oData) {
                const r = (oData.runs && oData.runs[runKey]) ? oData.runs[runKey] : null;
                return {
                    status: (r && r.qc_status) ? r.qc_status : oData.qc_status,
                    reason: (r && r.qc_reason) ? r.qc_reason : oData.qc_reason
                };
            }
            return { status: seg.qc_status, reason: seg.qc_reason };
        }

        function countsFor(idx, pol, runKey) {
            runKey = runKey || currentRun;
            if (pol !== 1 && pol !== -1) {
                return {auto: 0, added: 0, rej: 0, final: 0};
            }
            const baseQC = getBaseRunQCStatus(idx, pol, runKey);
            const manualRej = userSegmentRejection[idx];
            if ((manualRej && manualRej.rejected) || baseQC.status !== "PASS") {
                return {auto: 0, added: 0, rej: 0, final: 0};
            }
            const anns = getAnn(idx, runKey, String(pol));
            let auto = 0, added = 0, rej = 0, final = 0;
            anns.forEach(a => {
                if (a.status === "ACCEPTED" && a.origin === "auto") auto++;
                if (a.origin === "manual" && a.status === "ACCEPTED") added++;
                if ((a.origin === "auto" && a.original_status === "ACCEPTED" && a.status === "REJECTED") || a.manually_rejected) rej++;
                if (a.status === "ACCEPTED") final++;
            });
            return {auto, added, rej, final};
        }

        function getActiveQC(segIdx, activePol, runKey) {
            const seg = segments_data[segIdx];
            runKey = runKey || currentRun;

            const manualRej = userSegmentRejection[segIdx];
            const isManualRejected = manualRej && manualRej.rejected;

            if (activePol !== 1 && activePol !== -1) {
                if (isManualRejected) {
                    return {
                        status: "REJECT",
                        reason: `manual: ${manualRej.reason}`,
                        label: `Automatic QC: REJECT (${seg.qc_reason || "polarity_undetermined"}) | Manual QC: REJECT (${manualRej.reason}) | Final: REJECT`,
                        isInputRequired: false,
                        isZeroContractions: false,
                        isManualRejection: true,
                        manualReason: manualRej.reason,
                        autoStatus: "REJECT",
                        autoReason: seg.qc_reason || "polarity_undetermined",
                        nAccepted: 0
                    };
                }
                if (seg.qc_status !== "PASS" && seg.qc_reason !== "polarity_undetermined") {
                    return {
                        status: seg.qc_status,
                        reason: seg.qc_reason,
                        label: `QC: ${seg.qc_status} (${seg.qc_reason})`,
                        isInputRequired: false,
                        isZeroContractions: false,
                        isManualRejection: false,
                        nAccepted: 0
                    };
                }
                return {
                    status: "INPUT_REQUIRED",
                    reason: seg.auto_reason || "polarity_undetermined",
                    label: "POLARITY INPUT REQUIRED",
                    isInputRequired: true,
                    isZeroContractions: false,
                    isManualRejection: false,
                    nAccepted: 0
                };
            }
            const baseQC = getBaseRunQCStatus(segIdx, activePol, runKey);
            const baseStatus = baseQC.status;
            const baseReason = baseQC.reason;

            if (isManualRejected) {
                const autoDesc = (baseStatus === "PASS") ? (baseReason || "clean_signal") : baseReason;
                return {
                    status: "REJECT",
                    reason: `manual: ${manualRej.reason}`,
                    label: `Automatic QC: ${baseStatus} (${autoDesc}) | Manual QC: REJECT (${manualRej.reason}) | Final: REJECT`,
                    isInputRequired: false,
                    isZeroContractions: false,
                    isManualRejection: true,
                    manualReason: manualRej.reason,
                    autoStatus: baseStatus,
                    autoReason: autoDesc,
                    nAccepted: 0
                };
            }

            if (baseStatus !== "PASS") {
                return {
                    status: baseStatus,
                    reason: baseReason,
                    label: `QC: ${baseStatus} (${baseReason})`,
                    isInputRequired: false,
                    isZeroContractions: false,
                    isManualRejection: false,
                    nAccepted: 0
                };
            }

            const c = countsFor(segIdx, activePol, runKey);
            const nAcc = c.final;
            if (nAcc === 0) {
                return {
                    status: "PASS",
                    reason: "no_accepted_contractions",
                    label: "QC: PASS (0 contractions)",
                    isInputRequired: false,
                    isZeroContractions: true,
                    isManualRejection: false,
                    nAccepted: 0
                };
            }

            return {
                status: "PASS",
                reason: baseReason || "clean_signal",
                label: `QC: PASS (${nAcc} contractions)`,
                isInputRequired: false,
                isZeroContractions: false,
                isManualRejection: false,
                nAccepted: nAcc
            };
        }

        function populateSegmentSelector() {
            const sel = document.getElementById("segment_select");
            sel.innerHTML = "";
            segments_data.forEach((seg, idx) => {
                const opt = document.createElement("option");
                opt.value = idx;
                const polInfo = getActivePolarity(idx);
                const qc = getActiveQC(idx, polInfo.pol);
                let polDesc = "";
                if (polInfo.pol === 0) {
                    polDesc = qc.isInputRequired ? "POLARITY INPUT REQUIRED" : "undetermined";
                } else if (polInfo.isOverride) {
                    polDesc = `override: ${polInfo.pol > 0 ? '+1' : '-1'}`;
                } else {
                    polDesc = `auto: ${polInfo.pol > 0 ? '+1' : '-1'}`;
                }
                let qcShort = "";
                if (qc.isManualRejection) {
                    qcShort = `MANUAL REJECT (${qc.manualReason})`;
                } else if (qc.isInputRequired) {
                    qcShort = "QC: REJECT (polarity_undetermined)";
                } else if (qc.status !== "PASS") {
                    qcShort = `QC: ${qc.status} (${qc.reason})`;
                } else if (qc.isZeroContractions) {
                    qcShort = "QC: PASS (0 events)";
                } else {
                    qcShort = `QC: PASS (${qc.nAccepted})`;
                }
                opt.text = `${seg.sheet_name} (${polDesc}, ${qcShort})`;
                sel.appendChild(opt);
                if (segmentReviewed[idx] === undefined) segmentReviewed[idx] = false;
            });
            sel.value = currentSegIdx;
            updateNavButtons();
        }

        function updatePolaritySelect() {
            const psel = document.getElementById("polarity_select");
            psel.innerHTML = "";
            const seg = segments_data[currentSegIdx];
            const autoDesc = seg.auto_polarity === 0 ? "undetermined" : (seg.auto_polarity > 0 ? "+1 upward" : "-1 downward");

            const optAuto = document.createElement("option");
            optAuto.value = "auto";
            optAuto.text = `AUTO (${autoDesc})`;
            psel.appendChild(optAuto);

            const optUp = document.createElement("option");
            optUp.value = "1";
            optUp.text = "Raw +1 (upward contractions)";
            psel.appendChild(optUp);

            const optDown = document.createElement("option");
            optDown.value = "-1";
            optDown.text = "Raw -1 (downward contractions)";
            psel.appendChild(optDown);

            if (userPolarityOverride[currentSegIdx] !== undefined) {
                psel.value = String(userPolarityOverride[currentSegIdx]);
            } else if (seg.polarity_source === "manual_override") {
                psel.value = String(seg.effective_polarity);
            } else {
                psel.value = "auto";
            }
        }

        function populateSelectors() {
            populateSegmentSelector();
            const sel = document.getElementById("segment_select");
            sel.addEventListener("change", e => {
                currentSegIdx = parseInt(e.target.value);
                renderPlot();
            });

            const btnPrev = document.getElementById("btn_prev");
            if (btnPrev) {
                btnPrev.addEventListener("click", () => {
                    if (currentSegIdx > 0) {
                        currentSegIdx--;
                        sel.value = currentSegIdx;
                        renderPlot();
                    }
                });
            }

            const btnNext = document.getElementById("btn_next");
            if (btnNext) {
                btnNext.addEventListener("click", () => {
                    if (currentSegIdx < segments_data.length - 1) {
                        currentSegIdx++;
                        sel.value = currentSegIdx;
                        renderPlot();
                    }
                });
            }

            window.addEventListener("keydown", function(e) {
                const tag = document.activeElement ? document.activeElement.tagName.toUpperCase() : "";
                if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
                if (e.ctrlKey || e.altKey || e.metaKey) return;
                if (e.key === "ArrowLeft" || e.key === "p" || e.key === "P") {
                    if (currentSegIdx > 0) {
                        e.preventDefault();
                        currentSegIdx--;
                        sel.value = currentSegIdx;
                        renderPlot();
                    }
                } else if (e.key === "ArrowRight" || e.key === "n" || e.key === "N") {
                    if (currentSegIdx < segments_data.length - 1) {
                        e.preventDefault();
                        currentSegIdx++;
                        sel.value = currentSegIdx;
                        renderPlot();
                    }
                }
            });

            const psel = document.getElementById("polarity_select");
            psel.addEventListener("change", e => {
                const val = e.target.value;
                if (val === "auto") {
                    delete userPolarityOverride[currentSegIdx];
                } else {
                    userPolarityOverride[currentSegIdx] = parseInt(val);
                }
                populateSegmentSelector();
                renderPlot();
            });

            const rsel = document.getElementById("run_select");
            const firstSegRuns = (segments_data[0].orientations && segments_data[0].orientations["-1"]) ?
                segments_data[0].orientations["-1"].runs : segments_data[0].runs;
            Object.keys(firstSegRuns).forEach(name => {
                const opt = document.createElement("option");
                opt.value = name; opt.text = name; rsel.appendChild(opt);
            });
            rsel.value = DEFAULT_RUN;
            rsel.addEventListener("change", e => {
                currentRun = e.target.value;
                populateSegmentSelector();
                renderPlot();
            });

            document.getElementById("chk_reviewed").addEventListener("change", e => {
                segmentReviewed[currentSegIdx] = e.target.checked; updateSummaryTable();
            });
            document.getElementById("chk_rejected").addEventListener("change", renderPlot);
            const chkPrelim = document.getElementById("chk_prelim_diagnostic");
            if (chkPrelim) chkPrelim.addEventListener("change", renderPlot);
            document.getElementById("chk_raw").addEventListener("change", renderPlot);

            const chkRej = document.getElementById("chk_segment_rejected");
            const selRejReason = document.getElementById("segment_rejection_reason");
            chkRej.addEventListener("change", e => {
                if (e.target.checked) {
                    selRejReason.style.display = "inline-block";
                    userSegmentRejection[currentSegIdx] = {
                        rejected: true,
                        reason: selRejReason.value
                    };
                } else {
                    selRejReason.style.display = "none";
                    delete userSegmentRejection[currentSegIdx];
                }
                populateSegmentSelector();
                updateSummaryTable();
                renderPlot();
            });
            selRejReason.addEventListener("change", e => {
                if (userSegmentRejection[currentSegIdx]) {
                    userSegmentRejection[currentSegIdx].reason = e.target.value;
                    populateSegmentSelector();
                    updateSummaryTable();
                    renderPlot();
                }
            });

            const pselPeakToEdit = document.getElementById("peak_select");
            if (pselPeakToEdit) {
                pselPeakToEdit.addEventListener("change", e => {
                    const val = e.target.value;
                    const polInfo = getActivePolarity(currentSegIdx);
                    const activePol = polInfo.pol;
                    const polKey = (activePol !== 1 && activePol !== -1) ? "0" : String(activePol);
                    if (val === "none") {
                        selectedPeakInfo = null;
                    } else {
                        selectedPeakInfo = { segIdx: currentSegIdx, polKey: polKey, index: parseInt(val) };
                    }
                    const anns = getAnn(currentSegIdx, currentRun, polKey);
                    updateSelectedPeakBar(anns, polKey);
                    renderPlot();
                });
            }

            const pselPeak = document.getElementById("peak_strictness_select");
            if (pselPeak) {
                pselPeak.addEventListener("change", e => {
                    if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) return;
                    const preset = e.target.value;
                    const polMapKey = selectedPeakInfo.segIdx + "|" + selectedPeakInfo.polKey;
                    if (!perPeakStrictness[polMapKey]) perPeakStrictness[polMapKey] = {};
                    const seg = segments_data[currentSegIdx];
                    const w = Math.max(1, Math.round(0.04 * seg.fs));
                    // Clear any manual add/reject entry near this peak so detector preset rules apply cleanly
                    if (manualPeaksByPol[polMapKey]) {
                        Object.keys(manualPeaksByPol[polMapKey]).forEach(k => {
                            if (Math.abs(parseInt(k) - selectedPeakInfo.index) <= w) {
                                delete manualPeaksByPol[polMapKey][k];
                            }
                        });
                    }
                    // Update perPeakStrictness map
                    Object.keys(perPeakStrictness[polMapKey]).forEach(k => {
                        if (Math.abs(parseInt(k) - selectedPeakInfo.index) <= w) {
                            delete perPeakStrictness[polMapKey][k];
                        }
                    });
                    if (preset !== "inherit") {
                        perPeakStrictness[polMapKey][selectedPeakInfo.index] = preset;
                    }
                    syncPeakState(currentSegIdx, currentRun, selectedPeakInfo.polKey);
                    populateSegmentSelector();
                    updateSummaryTable();
                    renderPlot();
                });
            }

            const btnRemManual = document.getElementById("btn_remove_manual_peak");
            if (btnRemManual) {
                btnRemManual.addEventListener("click", () => {
                    if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx) return;
                    const polMapKey = selectedPeakInfo.segIdx + "|" + selectedPeakInfo.polKey;
                    const seg = segments_data[currentSegIdx];
                    const w = Math.max(1, Math.round(0.04 * seg.fs));
                    const selIdx = selectedPeakInfo.index;
                    if (manualPeaksByPol[polMapKey]) {
                        Object.keys(manualPeaksByPol[polMapKey]).forEach(k => {
                            if (Math.abs(parseInt(k) - selIdx) <= w) {
                                delete manualPeaksByPol[polMapKey][k];
                            }
                        });
                    }
                    if (perPeakStrictness[polMapKey]) {
                        Object.keys(perPeakStrictness[polMapKey]).forEach(k => {
                            if (Math.abs(parseInt(k) - selIdx) <= w) {
                                delete perPeakStrictness[polMapKey][k];
                            }
                        });
                    }
                    Object.keys(annotations).forEach(annK => {
                        if (annK.startsWith(currentSegIdx + "|" + selectedPeakInfo.polKey + "|")) {
                            const list = annotations[annK];
                            for (let i = list.length - 1; i >= 0; i--) {
                                if (Math.abs(list[i].index - selIdx) <= w) {
                                    if (list[i].origin === "manual" || list[i].original_status === "MISSING") {
                                        list.splice(i, 1);
                                    } else {
                                        list[i].status = list[i].original_status;
                                        list[i].reason = list[i].original_reason || "";
                                        list[i].origin = "auto";
                                        list[i].manually_rejected = false;
                                    }
                                }
                            }
                        }
                    });
                    selectedPeakInfo = null;
                    syncPeakState(currentSegIdx, currentRun, String(getActivePolarity(currentSegIdx).pol));
                    populateSegmentSelector();
                    updateSummaryTable();
                    renderPlot();
                });
            }

            document.getElementById("source_name").innerText = source_workbook;
            document.getElementById("btn_save").addEventListener("click", saveAnnotations);
            document.getElementById("btn_export_overrides").addEventListener("click", exportPolarityOverrides);
        }

        function updatePeakSelector(anns, polKey) {
            const psel = document.getElementById("peak_select");
            if (!psel) return;
            psel.innerHTML = "";
            const defOpt = document.createElement("option");
            defOpt.value = "none";
            defOpt.text = "-- Select peak --";
            psel.appendChild(defOpt);

            const sortedAnns = anns.slice().sort((a, b) => a.index - b.index);
            sortedAnns.forEach(a => {
                const opt = document.createElement("option");
                opt.value = String(a.index);
                const tag = (a.origin === "manual") ? "Manual [BLUE]" : (a.status === "ACCEPTED" ? "Auto accepted [GREEN]" : "Rejected auto [RED]");
                const strictTag = (a.per_peak_strictness && a.per_peak_strictness !== "inherit") ? ` (${a.per_peak_strictness})` : "";
                opt.text = `${a.time_s.toFixed(3)}s — ${tag}${strictTag}`;
                psel.appendChild(opt);
            });

            if (selectedPeakInfo && selectedPeakInfo.segIdx === currentSegIdx && selectedPeakInfo.polKey === polKey) {
                const seg = segments_data[currentSegIdx];
                const w = Math.max(1, Math.round(0.04 * seg.fs));
                const matched = anns.find(a => Math.abs(a.index - selectedPeakInfo.index) <= w);
                if (matched) {
                    psel.value = String(matched.index);
                } else {
                    psel.value = "none";
                }
            } else {
                psel.value = "none";
            }
        }

        function updateSelectedPeakBar(anns, polKey) {
            const lbl = document.getElementById("selected_peak_label");
            const sel = document.getElementById("peak_strictness_select");
            const btnRem = document.getElementById("btn_remove_manual_peak");
            if (!lbl || !sel) return;
            if (!selectedPeakInfo || selectedPeakInfo.segIdx !== currentSegIdx || selectedPeakInfo.polKey !== polKey) {
                lbl.innerText = "No peak selected (click any marker or choose from dropdown)";
                sel.value = "inherit";
                sel.disabled = true;
                if (btnRem) btnRem.style.display = "none";
                return;
            }
            const seg = segments_data[currentSegIdx];
            const w = Math.max(1, Math.round(0.04 * seg.fs));
            const ev = anns.find(a => Math.abs(a.index - selectedPeakInfo.index) <= w);
            if (!ev) {
                lbl.innerText = "No peak selected (click any marker or choose from dropdown)";
                sel.value = "inherit";
                sel.disabled = true;
                if (btnRem) btnRem.style.display = "none";
                return;
            }
            const originTxt = (ev.origin === "manual") ? "manual" : "auto";
            const isOverridden = (ev.per_peak_strictness && ev.per_peak_strictness !== "inherit");
            const strictVal = isOverridden ? ev.per_peak_strictness : currentRun;
            const strictSource = isOverridden ? "locally overridden" : "inherited";
            const strictTxt = `${strictVal} (${strictSource})`;
            const stateTxt = ev.status;
            const reasonTxt = ev.reason ? ` [Reason: ${ev.reason}]` : "";
            lbl.innerHTML = `Selected: <b>t=${ev.time_s.toFixed(3)}s</b> | Amp=<b>${ev.amp == null ? "n/a" : ev.amp.toFixed(4)}</b> | Origin: <b>${originTxt}</b> | Strictness: <b>${strictTxt}</b> | State: <b>${stateTxt}</b>${reasonTxt ? ` | <b>${reasonTxt}</b>` : ""}`;
            sel.disabled = false;
            sel.value = isOverridden ? ev.per_peak_strictness : "inherit";
            if (btnRem) {
                btnRem.style.display = (ev.origin === "manual") ? "inline-block" : "none";
            }
        }

        function updateSummaryTable() {
            const tbody = document.querySelector("#summary_table tbody");
            tbody.innerHTML = "";
            segments_data.forEach((seg, idx) => {
                const polInfo = getActivePolarity(idx);
                const c = countsFor(idx, polInfo.pol);
                const qc = getActiveQC(idx, polInfo.pol);
                const tr = document.createElement("tr");
                let polBadge = "";
                if (polInfo.pol === 0) {
                    polBadge = qc.isInputRequired ? '<span style="color:#d9534f; font-weight:bold;">POLARITY INPUT REQUIRED</span>' : '<span style="color:#6c757d;">undetermined (rejected)</span>';
                } else if (polInfo.pol > 0) {
                    polBadge = `+1 (up) ${polInfo.isOverride ? '<span style="color:#0275d8; font-weight:bold;">[override]</span>' : '<span style="color:#5cb85c;">[auto]</span>'}`;
                } else {
                    polBadge = `-1 (down) ${polInfo.isOverride ? '<span style="color:#0275d8; font-weight:bold;">[override]</span>' : '<span style="color:#5cb85c;">[auto]</span>'}`;
                }

                let qcHtml = "";
                if (qc.isManualRejection) {
                    qcHtml = `<span style="background:#f8d7da; color:#842029; padding:2px 6px; border-radius:3px; font-weight:bold;">MANUAL REJECT</span> <small>(${qc.manualReason})</small><br><small style="color:#666;">Auto: ${qc.autoStatus}</small>`;
                } else if (qc.isInputRequired) {
                    qcHtml = '<span style="color:#d9534f; font-weight:bold;">REJECT (polarity_undetermined)</span>';
                } else if (qc.status !== "PASS") {
                    qcHtml = `<span style="color:#d9534f; font-weight:bold;">${qc.status} (${qc.reason})</span>`;
                } else if (qc.isZeroContractions) {
                    qcHtml = '<span style="color:#e67e22; font-weight:bold;">PASS (0 contractions)</span>';
                } else {
                    qcHtml = `<span style="color:#28a745; font-weight:bold;">PASS (${qc.reason})</span>`;
                }

                tr.innerHTML = `<td>${seg.sheet_name}</td><td>${seg.sample_id}</td>
                    <td>${qcHtml}</td>
                    <td>${polBadge}</td><td>${segmentReviewed[idx] ? "yes" : "no"}</td>
                    <td>${c.auto}</td><td>${c.added}</td><td>${c.rej}</td><td>${c.final}</td>`;
                tbody.appendChild(tr);
            });
        }

        function finiteMax(arr) {
            let m = 0;
            for (let i = 0; i < arr.length; i++) if (Number.isFinite(arr[i]) && arr[i] > m) m = arr[i];
            return m;
        }

        function renderPlot() {
            const seg = segments_data[currentSegIdx];
            const polInfo = getActivePolarity(currentSegIdx);
            const activePol = polInfo.pol;
            const isUndet = (activePol !== 1 && activePol !== -1);
            const polKey = isUndet ? "0" : String(activePol);

            updatePolaritySelect();
            updateNavButtons();
            document.getElementById("chk_reviewed").checked = !!segmentReviewed[currentSegIdx];

            const chkRej = document.getElementById("chk_segment_rejected");
            const selRejReason = document.getElementById("segment_rejection_reason");
            const curManualRej = userSegmentRejection[currentSegIdx];
            if (curManualRej && curManualRej.rejected) {
                chkRej.checked = true;
                selRejReason.style.display = "inline-block";
                selRejReason.value = curManualRej.reason || "Poor signal / excessive noise";
            } else {
                chkRej.checked = false;
                selRejReason.style.display = "none";
            }

            const baseSig = getBaseSignals(currentSegIdx);
            const times = timesFor(seg);
            const qc = getActiveQC(currentSegIdx, activePol);
            const isQCFailed = (qc.status !== "PASS");

            let run = null;
            let ampArr = null;
            let baseArr = null;

            if (!isUndet && seg.orientations && seg.orientations[polKey]) {
                const oData = seg.orientations[polKey];
                const oArrays = getOrientationArrays(currentSegIdx, polKey);
                ampArr = oArrays.amp;
                baseArr = oArrays.base;
                run = oData.runs[currentRun];
            } else {
                run = seg.runs[currentRun] || {params: {}, reference_amplitude: null, amplitude_threshold: null, noise_sigma: null, events: {index:[]}};
            }

            const hasAmp = (ampArr !== null && ampArr.length > 0);
            const showRej = document.getElementById("chk_rejected").checked;
            const chkPrelim = document.getElementById("chk_prelim_diagnostic");
            const showPrelimDiag = chkPrelim ? chkPrelim.checked : false;
            const showRaw = document.getElementById("chk_raw").checked || !hasAmp || isQCFailed;

            let polTxt = "";
            if (isUndet) {
                polTxt = qc.isInputRequired
                    ? `<span style="color:#d9534f; font-weight:bold;">POLARITY INPUT REQUIRED</span> [${seg.auto_reason}]`
                    : `<span style="color:#6c757d; font-weight:bold;">undetermined</span> [${seg.auto_reason}]`;
            } else {
                const dir = activePol > 0 ? "+1 (contractions upward in raw)" : "-1 (contractions downward in raw)";
                const tag = polInfo.isOverride ? "[manual override]" : `[automatic: ${seg.polarity_method}]`;
                polTxt = `${dir} ${tag}`;
            }

            let qcHeaderTxt = "";
            if (qc.isManualRejection) {
                qcHeaderTxt = `<span style="background:#f8d7da; color:#842029; padding:2px 6px; border-radius:3px; font-weight:bold;">MANUAL REJECT</span> <small>(${qc.manualReason})</small>`;
            } else if (qc.isInputRequired) {
                qcHeaderTxt = '<span style="color:#d9534f; font-weight:bold;">QC: REJECT (polarity_undetermined)</span>';
            } else if (qc.status !== "PASS") {
                qcHeaderTxt = `<span style="color:#d9534f; font-weight:bold;">QC: ${qc.status} (${qc.reason})</span>`;
            } else if (qc.isZeroContractions) {
                qcHeaderTxt = '<span style="color:#e67e22; font-weight:bold;">QC: PASS (0 contractions)</span>';
            } else {
                qcHeaderTxt = `<span style="color:#28a745; font-weight:bold;">QC: PASS (${qc.reason})</span>`;
            }

            document.getElementById("segment_info").innerHTML =
                `Sample ID: ${seg.sample_id} | ${seg.length} pts | fs: ${seg.fs} Hz | t0: ${seg.t0.toFixed(3)} s | ` +
                `raw polarity: ${polTxt} | ${qcHeaderTxt}`;

            const fb = document.getElementById("fail_box");
            if (qc.isManualRejection) {
                fb.className = "fail";
                fb.style.display = "block";
                fb.style.background = "#fff3cd";
                fb.style.color = "#856404";
                fb.style.borderColor = "#ffeeba";
                fb.innerHTML = `⚠️ <b>MANUALLY REJECTED by researcher:</b> ${qc.manualReason}.<br>` +
                    `<small style="color:#555;">(Automatic QC: ${qc.autoStatus} [${qc.autoReason}]). Final reportable accepted contraction count is 0. Uncheck 'Reject entire segment' above to restore this recording.</small>`;
            } else if (qc.isInputRequired) {
                fb.className = "polarity-banner";
                fb.style.display = "block";
                fb.style.background = "";
                fb.style.color = "";
                fb.style.borderColor = "";
                fb.innerHTML = `⚠️ <b>POLARITY INPUT REQUIRED:</b> Automatic physiological polarity could not be determined for this recording (${seg.auto_reason}).<br>` +
                    `Select <b>Raw +1</b> or <b>Raw -1</b> in the Polarity dropdown above to inspect and evaluate contraction events.`;
            } else if (qc.status !== "PASS") {
                fb.className = "fail";
                fb.style.display = "block";
                fb.style.background = "";
                fb.style.color = "";
                fb.style.borderColor = "";
                if (qc.reason === "noise_dominated_uninterpretable") {
                    fb.innerHTML = `Segment interpretability failure (<b>${qc.status}: ${qc.reason}</b>) — signal is dominated by noise/artefacts without reliable contraction evidence. Final reportable accepted contractions: <b>0</b> (rate: <b>N/A</b>). Check <i>'Show preliminary candidates (diagnostic only)'</i> above to inspect excluded preliminary events.`;
                } else {
                    fb.innerHTML = `Segment quality control failure (<b>${qc.status}: ${qc.reason}</b>). Final reportable accepted contractions: <b>0</b> (rate: <b>N/A</b>).`;
                }
            } else if (qc.isZeroContractions) {
                fb.className = "params";
                fb.style.display = "block";
                fb.style.background = "";
                fb.style.color = "";
                fb.style.borderColor = "";
                fb.innerHTML = `ℹ️ <b>No Contractions Detected (Clean Quiescent Signal):</b> Technical and signal interpretability QC passed, and zero contractions met the amplitude threshold and recovery criteria under strictness [${currentRun}].`;
            } else {
                fb.style.display = "none";
            }

            const anns = getAnn(currentSegIdx, currentRun, polKey);
            updateSelectedPeakBar(anns, polKey);
            updatePeakSelector(anns, polKey);
            const activeCounts = countsFor(currentSegIdx, activePol, currentRun);
            const nAcc = activeCounts.final;

            const p = run.params || {};
            const sigCat = run.signal_quality_category || run.qc_layer2_reason || "n/a";
            const durTxt = (run.analyzable_duration_s !== null && run.analyzable_duration_s !== undefined)
                ? `${run.analyzable_duration_s.toFixed(2)}s (${(100 * (run.analyzable_fraction || 1)).toFixed(0)}%)`
                : "n/a";
            let rateTxt = "N/A";
            if (!isQCFailed && !qc.isManualRejection && run.analyzable_duration_s && run.analyzable_duration_s > 0 && nAcc > 0) {
                rateTxt = `${(nAcc / run.analyzable_duration_s).toFixed(2)} Hz`;
            }
            const exclTxt = (run.excluded_intervals_s && run.excluded_intervals_s.length > 0)
                ? `  |  excluded intervals=${run.excluded_intervals_s.map(iv => iv[0].toFixed(2) + "-" + iv[1].toFixed(2) + "s").join(", ")}`
                : "";
            const sepStrict = (run.separation_strictness !== null && run.separation_strictness !== undefined)
                ? run.separation_strictness.toFixed(2)
                : ((p.separation_strictness !== undefined && p.separation_strictness !== null) ? p.separation_strictness.toFixed(2) : "1.00");
            const ampStrict = (run.amplitude_strictness !== null && run.amplitude_strictness !== undefined)
                ? run.amplitude_strictness.toFixed(2)
                : ((p.amplitude_strictness !== undefined && p.amplitude_strictness !== null) ? p.amplitude_strictness.toFixed(2) : "1.00");

            document.getElementById("param_box").innerText =
                `Active detector parameters [${currentRun}]: strictness=(sep:${sepStrict}, amp:${ampStrict})  ` +
                `min_rel_amplitude=${p.min_rel_amplitude || "0.50"}  ` +
                `min_rel_recovery=${p.min_rel_recovery || "0.50"}  min_snr=${p.min_snr || "8"}  |  ` +
                `reference amplitude=${run.reference_amplitude == null ? "n/a" : run.reference_amplitude.toFixed(4)}  ` +
                `amplitude threshold=${run.amplitude_threshold == null ? "n/a" : run.amplitude_threshold.toFixed(4)}  ` +
                `noise sigma=${run.noise_sigma == null ? "n/a" : run.noise_sigma.toExponential(2)}  |  ` +
                `quality=${sigCat}  valid duration=${durTxt}  rate=${rateTxt}${exclTxt}`;

            const panels = [];
            if (hasAmp) panels.push({key: "amp", title: "Contraction amplitude A<br>(relative to resting baseline)", weight: 3});
            if (showRaw) panels.push({key: "raw", title: "Raw PixelCorrelation", weight: 2});
            if (baseSig.inten) panels.push({key: "int", title: "PixelIntensity", weight: 1.2});
            if (baseSig.cal) panels.push({key: "cal", title: "RatioMetricCalcium", weight: 1.2});

            const totalW = panels.reduce((s, q) => s + q.weight, 0);
            const gap = 0.04;
            let top = 1.0;
            panels.forEach((q, i) => {
                const h = (1 - gap * (panels.length - 1)) * q.weight / totalW;
                q.domain = [top - h, top]; top -= h + gap; q.axis = i + 1;
                q.ax = "y" + (i === 0 ? "" : (i + 1));
            });
            const P = {}; panels.forEach(q => P[q.key] = q);

            const traces = [];
            if (P.amp && ampArr) {
                traces.push({x: times, y: ampArr, mode: "lines", name: "amplitude", line: {color: "black", width: 1},
                             yaxis: P.amp.ax, xaxis: "x", hoverinfo: "x+y"});
                traces.push({x: [times[0], times[times.length - 1]], y: [0, 0], mode: "lines",
                             name: "resting baseline (A = 0)", line: {color: "gray", width: 1}, yaxis: P.amp.ax, xaxis: "x", hoverinfo: "skip"});
                if (run.amplitude_threshold !== null && run.amplitude_threshold !== undefined) {
                    traces.push({x: [times[0], times[times.length - 1]], y: [run.amplitude_threshold, run.amplitude_threshold],
                                 mode: "lines", name: "major-contraction threshold", line: {color: "orange", dash: "dash", width: 1.2},
                                 yaxis: P.amp.ax, xaxis: "x", hoverinfo: "skip"});
                }
            }
            if (P.raw) {
                traces.push({x: times, y: baseSig.raw, mode: "lines", name: "raw PixelCorrelation", line: {color: "blue", width: 1},
                             yaxis: P.raw.ax, xaxis: "x"});
                if (baseArr) traces.push({x: times, y: baseArr, mode: "lines", name: "resting baseline B_raw",
                             line: {color: "gray", dash: "dot", width: 1.2}, yaxis: P.raw.ax, xaxis: "x", hoverinfo: "skip"});
            }
            if (P.int && baseSig.inten) traces.push({x: timesForInt(seg), y: baseSig.inten, mode: "lines", name: "PixelIntensity", line: {color: "green", width: 1}, yaxis: P.int.ax, xaxis: "x", connectgaps: false});
            if (P.cal && baseSig.cal) traces.push({x: timesForCal(seg), y: baseSig.cal, mode: "lines", name: "RatioMetricCalcium", line: {color: "firebrick", width: 1}, yaxis: P.cal.ax, xaxis: "x", connectgaps: false});

            function customMarkerTrace(predicateFn, panelKey, yField, style, name) {
                const pk = P[panelKey]; if (!pk) return;
                const xs = [], ys = [], cd = [], tx = [];
                anns.forEach((a, i) => {
                    if (!predicateFn(a)) return;
                    const y = a[yField]; if (y === null || y === undefined) return;
                    xs.push(a.time_s); ys.push(y); cd.push(i);
                    const strictTag = (a.per_peak_strictness && a.per_peak_strictness !== "inherit")
                        ? ` [per-peak strictness: ${a.per_peak_strictness}]`
                        : "";
                    let sourceText = "";
                    if (isQCFailed) {
                        sourceText = `EXCLUDED (QC ${qc.status}: ${qc.reason}) [diagnostic preliminary candidate]`;
                    } else if (a.origin === "manual") {
                        sourceText = "Source: Manual addition (click to delete)" + strictTag;
                    } else if (a.manually_rejected) {
                        sourceText = "MANUALLY REJECTED (was auto ACCEPTED — click red cross to restore to GREEN)" + strictTag;
                    } else {
                        sourceText = a.status + (a.reason ? ": " + a.reason : "") + strictTag;
                    }
                    tx.push(`t=${a.time_s.toFixed(3)} s<br>A=${a.amp == null ? "n/a" : a.amp.toFixed(4)}<br>${sourceText}`);
                });
                traces.push(Object.assign({x: xs, y: ys, customdata: cd, text: tx, hoverinfo: "text", mode: "markers", name: name,
                                           yaxis: pk.ax, xaxis: "x", type: "scatter"}, style));
            }

            if (!isUndet && hasAmp) {
                if (isQCFailed || qc.isManualRejection) {
                    // Segment failed QC or was manually rejected: NO GREEN or BLUE accepted markers may appear!
                    if (showPrelimDiag || qc.isManualRejection) {
                        const predPrelim = a => (a.diagnostic_status === "ACCEPTED" || a.status === "ACCEPTED");
                        customMarkerTrace(predPrelim, "amp", "amp", {marker: {color: "#888888", size: 8, symbol: "circle-open"}, opacity: 0.65}, "EXCLUDED preliminary candidates (diagnostic)");
                        customMarkerTrace(predPrelim, "raw", "raw", {marker: {color: "#888888", size: 7, symbol: "circle-open"}, opacity: 0.65, showlegend: false}, "EXCLUDED (raw)");
                    }
                    if (showRej) {
                        const predDiagRej = a => (a.diagnostic_status !== "ACCEPTED" && a.status !== "ACCEPTED");
                        customMarkerTrace(predDiagRej, "amp", "amp", {marker: {color: "#dc3545", size: 8, symbol: "x"}, opacity: 0.65}, "REJECTED candidates (diagnostic)");
                        customMarkerTrace(predDiagRej, "raw", "raw", {marker: {color: "#dc3545", size: 8, symbol: "x"}, opacity: 0.65, showlegend: false}, "REJECTED (raw)");
                    }
                } else {
                    const nAutoAcc = anns.filter(a => a.status === "ACCEPTED" && a.origin !== "manual").length;
                    const nManualAcc = anns.filter(a => a.status === "ACCEPTED" && a.origin === "manual").length;
                    const nManRej = anns.filter(a => a.status === "REJECTED" && a.manually_rejected).length;

                    customMarkerTrace(a => a.status === "ACCEPTED" && a.origin !== "manual", "amp", "amp",
                        {marker: {color: "#28a745", size: 10, symbol: "circle"}}, `Automatically accepted (${nAutoAcc})`);
                    customMarkerTrace(a => a.status === "ACCEPTED" && a.origin !== "manual", "raw", "raw",
                        {marker: {color: "#28a745", size: 9, symbol: "circle"}, showlegend: false}, "Automatically accepted (raw)");

                    customMarkerTrace(a => a.status === "ACCEPTED" && a.origin === "manual", "amp", "amp",
                        {marker: {color: "#007bff", size: 10, symbol: "circle", line: {color: "#004085", width: 1.5}}}, `Manually added (${nManualAcc})`);
                    customMarkerTrace(a => a.status === "ACCEPTED" && a.origin === "manual", "raw", "raw",
                        {marker: {color: "#007bff", size: 9, symbol: "circle", line: {color: "#004085", width: 1.5}}, showlegend: false}, "Manually added (raw)");

                    // Explicitly manually rejected auto peaks ALWAYS remain visible as a RED CROSS regardless of chk_rejected!
                    customMarkerTrace(a => a.status === "REJECTED" && a.manually_rejected, "amp", "amp",
                        {marker: {color: "#dc3545", size: 10, symbol: "x", line: {color: "#a71d2a", width: 1.5}}}, `Rejected automatic peak (${nManRej})`);
                    customMarkerTrace(a => a.status === "REJECTED" && a.manually_rejected, "raw", "raw",
                        {marker: {color: "#dc3545", size: 9, symbol: "x", line: {color: "#a71d2a", width: 1.5}}, showlegend: false}, "Rejected automatic peak (raw)");

                    if (showRej) {
                        customMarkerTrace(a => a.status === "REJECTED" && !a.manually_rejected, "amp", "amp",
                            {marker: {color: "#888888", size: 8, symbol: "circle-open"}}, "Excluded candidates (diagnostic)");
                        customMarkerTrace(a => a.status === "REJECTED" && !a.manually_rejected, "raw", "raw",
                            {marker: {color: "#888888", size: 8, symbol: "circle-open"}, showlegend: false}, "Excluded (raw)");
                    }
                }
                // Highlight currently selected peak for per-peak strictness
                if (selectedPeakInfo && selectedPeakInfo.segIdx === currentSegIdx && selectedPeakInfo.polKey === polKey && P.amp) {
                    const w = Math.max(1, Math.round(0.04 * seg.fs));
                    const selEvIdx = anns.findIndex(a => Math.abs(a.index - selectedPeakInfo.index) <= w);
                    if (selEvIdx >= 0) {
                        const selEv = anns[selEvIdx];
                        traces.push({
                            x: [selEv.time_s], y: [selEv.amp], customdata: [selEvIdx],
                            mode: "markers", name: "SELECTED peak",
                            marker: {color: "rgba(0,0,0,0)", size: 16, symbol: "circle", line: {color: "#ff9800", width: 2.5}},
                            yaxis: P.amp.ax, xaxis: "x", hoverinfo: "skip"
                        });
                    }
                }
            }

            let titleCountTxt = `— accepted contractions: ${nAcc}`;
            if (qc.isManualRejection) {
                titleCountTxt = `— [MANUALLY REJECTED: ${qc.manualReason}] — accepted contractions: 0`;
            } else if (isQCFailed) {
                titleCountTxt = `— [QC ${qc.status}: ${qc.reason}] — accepted contractions: 0`;
            }
            const layout = {
                title: `${seg.sheet_name} — raw polarity ${activePol > 0 ? "+1" : (activePol < 0 ? "-1" : "UNDETERMINED")} ` +
                       `${titleCountTxt} — strictness: ${currentRun}`,
                hovermode: "closest", height: Math.max(520, 260 * panels.length + 120),
                uirevision: currentSegIdx + "|" + currentRun + "|" + polKey + "|" + showRaw,
                xaxis: {title: "Time (s)", anchor: "y" + (panels.length === 1 ? "" : panels.length), domain: [0, 1]},
                legend: {orientation: "h", y: -0.15}
            };
            panels.forEach(q => {
                const o = {domain: q.domain, anchor: "x", title: q.title};
                if (q.key === "amp" && ampArr) {
                    let ymax = finiteMax(ampArr);
                    anns.forEach(a => { if (a.amp !== null && a.amp > ymax) ymax = a.amp; });
                    if (run.amplitude_threshold !== null && run.amplitude_threshold > ymax) ymax = run.amplitude_threshold;
                    o.range = [0, ymax * 1.12 || 1];
                    o.zeroline = true;
                }
                layout["yaxis" + (q.axis === 1 ? "" : q.axis)] = o;
            });
            if (panels.length === 0) { layout.height = 300; }

            Plotly.react("plot", traces, layout);
            const el = document.getElementById("plot");
            el.removeAllListeners && el.removeAllListeners("plotly_click");
            el.on("plotly_click", function (data) {
                if (isUndet || isQCFailed || qc.isManualRejection) return;
                const pt = data.points[0];
                const name = pt.data.name || "";
                const polMapKey = currentSegIdx + "|" + polKey;
                if (!manualPeaksByPol[polMapKey]) manualPeaksByPol[polMapKey] = {};
                if (!perPeakStrictness[polMapKey]) perPeakStrictness[polMapKey] = {};
                const manMap = manualPeaksByPol[polMapKey];
                const ppMap = perPeakStrictness[polMapKey];
                const w = Math.max(1, Math.round(0.04 * seg.fs));

                function deleteManualPeak(ev) {
                    const selIdx = ev.index;
                    Object.keys(ppMap).forEach(k => { if (Math.abs(parseInt(k) - selIdx) <= w) delete ppMap[k]; });
                    Object.keys(manMap).forEach(k => { if (Math.abs(parseInt(k) - selIdx) <= w) delete manMap[k]; });
                    Object.keys(annotations).forEach(annK => {
                        if (annK.startsWith(currentSegIdx + "|" + polKey + "|")) {
                            const list = annotations[annK];
                            for (let i = list.length - 1; i >= 0; i--) {
                                if (Math.abs(list[i].index - selIdx) <= w) {
                                    if (list[i].origin === "manual" || list[i].original_status === "MISSING") {
                                        list.splice(i, 1);
                                    } else {
                                        list[i].status = list[i].original_status;
                                        list[i].reason = list[i].original_reason || "";
                                        list[i].origin = "auto";
                                        list[i].manually_rejected = false;
                                    }
                                }
                            }
                        }
                    });
                    selectedPeakInfo = null;
                }

                function toggleEventState(evObj) {
                    if (evObj.origin === "manual") {
                        // Manually added BLUE peak clicked again -> completely delete it!
                        // NOTHING -> BLUE -> NOTHING (no red cross, decrements count, leaves no ghost)
                        deleteManualPeak(evObj);
                        return;
                    }

                    // Auto peak: GREEN -> RED CROSS -> GREEN
                    Object.keys(ppMap).forEach(k => { if (Math.abs(parseInt(k) - evObj.index) <= w) delete ppMap[k]; });
                    evObj.per_peak_strictness = "inherit";

                    if (evObj.status === "ACCEPTED") {
                        // Auto GREEN peak clicked -> turns into visible RED CROSS
                        evObj.status = "REJECTED";
                        evObj.origin = "auto";
                        evObj.manually_rejected = true;
                        evObj.reason = "manual_rejection";
                        manMap[evObj.index] = { index: evObj.index, time_s: evObj.time_s, amp: evObj.amp, raw: evObj.raw, action: "REJECTED" };
                    } else {
                        // Event is currently REJECTED (a RED CROSS) -> restore back to GREEN!
                        evObj.status = "ACCEPTED";
                        evObj.origin = "auto";
                        evObj.manually_rejected = false;
                        evObj.reason = evObj.original_reason || "";
                        Object.keys(manMap).forEach(k => { if (Math.abs(parseInt(k) - evObj.index) <= w) delete manMap[k]; });
                    }
                }

                if (pt.customdata !== undefined && pt.customdata !== null) {
                    const a = anns[pt.customdata];
                    if (!a) return;
                    if (a.origin === "manual") {
                        deleteManualPeak(a);
                        selectedPeakInfo = null;
                    } else {
                        toggleEventState(a);
                        selectedPeakInfo = { segIdx: currentSegIdx, polKey: polKey, index: a.index };
                    }
                } else if (name === "amplitude" && P.amp && ampArr) {
                    let c = Math.round((pt.x - seg.t0) * seg.fs);
                    let lo = Math.max(0, c - w), hi = Math.min(seg.length - 1, c + w), best = -1, bv = -Infinity;
                    for (let i = lo; i <= hi; i++) if (Number.isFinite(ampArr[i]) && ampArr[i] > bv) { bv = ampArr[i]; best = i; }
                    if (best < 0) return;
                    const existing = anns.find(a => Math.abs(a.index - best) <= w);
                    if (existing && (existing.status === "ACCEPTED" || existing.manually_rejected)) {
                        if (existing.origin === "manual") {
                            deleteManualPeak(existing);
                            selectedPeakInfo = null;
                        } else {
                            toggleEventState(existing);
                            selectedPeakInfo = { segIdx: currentSegIdx, polKey: polKey, index: existing.index };
                        }
                    } else {
                        // Unmarked trace location: add BLUE manually added peak
                        let targetEv = existing;
                        if (!targetEv) {
                            targetEv = {
                                index: best, time_s: seg.t0 + best / seg.fs, amp: ampArr[best], raw: baseSig.raw[best],
                                status: "ACCEPTED", original_status: "MISSING", original_reason: "",
                                reason: "", origin: "manual", manually_rejected: false, per_peak_strictness: "inherit"
                            };
                            anns.push(targetEv);
                        } else {
                            targetEv.status = "ACCEPTED";
                            targetEv.origin = "manual";
                            targetEv.manually_rejected = false;
                            targetEv.reason = "";
                        }
                        manMap[targetEv.index] = { index: targetEv.index, time_s: targetEv.time_s, amp: targetEv.amp, raw: targetEv.raw, action: "ADDED" };
                        selectedPeakInfo = { segIdx: currentSegIdx, polKey: polKey, index: targetEv.index };
                    }
                } else { return; }
                syncPeakState(currentSegIdx, currentRun, polKey);
                updateSummaryTable();
                populateSegmentSelector();
                renderPlot();
            });
            updateSummaryTable();
        }

        function exportPolarityOverrides() {
            const overrides = {};
            let count = 0;
            segments_data.forEach((seg, idx) => {
                const polInfo = getActivePolarity(idx);
                if (polInfo.isOverride && (polInfo.pol === 1 || polInfo.pol === -1)) {
                    overrides[String(seg.segment_number)] = polInfo.pol;
                    count++;
                }
            });

            const rejections = {};
            let nRejections = 0;
            segments_data.forEach((seg, idx) => {
                const r = userSegmentRejection[idx];
                if (r && r.rejected) {
                    rejections[String(seg.segment_number)] = {
                        rejected: true,
                        reason: r.reason || "Other / researcher decision"
                    };
                    nRejections++;
                }
            });

            const perPeakOverrides = {};
            let nPerPeak = 0;
            Object.entries(perPeakStrictness).forEach(([polMapKey, mapObj]) => {
                const [sIdxStr, pKey] = polMapKey.split("|");
                const seg = segments_data[parseInt(sIdxStr)];
                if (!seg) return;
                Object.entries(mapObj).forEach(([sampleIdxStr, preset]) => {
                    if (preset && preset !== "inherit") {
                        const sNum = String(seg.segment_number);
                        if (!perPeakOverrides[sNum]) perPeakOverrides[sNum] = [];
                        const sIdx = parseInt(sampleIdxStr);
                        perPeakOverrides[sNum].push({
                            polarity: parseInt(pKey),
                            sample_index: sIdx,
                            time_s: Number((seg.t0 + sIdx / seg.fs).toFixed(4)),
                            per_peak_strictness: preset
                        });
                        nPerPeak++;
                    }
                });
            });

            const manualEdits = {};
            let nManualEdits = 0;
            Object.entries(manualPeaksByPol).forEach(([polMapKey, mapObj]) => {
                const [sIdxStr, pKey] = polMapKey.split("|");
                const seg = segments_data[parseInt(sIdxStr)];
                if (!seg) return;
                Object.values(mapObj).forEach(mRec => {
                    const sNum = String(seg.segment_number);
                    if (!manualEdits[sNum]) manualEdits[sNum] = [];
                    manualEdits[sNum].push({
                        polarity: parseInt(pKey),
                        sample_index: mRec.index,
                        time_s: Number(mRec.time_s.toFixed(4)),
                        amplitude: mRec.amp,
                        action: mRec.action,
                        origin: (mRec.action === "ADDED" || mRec.action === "ADDED_REJECTED") ? "manual" : "auto"
                    });
                    nManualEdits++;
                });
            });

            if (count === 0 && nRejections === 0 && nPerPeak === 0 && nManualEdits === 0) {
                alert("No manual polarity overrides, segment rejections, per-peak strictness overrides, or manual peak edits have been specified yet.");
                return;
            }
            const payload = {
                "workbook_name": source_workbook,
                "workbook_sha256": source_sha256,
                "export_timestamp": new Date().toISOString(),
                "total_overrides": count,
                "total_segment_rejections": nRejections,
                "total_per_peak_strictness_overrides": nPerPeak,
                "total_manual_peak_edits": nManualEdits,
                "overrides": overrides,
                "segment_rejections": rejections,
                "per_peak_strictness": perPeakOverrides,
                "manual_peak_edits": manualEdits
            };
            const jsonStr = JSON.stringify(payload, null, 2);
            const blob = new Blob([jsonStr], {type: "application/json;charset=utf-8;"});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", `${source_workbook}.polarity_overrides.json`);
            link.style.visibility = "hidden";
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
        }

        function saveAnnotations() {
            if (!Object.values(segmentReviewed).some(v => v)) {
                alert("No segments have been marked as Reviewed. Cannot export.");
                return;
            }
            const annotatorId = prompt("Enter annotator initials (e.g., JD):", "XX");
            if (!annotatorId) return;
            let csv = "source_workbook,sheet_name,sample_id,time_s,confidence,annotator_id,amplitude,origin,strictness,per_peak_strictness,min_rel_amplitude,min_rel_recovery,min_snr,polarity,qc_status,qc_reason\n";
            segments_data.forEach((seg, idx) => {
                if (!segmentReviewed[idx]) return;
                const manualRej = userSegmentRejection[idx];
                if (manualRej && manualRej.rejected) return;
                const polInfo = getActivePolarity(idx);
                if (polInfo.pol !== 1 && polInfo.pol !== -1) return;
                const polKey = String(polInfo.pol);
                const qc = getActiveQC(idx, polInfo.pol, currentRun);
                if (qc.status !== "PASS") return;
                getAnn(idx, currentRun, polKey).forEach(a => {
                    if (a.status !== "ACCEPTED") return;
                    const effStrictness = (a.per_peak_strictness && a.per_peak_strictness !== "inherit") ? a.per_peak_strictness : currentRun;
                    const sp = (seg.orientations[polKey].runs[effStrictness] || seg.orientations[polKey].runs[currentRun]).params;
                    const ppStrict = a.per_peak_strictness || "inherit";
                    csv += `"${source_workbook}","${seg.sheet_name}","${seg.sample_id}",${a.time_s},1,"${annotatorId}",` +
                           `${a.amp === null ? "" : a.amp},${a.origin},${effStrictness},${ppStrict},${sp.min_rel_amplitude},${sp.min_rel_recovery},${sp.min_snr},${polInfo.pol},` +
                           `"${qc.status}","${qc.reason}"\n`;
                });
            });
            const blob = new Blob([csv], {type: "text/csv;charset=utf-8;"});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", `contraction_annotations_${source_workbook}.csv`);
            link.style.visibility = "hidden";
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
        }

        try {
            populateSelectors();
            (function () {
                const q = new URLSearchParams(window.location.search);
                if (q.has('seg')) { currentSegIdx = Math.min(segments_data.length - 1, Math.max(0, parseInt(q.get('seg')))); document.getElementById('segment_select').value = currentSegIdx; }
                if (q.has('run') && segments_data[0] && segments_data[0].orientations && segments_data[0].orientations["-1"] && segments_data[0].orientations["-1"].runs && segments_data[0].orientations["-1"].runs[q.get('run')]) {
                    currentRun = q.get('run');
                    document.getElementById('run_select').value = currentRun;
                }
                if (q.has('pol')) {
                    const p = parseInt(q.get('pol'));
                    if (p === 1 || p === -1) userPolarityOverride[currentSegIdx] = p;
                }
                if (q.get('rej') === '1') document.getElementById('chk_rejected').checked = true;
                if (q.get('raw') === '1') document.getElementById('chk_raw').checked = true;
            })();
            renderPlot();
        } catch (err) {
            console.error("Initialization error:", err);
            const eb = document.getElementById("client_error_banner");
            if (eb) {
                eb.style.display = "block";
                eb.innerHTML = `⚠️ <b>Error initializing review report:</b> ${err.message}<br><small>${err.stack || ""}</small>`;
            }
        }
    </script>
</body>
</html>
"""


def generate_html(segments_data, source_workbook, out_path, default_run="default", workbook_path=None, config=None, segment_rejections=None):
    if not segments_data:
        raise ValueError(f"Cannot generate HTML review report for '{source_workbook}': segments_data is empty (zero segments extracted).")
    runs_available = list(segments_data[0]["runs"].keys()) if segments_data and "runs" in segments_data[0] else []
    if default_run not in runs_available and runs_available:
        default_run = runs_available[0]

    sha256 = ""
    if workbook_path and Path(workbook_path).exists():
        try:
            sha256 = compute_file_sha256(workbook_path)
        except Exception:
            sha256 = ""

    use_ca = bool(getattr(config, "v2_use_calcium_polarity", False)) if config else False
    polarity_mode_str = "Calcium-Assisted (ECC Timing Enabled)" if use_ca else "PixelCorrelation-Only (Default)"

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(segments_data, allow_nan=False))
    html = html.replace("__SOURCE__", str(source_workbook).replace('"', "'"))
    html = html.replace("__SHA256__", str(sha256))
    html = html.replace("__DEFAULT_RUN__", default_run)
    html = html.replace("__POLARITY_MODE__", polarity_mode_str)
    html = html.replace("__SEGMENT_REJECTIONS__", json.dumps(segment_rejections or {}, allow_nan=False))

    # Embed Plotly JS for strictly offline use
    plotly_js = None
    import urllib.request
    is_mocked = hasattr(urllib.request.urlopen, "mock_calls") or "Mock" in type(urllib.request.urlopen).__name__
    if not is_mocked:
        local_candidates = [
            Path("plotly.min.js"),
            Path(__file__).parent / "plotly.min.js",
            Path("docs") / "plotly.min.js",
        ]
        for cand in local_candidates:
            if cand.exists():
                try:
                    c = cand.read_text(encoding="utf-8")
                    if len(c) > 10000:
                        plotly_js = c
                        break
                except Exception:
                    pass

    if plotly_js is None:
        try:
            print("Downloading Plotly.js to embed it for offline use...")
            with urllib.request.urlopen("https://cdn.plot.ly/plotly-2.27.0.min.js", timeout=10) as response:
                plotly_js = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to load Plotly.js ({e}). Offline generation requires local plotly.min.js or network access.") from e

    html = html.replace("__PLOTLY_JS__", plotly_js)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReview report generated successfully: {out_path}")


def build_params_by_name(args) -> dict:
    runs = dict(STRICTNESS_PRESETS)
    from pixel_counter.v2_detector import (
        amplitude_strictness_to_amplitude,
        separation_strictness_to_recovery,
    )

    sep_strict = getattr(args, "separation_strictness", None)
    amp_strict = getattr(args, "amplitude_strictness", None)
    def_params = STRICTNESS_PRESETS["default"]

    rec = None
    if sep_strict is not None:
        rec = separation_strictness_to_recovery(sep_strict)
    elif getattr(args, "min_rel_recovery", None) is not None:
        rec = args.min_rel_recovery
    else:
        rec = def_params.min_rel_recovery

    amp = None
    if amp_strict is not None:
        amp = amplitude_strictness_to_amplitude(amp_strict)
    elif getattr(args, "min_rel_amplitude", None) is not None:
        amp = args.min_rel_amplitude
    else:
        amp = def_params.min_rel_amplitude

    snr = args.min_snr if getattr(args, "min_snr", None) is not None else def_params.min_snr

    eff_sep_strict = float(sep_strict) if sep_strict is not None else 1.0
    eff_amp_strict = float(amp_strict) if amp_strict is not None else 1.0

    custom = DetectorParams(
        min_rel_amplitude=amp,
        min_rel_recovery=rec,
        min_snr=snr,
        separation_strictness=eff_sep_strict,
        amplitude_strictness=eff_amp_strict,
    ).validate()

    is_custom = (
        sep_strict is not None
        or amp_strict is not None
        or getattr(args, "min_rel_amplitude", None) is not None
        or getattr(args, "min_rel_recovery", None) is not None
        or getattr(args, "min_snr", None) is not None
    ) and (
        abs(custom.min_rel_amplitude - def_params.min_rel_amplitude) > 1e-6
        or abs(custom.min_rel_recovery - def_params.min_rel_recovery) > 1e-6
        or abs(custom.min_snr - def_params.min_snr) > 1e-6
    )

    if is_custom:
        # the custom run is listed first and is the selected one
        runs = {"custom": custom, **runs}
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cytocypher v2 contraction review utility")
    parser.add_argument("input_file", help="Path to input Excel workbook")
    parser.add_argument("--output", default="v2_validation_review.html", help="Path to output HTML report")
    parser.add_argument("--separation-strictness", type=float, default=None,
                        help="Continuous separation strictness in [0.0, 2.0] (default 1.0 -> rho=0.50)")
    parser.add_argument("--amplitude-strictness", type=float, default=None,
                        help="Continuous amplitude strictness in [0.0, 2.0] (default 1.0 -> alpha=0.50)")
    parser.add_argument("--min-rel-amplitude", type=float, default=None,
                        help="alpha in (0,1): a contraction must be >= alpha x typical major amplitude (default 0.50)")
    parser.add_argument("--min-rel-recovery", type=float, default=None,
                        help="rho in (0,1): separate cycles must recover >= rho of the lower peak height (default 0.50)")
    parser.add_argument("--min-snr", type=float, default=None,
                        help="advanced: absolute floor, height >= min_snr x noise sigma (default 8)")
    parser.add_argument("--polarity", choices=["auto", "positive", "negative"], default="auto",
                        help="auto (default) = decided once from the raw waveform; positive/negative = global raw-polarity override")
    parser.add_argument("--use-calcium-polarity", action="store_true", default=False,
                        help="Enable optional RatioMetricCalcium-assisted physiological polarity determination (default: False, PixelCorrelation-only)")
    parser.add_argument("--segments", type=int, nargs="*", default=None, help="restrict to these segment numbers")
    parser.add_argument("--overrides", type=str, default=None,
                        help="Path to JSON file containing polarity overrides for this workbook")
    parser.add_argument("--override-polarity", type=str, default=None,
                        help="Comma-separated segment polarity overrides, e.g. '17=-1,18=-1,45=1'")
    parser.add_argument("--save-overrides", type=str, default=None,
                        help="Path to save active polarity overrides to JSON")
    parser.add_argument("--inspect-polarity", action="store_true", default=False,
                        help="Run Stage 1 polarity inspection and print summary table, then exit")
    args = parser.parse_args(argv)

    try:
        overrides = {}
        segment_rejections = {}
        ov_path = Path(args.overrides) if args.overrides else (default_overrides_path(args.input_file) if args.input_file else None)
        if ov_path and ov_path.exists():
            try:
                overrides.update(load_polarity_overrides(args.input_file, ov_path, strict_hash=False))
                segment_rejections.update(load_segment_rejections(args.input_file, ov_path))
                print(f"Loaded sidecar: {len(overrides)} polarity overrides, {len(segment_rejections)} segment rejections from {ov_path.name}")
            except Exception as e:
                print(f"Warning: Could not load overrides from {ov_path.name}: {e}")
        if args.override_polarity:
            overrides.update(parse_polarity_override_str(args.override_polarity))
        if args.save_overrides:
            saved_p = save_polarity_overrides(args.input_file, overrides, args.save_overrides, segment_rejections=segment_rejections)
            print(f"Saved polarity overrides and rejections to {saved_p}")

        params_by_name = build_params_by_name(args)
        config = replace(
            BeatCounterConfig(),
            v2_polarity_mode=args.polarity,
            v2_use_calcium_polarity=args.use_calcium_polarity,
            v2_separation_strictness=args.separation_strictness if args.separation_strictness is not None else 1.0,
            v2_amplitude_strictness=args.amplitude_strictness if args.amplitude_strictness is not None else 1.0,
        )

        if args.inspect_polarity:
            process_workbook(args.input_file, params_by_name, config, args.segments,
                             polarity_overrides=overrides, inspect_only=True)
            return

        segments_data, source_name = process_workbook(
            args.input_file,
            params_by_name,
            config,
            args.segments,
            polarity_overrides=overrides,
        )
        generate_html(
            segments_data,
            source_name,
            args.output,
            default_run="custom" if "custom" in params_by_name else "default",
            workbook_path=args.input_file,
            config=config,
            segment_rejections=segment_rejections,
        )

        print("\nINSTRUCTIONS:")
        print(f"1. Open '{args.output}' in your web browser.")
        print("2. Review segments with the dropdown; switch strictness presets; tick 'Show rejected candidates' for diagnostics.")
        print("3. Click a marker to flip ACCEPTED <-> REJECTED; click the amplitude trace to add a missed contraction.")
        print("4. Use the Polarity dropdown to inspect or override polarity (+1 or -1) interactively.")
        print("5. Click 'Export polarity overrides (JSON)' to save your orientation decisions for automated re-runs.")
        print("6. Click 'Save annotations' to export a CSV of verified contractions.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

