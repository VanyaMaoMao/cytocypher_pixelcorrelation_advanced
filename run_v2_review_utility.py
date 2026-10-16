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
    load_polarity_overrides,
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
    return {
        "index": [int(i) for i in ev["index"]],
        "time_s": [float(t) for t in ev["time_abs_s"]],
        "amplitude": [_num(a) for a in ev["amplitude"]],
        "raw_value": [_num(a) for a in ev["raw_value"]] if "raw_value" in ev else [None] * len(ev),
        "status": [str(s) for s in ev["status"]],
        "reason": [str(s) for s in ev["reason"]],
    }


def _run_payload(v2_res):
    md = v2_res["metadata"]
    return {
        "params": md.get("detector_params", {}),
        "qc_status": md.get("qc_status", "UNKNOWN"),
        "qc_reason": md.get("qc_reason", ""),
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
                if len(sig_o) == len(sig_raw) and abs(t0_o - t0) < 0.5 / fs:
                    return sig_o
                tt = t0_o + np.arange(len(sig_o)) / fs_o
                return np.interp(t0 + np.arange(len(sig_raw)) / fs, tt, sig_o, left=np.nan, right=np.nan)
            except Exception as e:
                return None

        sig_cal = load_aux_channel("RatioMetricCalcium")
        sig_int = load_aux_channel("PixelIntensity")
        meta["sig_cal"] = sig_cal
        meta["sig_int"] = sig_int

        cal_for_polarity = sig_cal if use_ca else None
        auto_pol = determine_raw_polarity(sig_raw, fs, sig_cal=cal_for_polarity, fs_cal=fs, t0_raw=t0, t0_cal=t0)
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
            "sig_cal": sig_cal,
            "sig_int": sig_int,
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
        first = None
        for name, p in params_by_name.items():
            res = analyze_v2_pipeline(sig_raw, fs, config, t0=t0, meta=meta, params=p,
                                      polarity_override=eff if eff in (-1, 1) else None)
            runs[name] = _run_payload(res)
            if first is None:
                first = res
        wf = first["waveform"]
        pol = first["polarity_result"]
        has_amp = bool(np.isfinite(wf["amplitude"]).any())

        # Precompute runs for both orientations (+1 and -1) for interactive offline review
        orientations = {}
        for fixed_pol, fixed_key in [(1, "1"), (-1, "-1")]:
            fixed_runs = {}
            first_fixed = None
            for name, p in params_by_name.items():
                r_fix = analyze_v2_pipeline(sig_raw, fs, config, t0=t0, meta=meta, params=p,
                                            polarity_override=fixed_pol)
                fixed_runs[name] = _run_payload(r_fix)
                if first_fixed is None:
                    first_fixed = r_fix
            wf_fix = first_fixed["waveform"]
            orientations[fixed_key] = {
                "amplitude_b64": b64_encode_array(wf_fix["amplitude"]),
                "baseline_raw_b64": b64_encode_array(wf_fix["baseline_raw"]),
                "runs": fixed_runs,
                "qc_status": first_fixed["metadata"].get("qc_status", "PASS"),
                "qc_reason": first_fixed["metadata"].get("qc_reason", ""),
                "n_accepted": int(len(first_fixed["candidates"]["time_s"])),
            }

        sig_int = item.get("sig_int")
        sig_cal = item.get("sig_cal")

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
            "sig_cal_b64": b64_encode_array(sig_cal) if sig_cal is not None and len(sig_cal) else None,
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
        <label><input type="checkbox" id="chk_raw"> Show raw PixelCorrelation</label>
        <label style="background:#eee; padding:5px; border-radius:3px;">
            <input type="checkbox" id="chk_reviewed"> Mark segment as reviewed</label>
        <button id="btn_save">Save annotations (CSV)</button>
        <button id="btn_export_overrides" class="btn-primary">Export polarity overrides (JSON)</button>
    </div>
    <div style="margin-bottom:8px;">
        <b>Review:</b> click a green (ACCEPTED) or red (REJECTED) marker to flip its decision;
        click on the amplitude trace to add a missed contraction (snapped to the local maximum).
        Use the <b>Polarity</b> dropdown to inspect or override physiological orientation (+1 or -1) without re-running Python code.
    </div>

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
        const segments_data = __DATA__;
        const source_workbook = "__SOURCE__";
        const source_sha256 = "__SHA256__";
        const DEFAULT_RUN = "__DEFAULT_RUN__";

        let currentSegIdx = 0;
        let currentRun = DEFAULT_RUN;
        const userPolarityOverride = {}; // key: segIdx -> 1, -1, or "auto"
        const annotations = {};          // key `${seg}|${polKey}|${run}` -> array of event records
        const segmentReviewed = {};
        const decodedSignals = {};
        const decodedOrientations = {};

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

        function getAnn(segIdx, run, polKey) {
            const key = segIdx + "|" + polKey + "|" + run;
            if (!annotations[key]) {
                const seg = segments_data[segIdx];
                let ev = null;
                if (polKey === "1" || polKey === "-1") {
                    if (seg.orientations && seg.orientations[polKey]) {
                        ev = seg.orientations[polKey].runs[run].events;
                    }
                }
                if (!ev) {
                    ev = seg.runs[run] ? seg.runs[run].events : {index: [], time_s: [], amplitude: [], raw_value: [], status: [], reason: []};
                }
                annotations[key] = ev.index.map((idx, i) => ({
                    index: idx, time_s: ev.time_s[i], amp: ev.amplitude[i], raw: ev.raw_value[i],
                    status: ev.status[i], original_status: ev.status[i], reason: ev.reason[i], origin: "auto"
                }));
            }
            return annotations[key];
        }

        function updateNavButtons() {
            const btnPrev = document.getElementById("btn_prev");
            const btnNext = document.getElementById("btn_next");
            if (btnPrev) btnPrev.disabled = (currentSegIdx <= 0);
            if (btnNext) btnNext.disabled = (currentSegIdx >= segments_data.length - 1);
        }

        function countsFor(idx, pol, runKey) {
            runKey = runKey || currentRun;
            if (pol !== 1 && pol !== -1) {
                return {auto: 0, added: 0, rej: 0, final: 0};
            }
            const anns = getAnn(idx, runKey, String(pol));
            let auto = 0, added = 0, rej = 0, final = 0;
            anns.forEach(a => {
                if (a.original_status === "ACCEPTED" && a.origin === "auto") auto++;
                if (a.origin === "manual" && a.status === "ACCEPTED") added++;
                if (a.origin === "auto" && a.original_status === "ACCEPTED" && a.status === "REJECTED") rej++;
                if (a.status === "ACCEPTED") final++;
            });
            return {auto, added, rej, final};
        }

        function getActiveQC(segIdx, activePol, runKey) {
            const seg = segments_data[segIdx];
            runKey = runKey || currentRun;
            if (activePol !== 1 && activePol !== -1) {
                return {
                    status: "INPUT_REQUIRED",
                    reason: seg.auto_reason || "polarity_undetermined",
                    label: "POLARITY INPUT REQUIRED",
                    isInputRequired: true,
                    isZeroContractions: false,
                    nAccepted: 0
                };
            }
            const polKey = String(activePol);
            const oData = seg.orientations ? seg.orientations[polKey] : null;
            let baseStatus = "PASS";
            let baseReason = "clean_signal";
            if (oData) {
                const r = (oData.runs && oData.runs[runKey]) ? oData.runs[runKey] : null;
                baseStatus = (r && r.qc_status) ? r.qc_status : oData.qc_status;
                baseReason = (r && r.qc_reason) ? r.qc_reason : oData.qc_reason;
            } else {
                baseStatus = seg.qc_status;
                baseReason = seg.qc_reason;
            }

            if (baseStatus !== "PASS") {
                return {
                    status: baseStatus,
                    reason: baseReason,
                    label: `QC: ${baseStatus} (${baseReason})`,
                    isInputRequired: false,
                    isZeroContractions: false,
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
                    nAccepted: 0
                };
            }

            return {
                status: "PASS",
                reason: baseReason || "clean_signal",
                label: `QC: PASS (${nAcc} contractions)`,
                isInputRequired: false,
                isZeroContractions: false,
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
                let polDesc = "";
                if (polInfo.pol === 0) {
                    polDesc = "POLARITY INPUT REQUIRED";
                } else if (polInfo.isOverride) {
                    polDesc = `override: ${polInfo.pol > 0 ? '+1' : '-1'}`;
                } else {
                    polDesc = `auto: ${polInfo.pol > 0 ? '+1' : '-1'}`;
                }
                const qc = getActiveQC(idx, polInfo.pol);
                let qcShort = "";
                if (qc.isInputRequired) {
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

            const polInfo = getActivePolarity(currentSegIdx);
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
            document.getElementById("chk_raw").addEventListener("change", renderPlot);
            document.getElementById("source_name").innerText = source_workbook;
            document.getElementById("btn_save").addEventListener("click", saveAnnotations);
            document.getElementById("btn_export_overrides").addEventListener("click", exportPolarityOverrides);
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
                    polBadge = '<span style="color:#d9534f; font-weight:bold;">POLARITY INPUT REQUIRED</span>';
                } else if (polInfo.pol > 0) {
                    polBadge = `+1 (up) ${polInfo.isOverride ? '<span style="color:#0275d8; font-weight:bold;">[override]</span>' : '<span style="color:#5cb85c;">[auto]</span>'}`;
                } else {
                    polBadge = `-1 (down) ${polInfo.isOverride ? '<span style="color:#0275d8; font-weight:bold;">[override]</span>' : '<span style="color:#5cb85c;">[auto]</span>'}`;
                }

                let qcHtml = "";
                if (qc.isInputRequired) {
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

            const baseSig = getBaseSignals(currentSegIdx);
            const times = timesFor(seg);
            const qc = getActiveQC(currentSegIdx, activePol);

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
            const showRaw = document.getElementById("chk_raw").checked || !hasAmp;

            let polTxt = "";
            if (isUndet) {
                polTxt = `<span style="color:#d9534f; font-weight:bold;">POLARITY INPUT REQUIRED</span> [${seg.auto_reason}]`;
            } else {
                const dir = activePol > 0 ? "+1 (contractions upward in raw)" : "-1 (contractions downward in raw)";
                const tag = polInfo.isOverride ? "[manual override]" : `[automatic: ${seg.polarity_method}]`;
                polTxt = `${dir} ${tag}`;
            }

            let qcHeaderTxt = "";
            if (qc.isInputRequired) {
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
            if (qc.isInputRequired) {
                fb.className = "polarity-banner";
                fb.style.display = "block";
                fb.innerHTML = `⚠️ <b>POLARITY INPUT REQUIRED:</b> Automatic physiological polarity could not be determined for this recording (${seg.auto_reason}).<br>` +
                    `Select <b>Raw +1</b> or <b>Raw -1</b> in the Polarity dropdown above to inspect and evaluate contraction events.`;
            } else if (qc.status !== "PASS") {
                fb.className = "fail";
                fb.style.display = "block";
                fb.innerText = `Segment technical failure (${qc.status}): ${qc.reason}.`;
            } else if (qc.isZeroContractions) {
                fb.className = "params";
                fb.style.display = "block";
                fb.innerHTML = `ℹ️ <b>No Contractions Detected:</b> Signal processing succeeded (QC PASS), but zero contractions met the amplitude threshold and recovery criteria under strictness [${currentRun}].`;
            } else {
                fb.style.display = "none";
            }

            const p = run.params || {};
            document.getElementById("param_box").innerText =
                `Active detector parameters [${currentRun}]: min_rel_amplitude=${p.min_rel_amplitude || "0.50"}  ` +
                `min_rel_recovery=${p.min_rel_recovery || "0.50"}  min_snr=${p.min_snr || "8"}  |  ` +
                `reference amplitude=${run.reference_amplitude === null ? "n/a" : run.reference_amplitude.toFixed(4)}  ` +
                `amplitude threshold=${run.amplitude_threshold === null ? "n/a" : run.amplitude_threshold.toFixed(4)}  ` +
                `noise sigma=${run.noise_sigma === null ? "n/a" : run.noise_sigma.toExponential(2)}`;

            const anns = getAnn(currentSegIdx, currentRun, polKey);
            const nAcc = anns.filter(a => a.status === "ACCEPTED").length;

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
            if (P.int && baseSig.inten) traces.push({x: times, y: baseSig.inten, mode: "lines", name: "PixelIntensity", line: {color: "green", width: 1}, yaxis: P.int.ax, xaxis: "x"});
            if (P.cal && baseSig.cal) traces.push({x: times, y: baseSig.cal, mode: "lines", name: "RatioMetricCalcium", line: {color: "firebrick", width: 1}, yaxis: P.cal.ax, xaxis: "x"});

            function markerTrace(status, panelKey, yField, style, name) {
                const pk = P[panelKey]; if (!pk) return;
                const xs = [], ys = [], cd = [], tx = [];
                anns.forEach((a, i) => {
                    if (a.status !== status) return;
                    const y = a[yField]; if (y === null || y === undefined) return;
                    xs.push(a.time_s); ys.push(y); cd.push(i);
                    tx.push(`t=${a.time_s.toFixed(3)} s<br>A=${a.amp === null ? "n/a" : a.amp.toFixed(4)}<br>` +
                            `${a.origin === "manual" ? "manual" : (a.original_status + (a.reason ? ": " + a.reason : ""))}`);
                });
                traces.push(Object.assign({x: xs, y: ys, customdata: cd, text: tx, hoverinfo: "text", mode: "markers", name: name,
                                           yaxis: pk.ax, xaxis: "x", type: "scatter"}, style));
            }

            if (!isUndet && hasAmp) {
                markerTrace("ACCEPTED", "amp", "amp", {marker: {color: "green", size: 10, symbol: "circle"}}, `ACCEPTED (${nAcc})`);
                markerTrace("ACCEPTED", "raw", "raw", {marker: {color: "green", size: 9, symbol: "circle"}, showlegend: false}, "ACCEPTED (raw)");
                if (showRej) {
                    markerTrace("REJECTED", "amp", "amp", {marker: {color: "red", size: 8, symbol: "x"}}, "REJECTED candidates");
                    markerTrace("REJECTED", "raw", "raw", {marker: {color: "red", size: 8, symbol: "x"}, showlegend: false}, "REJECTED (raw)");
                }
            }

            const layout = {
                title: `${seg.sheet_name} — raw polarity ${activePol > 0 ? "+1" : (activePol < 0 ? "-1" : "UNDETERMINED")} ` +
                       `— accepted contractions: ${nAcc} — strictness: ${currentRun}`,
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
                if (isUndet) return;
                const pt = data.points[0];
                const name = pt.data.name || "";
                if (pt.customdata !== undefined && pt.customdata !== null) {
                    const a = anns[pt.customdata];
                    a.status = (a.status === "ACCEPTED") ? "REJECTED" : "ACCEPTED";
                } else if (name === "amplitude" && P.amp && ampArr) {
                    const w = Math.round(0.04 * seg.fs);
                    let c = Math.round((pt.x - seg.t0) * seg.fs);
                    let lo = Math.max(0, c - w), hi = Math.min(seg.length - 1, c + w), best = -1, bv = -Infinity;
                    for (let i = lo; i <= hi; i++) if (Number.isFinite(ampArr[i]) && ampArr[i] > bv) { bv = ampArr[i]; best = i; }
                    if (best < 0) return;
                    const existing = anns.find(a => Math.abs(a.index - best) <= w);
                    if (existing) { existing.status = existing.status === "ACCEPTED" ? "REJECTED" : "ACCEPTED"; }
                    else anns.push({index: best, time_s: seg.t0 + best / seg.fs, amp: ampArr[best], raw: baseSig.raw[best],
                                    status: "ACCEPTED", original_status: "MISSING", reason: "", origin: "manual"});
                } else { return; }
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
            if (count === 0) {
                alert("No manual polarity overrides have been specified yet.\n\nSelect Raw +1 or Raw -1 from the Polarity dropdown for any segment before exporting.");
                return;
            }
            const payload = {
                "workbook_name": source_workbook,
                "workbook_sha256": source_sha256,
                "export_timestamp": new Date().toISOString(),
                "total_overrides": count,
                "overrides": overrides
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
            let csv = "source_workbook,sheet_name,sample_id,time_s,confidence,annotator_id,amplitude,origin,strictness,min_rel_amplitude,min_rel_recovery,min_snr,polarity,qc_status,qc_reason\n";
            segments_data.forEach((seg, idx) => {
                if (!segmentReviewed[idx]) return;
                const polInfo = getActivePolarity(idx);
                if (polInfo.pol !== 1 && polInfo.pol !== -1) return;
                const polKey = String(polInfo.pol);
                const sp = seg.orientations[polKey].runs[currentRun].params;
                const qc = getActiveQC(idx, polInfo.pol, currentRun);
                getAnn(idx, currentRun, polKey).forEach(a => {
                    if (a.status !== "ACCEPTED") return;
                    csv += `"${source_workbook}","${seg.sheet_name}","${seg.sample_id}",${a.time_s},1,"${annotatorId}",` +
                           `${a.amp === null ? "" : a.amp},${a.origin},${currentRun},${sp.min_rel_amplitude},${sp.min_rel_recovery},${sp.min_snr},${polInfo.pol},` +
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

        populateSelectors();
        (function () {
            const q = new URLSearchParams(window.location.search);
            if (q.has('seg')) { currentSegIdx = Math.min(segments_data.length - 1, Math.max(0, parseInt(q.get('seg')))); document.getElementById('segment_select').value = currentSegIdx; }
            if (q.has('run') && segments_data[0].orientations["-1"].runs[q.get('run')]) { currentRun = q.get('run'); document.getElementById('run_select').value = currentRun; }
            if (q.has('pol')) {
                const p = parseInt(q.get('pol'));
                if (p === 1 || p === -1) userPolarityOverride[currentSegIdx] = p;
            }
            if (q.get('rej') === '1') document.getElementById('chk_rejected').checked = true;
            if (q.get('raw') === '1') document.getElementById('chk_raw').checked = true;
        })();
        renderPlot();
    </script>
</body>
</html>
"""


def generate_html(segments_data, source_workbook, out_path, default_run="default", workbook_path=None, config=None):
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
    custom = DetectorParams(
        min_rel_amplitude=args.min_rel_amplitude if args.min_rel_amplitude is not None else STRICTNESS_PRESETS["default"].min_rel_amplitude,
        min_rel_recovery=args.min_rel_recovery if args.min_rel_recovery is not None else STRICTNESS_PRESETS["default"].min_rel_recovery,
        min_snr=args.min_snr if args.min_snr is not None else STRICTNESS_PRESETS["default"].min_snr,
    ).validate()
    if custom != STRICTNESS_PRESETS["default"]:
        # the custom run is listed first and is the selected one
        runs = {"custom": custom, **runs}
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cytocypher v2 contraction review utility")
    parser.add_argument("input_file", help="Path to input Excel workbook")
    parser.add_argument("--output", default="v2_validation_review.html", help="Path to output HTML report")
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
        if args.overrides:
            overrides.update(load_polarity_overrides(args.input_file, args.overrides, strict_hash=False))
        if args.override_polarity:
            overrides.update(parse_polarity_override_str(args.override_polarity))
        if args.save_overrides:
            saved_p = save_polarity_overrides(args.input_file, overrides, args.save_overrides)
            print(f"Saved polarity overrides to {saved_p}")

        params_by_name = build_params_by_name(args)
        config = replace(
            BeatCounterConfig(),
            v2_polarity_mode=args.polarity,
            v2_use_calcium_polarity=args.use_calcium_polarity,
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

