import argparse
import sys
import json
import base64
from pathlib import Path

import numpy as np
import pandas as pd

from pixel_counter.config import BeatCounterConfig
from pixel_counter.io_utils import load_cytocypher_excel, extract_sample_id_from_segment_sheet
from pixel_counter.preprocessing import build_concatenated_signal
from pixel_counter.analysis import analyze_v2_pipeline


def b64_encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 for embedding in HTML"""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("utf-8")


def process_workbook(file_path: str):
    print(f"Loading {file_path}...")
    xl = pd.ExcelFile(file_path)

    # Discover PixelCorrelation sheets
    corr_sheets = [s for s in xl.sheet_names if "PixelCorrelation" in s and "Segment" in s]
    # Sort them by segment number if possible
    def get_seg_num(s):
        try:
            return int(s.split("Segment")[-1].strip())
        except ValueError:
            return 9999
    corr_sheets.sort(key=get_seg_num)

    segments_data = []

    config = BeatCounterConfig()

    for i, corr_sheet in enumerate(corr_sheets):
        seg_num = get_seg_num(corr_sheet)
        print(f"Processing segment {seg_num} ({i+1}/{len(corr_sheets)}): {corr_sheet}")

        # 1. PixelCorrelation
        df_corr, y_cols_corr, fs, t0 = load_cytocypher_excel(file_path, sheet_name=corr_sheet)
        sample_id = extract_sample_id_from_segment_sheet(file_path, corr_sheet)
        sig_corr, seg_meta, meta = build_concatenated_signal(df_corr, y_cols_corr, fs, config)

        # Run v2 pipeline
        v2_res = analyze_v2_pipeline(sig_corr, fs, config)

        candidates = v2_res["candidates"]

        # 2. PixelIntensity (optional)
        int_sheet = corr_sheet.replace("PixelCorrelation", "PixelIntensity")
        sig_int = []
        if int_sheet in xl.sheet_names:
            try:
                df_int, y_cols_int, _, _ = load_cytocypher_excel(file_path, sheet_name=int_sheet)
                sig_int, _, _ = build_concatenated_signal(df_int, y_cols_int, fs, config)
            except Exception as e:
                print(f"  Warning: could not process {int_sheet}: {e}")

        # 3. RatioMetricCalcium (optional)
        cal_sheet = corr_sheet.replace("PixelCorrelation", "RatioMetricCalcium")
        sig_cal = []
        if cal_sheet in xl.sheet_names:
            try:
                df_cal, y_cols_cal, _, _ = load_cytocypher_excel(file_path, sheet_name=cal_sheet)
                sig_cal, _, _ = build_concatenated_signal(df_cal, y_cols_cal, fs, config)
            except Exception as e:
                print(f"  Warning: could not process {cal_sheet}: {e}")

        # We need absolute times for the x-axis
        times = t0 + np.arange(len(sig_corr)) / fs

        seg_data = {
            "sheet_name": corr_sheet,
            "segment_number": seg_num,
            "sample_id": str(sample_id) if pd.notna(sample_id) else "",
            "fs": fs,
            "t0": t0,
            "length": len(sig_corr),
            "qc_status": v2_res["metadata"].get("qc_status", "UNKNOWN"),

            # Base64 encoded signals (to save space in JSON)
            "sig_corr_b64": b64_encode_array(sig_corr),
            "sig_int_b64": b64_encode_array(sig_int) if len(sig_int) > 0 else None,
            "sig_cal_b64": b64_encode_array(sig_cal) if len(sig_cal) > 0 else None,

            # Detected events
            "candidates": {
                "time_s": candidates.get("time_s", np.array([])).tolist(),
                "amplitude": candidates.get("amplitude", np.array([])).tolist(),
                "status": candidates.get("status", np.array([])).tolist()
            }
        }
        segments_data.append(seg_data)

    return segments_data, Path(file_path).name


def generate_html(segments_data, source_workbook, out_path):
    # HTML template with embedded JS and Plotly via CDN
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cytocypher Step 19 - Real-data Review</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .controls { display: flex; gap: 20px; margin-bottom: 20px; align-items: center; flex-wrap: wrap;}
        #plot { width: 100%; height: 800px; }
        .info { margin-bottom: 10px; font-weight: bold; }
        .legend { font-size: 14px; margin-top: 10px; }
        button { padding: 8px 16px; cursor: pointer; }
        .summary { margin-top: 30px; border-top: 1px solid #ccc; padding-top: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #eee; }
    </style>
</head>
<body>
    <h2>Cytocypher V2 Validation Review Utility</h2>
    <div class="info">Source: <span id="source_name"></span></div>

    <div class="controls">
        <label>Segment:
            <select id="segment_select"></select>
        </label>
        <button id="btn_save">Save Annotations (CSV)</button>
        <div style="flex-grow: 1;"></div>
        <div>
            <b>Review Instructions:</b> Click an event to cycle its status: PASS -> REJECT -> REVIEW. Click on the line to add a missed event.
        </div>
    </div>

    <div id="segment_info" class="info"></div>
    <div id="plot"></div>

    <div class="summary">
        <h3>Summary</h3>
        <table id="summary_table">
            <thead>
                <tr>
                    <th>Segment</th>
                    <th>Sample ID</th>
                    <th>QC Status</th>
                    <th>Pass (Auto)</th>
                    <th>Manual Additions</th>
                    <th>Rejected</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <script>
        const segments_data = __DATA__;
        const source_workbook = "__SOURCE__";

        let currentSegIdx = 0;
        let annotations = {}; // segIdx -> { time_s, confidence, added, rejected, original_status }

        // Initialize annotations state
        segments_data.forEach((seg, idx) => {
            annotations[idx] = [];
            const cands = seg.candidates;
            for(let i=0; i<cands.time_s.length; i++) {
                annotations[idx].push({
                    time_s: cands.time_s[i],
                    amp: cands.amplitude[i],
                    status: cands.status[i],
                    original_status: cands.status[i],
                    added: false
                });
            }
        });

        // Base64 decoding for float32 array
        function decodeFloat32Array(b64) {
            const raw = window.atob(b64);
            const rawLength = raw.length;
            const array = new Uint8Array(new ArrayBuffer(rawLength));
            for(let i = 0; i < rawLength; i++) {
                array[i] = raw.charCodeAt(i);
            }
            return new Float32Array(array.buffer);
        }

        function populateSelector() {
            const sel = document.getElementById("segment_select");
            segments_data.forEach((seg, idx) => {
                const opt = document.createElement("option");
                opt.value = idx;
                opt.text = `${seg.sheet_name} (QC: ${seg.qc_status})`;
                sel.appendChild(opt);
            });
            sel.addEventListener("change", (e) => {
                currentSegIdx = parseInt(e.target.value);
                renderPlot();
            });

            document.getElementById("source_name").innerText = source_workbook;

            document.getElementById("btn_save").addEventListener("click", saveAnnotations);
        }

        function updateSummaryTable() {
            const tbody = document.querySelector("#summary_table tbody");
            tbody.innerHTML = "";
            segments_data.forEach((seg, idx) => {
                const tr = document.createElement("tr");

                let passCount = 0;
                let addedCount = 0;
                let rejectedCount = 0;

                annotations[idx].forEach(ann => {
                    if (ann.status === "PASS") passCount++;
                    if (ann.added) addedCount++;
                    if (ann.status === "REJECT" && ann.original_status === "PASS") rejectedCount++;
                });

                tr.innerHTML = `
                    <td>${seg.sheet_name}</td>
                    <td>${seg.sample_id}</td>
                    <td>${seg.qc_status}</td>
                    <td>${passCount}</td>
                    <td>${addedCount}</td>
                    <td>${rejectedCount}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function renderPlot() {
            const seg = segments_data[currentSegIdx];
            document.getElementById("segment_info").innerText = `Sample ID: ${seg.sample_id} | Length: ${seg.length} pts | fs: ${seg.fs} Hz`;

            const sig_corr = decodeFloat32Array(seg.sig_corr_b64);
            const times = new Float32Array(seg.length);
            for(let i=0; i<seg.length; i++) {
                times[i] = seg.t0 + i / seg.fs;
            }

            const traces = [];

            // PixelCorrelation Trace
            traces.push({
                x: times,
                y: sig_corr,
                mode: 'lines',
                name: 'PixelCorrelation',
                line: {color: 'blue'},
                yaxis: 'y1'
            });

            // PixelIntensity Trace
            if (seg.sig_int_b64) {
                const sig_int = decodeFloat32Array(seg.sig_int_b64);

                traces.push({
                    x: times,
                    y: sig_int,
                    mode: 'lines',
                    name: 'PixelIntensity',
                    line: {color: 'green', width: 1},
                    yaxis: 'y2',
                    visible: 'legendonly'
                });
            }

            // RatioMetricCalcium Trace
            if (seg.sig_cal_b64) {
                const sig_cal = decodeFloat32Array(seg.sig_cal_b64);

                traces.push({
                    x: times,
                    y: sig_cal,
                    mode: 'lines',
                    name: 'RatioMetricCalcium',
                    line: {color: 'red', width: 1},
                    yaxis: 'y3',
                    visible: 'legendonly'
                });
            }

            // Markers
            const anns = annotations[currentSegIdx];
            const pass_x = [], pass_y = [];
            const rej_x = [], rej_y = [];
            const rev_x = [], rev_y = [];

            anns.forEach(ann => {
                if (ann.status === "PASS") {
                    pass_x.push(ann.time_s); pass_y.push(ann.amp);
                } else if (ann.status === "REJECT") {
                    rej_x.push(ann.time_s); rej_y.push(ann.amp);
                } else if (ann.status === "REVIEW") {
                    rev_x.push(ann.time_s); rev_y.push(ann.amp);
                }
            });

            traces.push({
                x: pass_x, y: pass_y,
                mode: 'markers', name: 'PASS (Confirmed)',
                marker: { color: 'green', size: 10, symbol: 'circle' },
                yaxis: 'y1', type: 'scatter'
            });
            traces.push({
                x: rej_x, y: rej_y,
                mode: 'markers', name: 'REJECT (False Pos)',
                marker: { color: 'red', size: 8, symbol: 'x' },
                yaxis: 'y1', type: 'scatter'
            });
            traces.push({
                x: rev_x, y: rev_y,
                mode: 'markers', name: 'REVIEW (Uncertain)',
                marker: { color: 'orange', size: 10, symbol: 'square' },
                yaxis: 'y1', type: 'scatter'
            });

            const layout = {
                title: seg.sheet_name,
                hovermode: 'closest',
                xaxis: { title: 'Time (s)' },
                yaxis: { title: 'Correlation', domain: [0, 1] },
                yaxis2: { title: 'Intensity', overlaying: 'y', side: 'right', showgrid: false },
                yaxis3: { title: 'Calcium', overlaying: 'y', side: 'right', position: 0.95, showgrid: false },
                clickmode: 'event+select'
            };

            Plotly.newPlot('plot', traces, layout);

            document.getElementById('plot').on('plotly_click', function(data){
                const pt = data.points[0];
                const click_x = pt.x;

                // Find if we clicked on an existing marker
                let clicked_existing = false;
                for (let i=0; i<anns.length; i++) {
                    if (Math.abs(anns[i].time_s - click_x) < 0.001) {
                        // Cycle status: PASS -> REJECT -> REVIEW -> PASS
                        if (anns[i].status === "PASS") anns[i].status = "REJECT";
                        else if (anns[i].status === "REJECT") anns[i].status = "REVIEW";
                        else anns[i].status = "PASS";
                        clicked_existing = true;
                        break;
                    }
                }

                // If clicked on the line trace (PixelCorrelation), add a new event
                if (!clicked_existing && pt.data.name === 'PixelCorrelation') {
                    anns.push({
                        time_s: pt.x,
                        amp: pt.y,
                        status: "PASS",
                        original_status: "MISSING",
                        added: true
                    });
                }

                updateSummaryTable();

                // We could just update the traces using Plotly.react, but re-rendering is easy enough
                renderPlot();
            });

            updateSummaryTable();
        }

        function saveAnnotations() {
            const annotatorId = prompt("Enter annotator initials (e.g., JD):", "XX");
            if (!annotatorId) return;

            let csvContent = "source_workbook,sheet_name,sample_id,time_s,confidence,annotator_id\n";

            segments_data.forEach((seg, idx) => {
                const anns = annotations[idx];
                anns.forEach(ann => {
                    if (ann.status === "PASS" || ann.status === "REVIEW") {
                        const conf = (ann.status === "PASS") ? 1 : 0;
                        csvContent += `"${source_workbook}","${seg.sheet_name}","${seg.sample_id}",${ann.time_s},${conf},"${annotatorId}"\n`;
                    }
                });
            });

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", `ground_truth_annotations_${source_workbook}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        // Start
        populateSelector();
        renderPlot();
    </script>
</body>
</html>
"""

    # Inject data
    html = html_template.replace("__DATA__", json.dumps(segments_data))
    html = html.replace("__SOURCE__", source_workbook)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReview report generated successfully: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cytocypher Step 19 Validation Review Utility")
    parser.add_argument("input_file", help="Path to input Excel workbook")
    parser.add_argument("--output", default="v2_validation_review.html", help="Path to output HTML report")

    args = parser.parse_args()

    # Process
    try:
        segments_data, source_name = process_workbook(args.input_file)
        generate_html(segments_data, source_name, args.output)

        print("\nINSTRUCTIONS:")
        print(f"1. Open '{args.output}' in your web browser.")
        print("2. Review segments using the dropdown.")
        print("3. Click existing points to change status (PASS -> REJECT -> REVIEW).")
        print("4. Click on the raw blue trace (PixelCorrelation) to add missed events.")
        print("5. Click 'Save Annotations' when done to generate a CSV matching the required schema.")
        print("NOTE: The holdout dataset 240926_IMR90 should not be processed here until final validation.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
