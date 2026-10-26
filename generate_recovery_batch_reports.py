"""
Batch report generator for Sarcomere Shortening and Calcium Assistance Recovery.
Processes all 4 experimental workbooks into batch_reviews_sarcomere_recovery/.
"""

import time
import csv
from pathlib import Path
from pixel_counter.sarcomere_review import process_sarcomere_workbook, generate_sarcomere_html

WORKBOOKS = [
    {
        "id": "01",
        "label": "CAR+ROLI+IBMX DISH1 (Primary Regression)",
        "file": "(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx",
        "out_sarc": "review_01_car_dish1_sarcomere.html",
        "out_comb": "review_01_car_dish1_sarcomere_calcium.html",
    },
    {
        "id": "02",
        "label": "CAR+ROLI+IBMX DISH2",
        "file": "(MI) CGP+CAR+ROLI+IBMX   individual transients result.xlsx",
        "out_sarc": "review_02_car_dish2_sarcomere.html",
        "out_comb": "review_02_car_dish2_sarcomere_calcium.html",
    },
    {
        "id": "03",
        "label": "ISO+ROLI+IBMX DISH1 (High Frequency Reference)",
        "file": "(MI) CGP+ISO+ROLI+IBMX DISH1  individual transients result.xlsx",
        "out_sarc": "review_03_iso_dish1_sarcomere.html",
        "out_comb": "review_03_iso_dish1_sarcomere_calcium.html",
    },
    {
        "id": "04",
        "label": "ISO+ROLI+IBMX DISH2",
        "file": "(MI) CGP+ISO+ROLI+IBMX  individual transients result.xlsx",
        "out_sarc": "review_04_iso_dish2_sarcomere.html",
        "out_comb": "review_04_iso_dish2_sarcomere_calcium.html",
    },
]

out_dir = Path("batch_reviews_sarcomere_recovery")
out_dir.mkdir(parents=True, exist_ok=True)

manifest_rows = [
    [
        "workbook_id",
        "label",
        "workbook_file",
        "analysis_mode",
        "assistance_enabled",
        "html_report",
        "total_segments",
        "pass_count",
        "pass_partial_count",
        "reject_count",
        "total_accepted_contractions",
        "total_main_contractions",
        "total_secondary_contractions",
        "total_ambiguous_contractions",
        "mean_shortening_pct",
        "mean_frequency_hz",
    ]
]

t0_total = time.time()

for wb in WORKBOOKS:
    src_file = Path("sarcomere shortening") / wb["file"]
    print(f"\n=======================================================")
    print(f"Workbook {wb['id']}: {wb['label']}")
    print(f"File: {src_file.name}")
    print(f"=======================================================")

    # Configuration 1: Sarcomere Only
    print(f"\n---> Running SARCOMERE ONLY (2 panels)...")
    t0 = time.time()
    sarc_data, sarc_name = process_sarcomere_workbook(src_file, analysis_mode="sarcomere")
    sarc_html = out_dir / wb["out_sarc"]
    generate_sarcomere_html(sarc_data, sarc_name, sarc_html, analysis_mode="sarcomere")

    # Aggregate Sarcomere Only statistics
    n_seg = len(sarc_data)
    n_pass = sum(1 for s in sarc_data if s["qc_status"] == "PASS")
    n_partial = sum(1 for s in sarc_data if s["qc_status"] == "PASS_PARTIAL")
    n_reject = sum(1 for s in sarc_data if s["qc_status"] not in ("PASS", "PASS_PARTIAL"))
    tot_acc = sum(s["runs"]["default"]["n_accepted"] for s in sarc_data)
    tot_main = sum(s["runs"]["default"]["n_main"] for s in sarc_data)
    tot_sec = sum(s["runs"]["default"]["n_secondary"] for s in sarc_data)
    tot_amb = sum(s["runs"]["default"]["n_ambiguous"] for s in sarc_data)
    
    valid_pcts = [s["runs"]["default"]["mean_pct_shortening"] for s in sarc_data if s["runs"]["default"]["mean_pct_shortening"] is not None]
    mean_pct = round(sum(valid_pcts) / len(valid_pcts), 2) if valid_pcts else "N/A"

    valid_freqs = [s["runs"]["default"]["frequency_hz"] for s in sarc_data if s["runs"]["default"]["frequency_hz"] is not None]
    mean_freq = round(sum(valid_freqs) / len(valid_freqs), 2) if valid_freqs else "N/A"

    manifest_rows.append([
        wb["id"], wb["label"], wb["file"], "sarcomere", "OFF",
        wb["out_sarc"], n_seg, n_pass, n_partial, n_reject,
        tot_acc, tot_main, tot_sec, tot_amb, mean_pct, mean_freq
    ])
    print(f"Completed SARCOMERE ONLY in {time.time()-t0:.1f}s | Segments: {n_seg} (PASS: {n_pass}, PARTIAL: {n_partial}, REJ: {n_reject}) | Contractions: {tot_acc} (M:{tot_main}, S:{tot_sec})")

    # Configuration 2: Sarcomere + Calcium (Assistance ON)
    print(f"\n---> Running SARCOMERE + CALCIUM (Assistance ON, 3 panels)...")
    t0 = time.time()
    comb_data, comb_name = process_sarcomere_workbook(src_file, analysis_mode="sarcomere-calcium", enable_assistance=True)
    comb_html = out_dir / wb["out_comb"]
    generate_sarcomere_html(comb_data, comb_name, comb_html, analysis_mode="sarcomere-calcium")

    # Aggregate Sarcomere + Calcium statistics
    n_pass_c = sum(1 for s in comb_data if s["qc_status"] == "PASS")
    n_partial_c = sum(1 for s in comb_data if s["qc_status"] == "PASS_PARTIAL")
    n_reject_c = sum(1 for s in comb_data if s["qc_status"] not in ("PASS", "PASS_PARTIAL"))
    tot_acc_c = sum(s["runs"]["default"]["n_accepted"] for s in comb_data)
    tot_main_c = sum(s["runs"]["default"]["n_main"] for s in comb_data)
    tot_sec_c = sum(s["runs"]["default"]["n_secondary"] for s in comb_data)
    tot_amb_c = sum(s["runs"]["default"]["n_ambiguous"] for s in comb_data)

    valid_pcts_c = [s["runs"]["default"]["mean_pct_shortening"] for s in comb_data if s["runs"]["default"]["mean_pct_shortening"] is not None]
    mean_pct_c = round(sum(valid_pcts_c) / len(valid_pcts_c), 2) if valid_pcts_c else "N/A"

    valid_freqs_c = [s["runs"]["default"]["frequency_hz"] for s in comb_data if s["runs"]["default"]["frequency_hz"] is not None]
    mean_freq_c = round(sum(valid_freqs_c) / len(valid_freqs_c), 2) if valid_freqs_c else "N/A"

    manifest_rows.append([
        wb["id"], wb["label"], wb["file"], "sarcomere-calcium", "ON",
        wb["out_comb"], n_seg, n_pass_c, n_partial_c, n_reject_c,
        tot_acc_c, tot_main_c, tot_sec_c, tot_amb_c, mean_pct_c, mean_freq_c
    ])
    print(f"Completed SARCOMERE + CALCIUM in {time.time()-t0:.1f}s | Segments: {n_seg} (PASS: {n_pass_c}, PARTIAL: {n_partial_c}, REJ: {n_reject_c}) | Contractions: {tot_acc_c} (M:{tot_main_c}, S:{tot_sec_c})")

# Write manifest CSV
manifest_path = out_dir / "manifest.csv"
with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(manifest_rows)

print(f"\n=======================================================")
print(f"ALL 8 REPORTS GENERATED IN {time.time()-t0_total:.1f}s!")
print(f"Manifest written to: {manifest_path}")
print(f"=======================================================")
