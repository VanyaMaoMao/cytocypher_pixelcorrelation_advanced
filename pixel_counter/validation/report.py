import json
import os
from pathlib import Path

def generate_comparison_report(results: dict, output_path: str):
    """
    Generates a reproducible comparison report.
    """
    report_md = f"# Cytocypher Scientific Validation Report\n\n"

    # Check if this is real validation or synthetic fallback
    is_synthetic = results.get("is_synthetic", True)

    if is_synthetic:
        report_md += "## ⚠️ WARNING: SYNTHETIC DATA FALLBACK ⚠️\n\n"
        report_md += "This validation report was generated using **deterministic synthetic ground truth** because independently annotated real recordings are unavailable.\n\n"
        report_md += "**Status: BLOCKED**\n\n"
        report_md += "Scientific validation of the v2 detector is BLOCKED. The detector is NOT validated or ready to replace the legacy pipeline. Step 20 cannot proceed until real annotations are collected and evaluated against the predefined acceptance criteria.\n\n"
    else:
        report_md += "## Real Data Validation\n\n"
        report_md += "**Status: COMPLETE** (Pending review of acceptance criteria)\n\n"

    report_md += "## Evaluation Parameters\n"
    report_md += f"- **Tolerance (s):** {results.get('tolerance_s', 'Unknown')}\n"
    report_md += f"- **V2 Config:** `{results.get('v2_config', 'Default')}`\n"

    report_md += "\n## Overall Results Summary\n\n"
    report_md += "| Metric | Legacy Pipeline | V2 Pipeline |\n"
    report_md += "|---|---|---|\n"

    legacy_agg = results.get("legacy_aggregated", {})
    v2_agg = results.get("v2_aggregated", {})

    metrics_to_show = ["f1", "precision", "recall", "abs_count_error", "tp", "fp", "fn"]
    for m in metrics_to_show:
        l_val = legacy_agg.get(m, "N/A")
        v_val = v2_agg.get(m, "N/A")
        if isinstance(l_val, float): l_val = f"{l_val:.4f}"
        if isinstance(v_val, float): v_val = f"{v_val:.4f}"
        report_md += f"| {m} | {l_val} | {v_val} |\n"

    report_md += "\n## Per-Recording Results\n\n"

    for rec_id, rec_res in results.get("recordings", {}).items():
        report_md += f"### Recording: {rec_id}\n"
        report_md += f"- True Event Count: {rec_res.get('true_count')}\n\n"

        report_md += "| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |\n"
        report_md += "|---|---|---|---|---|---|---|---|---|---|---|\n"

        for ver in ["legacy", "v2"]:
            v_res = rec_res.get(ver, {})
            metrics = v_res.get("metrics", {})
            f1 = metrics.get('f1', 0)
            f1 = f"{f1:.4f}" if isinstance(f1, float) else f1
            prec = metrics.get('precision', 0)
            prec = f"{prec:.4f}" if isinstance(prec, float) else prec
            rec = metrics.get('recall', 0)
            rec = f"{rec:.4f}" if isinstance(rec, float) else rec

            tp = metrics.get('tp', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            c_err = metrics.get('count_error', 0)
            hz_err = metrics.get('hz_error', 'NaN')
            if isinstance(hz_err, float) and not json.dumps(hz_err) == 'NaN':
                hz_err = f"{hz_err:.4f}"

            pol_correct = v_res.get("polarity_correct", "N/A")
            qc = v_res.get("qc_status", "N/A")

            report_md += f"| {ver} | {f1} | {prec} | {rec} | {tp} | {fp} | {fn} | {c_err} | {hz_err} | {pol_correct} | {qc} |\n"
        report_md += "\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Write JSON as well for programmatic access
    json_path = output_path.replace('.md', '.json')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
