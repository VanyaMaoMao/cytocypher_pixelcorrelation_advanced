import numpy as np
import pandas as pd
from typing import Dict, Any

from pixel_counter.config import BeatCounterConfig
from pixel_counter.analysis import _analyze_prebuilt_signal, analyze_v2_pipeline
from pixel_counter.validation.synthetic import generate_synthetic_validation_dataset
from pixel_counter.validation.matching import match_events_one_to_one
from pixel_counter.validation.metrics import calculate_validation_metrics
from pixel_counter.validation.report import generate_comparison_report

def run_synthetic_validation(tolerance_s: float = 0.05, output_dir: str = "docs/validation") -> Dict[str, Any]:
    dataset = generate_synthetic_validation_dataset()

    config = BeatCounterConfig(
        v2_polarity_mode="auto",
        v2_two_pass_enabled=False
    )

    results = {
        "is_synthetic": True,
        "tolerance_s": tolerance_s,
        "v2_config": "auto, no_two_pass",
        "recordings": {}
    }

    legacy_totals = {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "true": 0, "abs_count_error": 0}
    v2_totals = {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "true": 0, "abs_count_error": 0}

    for rec_id, rec_data in dataset.items():
        sig = rec_data["signal"]
        fs = rec_data["fs"]
        true_times = rec_data["true_times"]
        true_polarity = rec_data["true_polarity"]

        # 1. Run legacy pipeline
        # Need to provide valid mask and dummy seg_meta, orient_meta
        seg_meta = [(0, len(sig), 1, 0.0, -1)]
        orient_meta = {"manual_orientation": true_polarity}

        df_source = pd.DataFrame({"y 0": sig})

        bpm, legacy_status, legacy_events_df, debug_res = _analyze_prebuilt_signal(
            sig=sig,
            fs=fs,
            t0=0.0,
            seg_meta=seg_meta,
            orient_meta=orient_meta,
            config=config,
            file_path="synthetic.xlsx",
            display_name=rec_id,
            sheet_name="synthetic",
            candidate_label="synthetic",
            df_source=df_source,
            y_cols_source=["y 0"],
            sample_id="synth1"
        )

        legacy_times = legacy_events_df["Time_s"].values if not legacy_events_df.empty else np.array([])

        l_tp_idx, l_fp_idx, l_fn_idx = match_events_one_to_one(legacy_times, true_times, tolerance_s)

        legacy_rate_hz = bpm / 60.0 if (not np.isnan(bpm) and legacy_status > 0) else np.nan
        legacy_status_str = "PASS" if legacy_status == 1 else "ERROR/REJECT"

        l_metrics = calculate_validation_metrics(
            len(l_tp_idx), len(l_fp_idx), len(l_fn_idx),
            len(legacy_times), len(true_times),
            legacy_rate_hz,
            len(true_times) / (len(sig)/fs), # Approx true rate
            len(sig)/fs
        )

        # 2. Run v2 pipeline
        v2_res = analyze_v2_pipeline(sig, fs, config)
        v2_times = v2_res["candidates"]["time_s"] if "time_s" in v2_res["candidates"] else np.array([])

        v_tp_idx, v_fp_idx, v_fn_idx = match_events_one_to_one(v2_times, true_times, tolerance_s)

        v2_count = len(v2_times)
        v2_rate_hz = v2_count / (len(sig)/fs) if v2_res["metadata"].get("qc_status") in ["PASS", "REVIEW"] else np.nan

        v_metrics = calculate_validation_metrics(
            len(v_tp_idx), len(v_fp_idx), len(v_fn_idx),
            v2_count, len(true_times),
            v2_rate_hz,
            len(true_times) / (len(sig)/fs),
            len(sig)/fs
        )

        # Check polarity
        v2_polarity = v2_res["polarity_result"]["best_polarity"]
        pol_correct = (v2_polarity == true_polarity)

        results["recordings"][rec_id] = {
            "true_count": len(true_times),
            "description": rec_data["description"],
            "legacy": {
                "metrics": l_metrics,
                "qc_status": legacy_status_str
            },
            "v2": {
                "metrics": v_metrics,
                "polarity_correct": pol_correct,
                "qc_status": v2_res["metadata"]["qc_status"]
            }
        }

        # Accumulate totals
        for totals, met in [(legacy_totals, l_metrics), (v2_totals, v_metrics)]:
            totals["tp"] += met["tp"]
            totals["fp"] += met["fp"]
            totals["fn"] += met["fn"]
            totals["pred"] += met["tp"] + met["fp"]
            totals["true"] += len(true_times)
            totals["abs_count_error"] += met["abs_count_error"]

    # Calculate aggregated metrics
    for totals, key in [(legacy_totals, "legacy_aggregated"), (v2_totals, "v2_aggregated")]:
        tp, fp, fn, true_c = totals["tp"], totals["fp"], totals["fn"], totals["true"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / true_c if true_c > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[key] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "abs_count_error": totals["abs_count_error"]
        }

    generate_comparison_report(results, f"{output_dir}/synthetic_validation_report.md")

    return results

if __name__ == "__main__":
    res = run_synthetic_validation()
    print("Synthetic validation run completed. Report saved.")
