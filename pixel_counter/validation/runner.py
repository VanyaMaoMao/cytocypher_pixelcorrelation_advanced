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
        row_len = int(fs)
        seg_meta = []
        for i in range(len(sig) // row_len):
            s = i * row_len
            e = s + row_len
            seg_meta.append((i, s, e, float(i), -1))

        if len(sig) % row_len != 0:
            s = (len(sig) // row_len) * row_len
            seg_meta.append((len(seg_meta), s, len(sig), float(s/fs), -1))

        orient_meta = {
            "manual_orientation": true_polarity,
            "confidence": 1.0,
            "up_fraction": 1.0 if true_polarity == 1 else 0.0,
            "down_fraction": 1.0 if true_polarity == -1 else 0.0,
            "row_direction_transition_score": 0.0
        }

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

        legacy_evaluable = True
        if legacy_status == 0 and len(legacy_times) == 0:
            legacy_status_str = "NOT_EVALUABLE"
            legacy_evaluable = False

        l_metrics = calculate_validation_metrics(
            len(l_tp_idx), len(l_fp_idx), len(l_fn_idx),
            len(legacy_times), len(true_times),
            legacy_rate_hz,
            len(true_times) / (len(sig)/fs),
            len(sig)/fs
        )

        # Override metrics if not evaluable
        if not legacy_evaluable:
            for k in ["precision", "recall", "f1", "hz_error", "bpm_error", "count_error", "abs_count_error"]:
                l_metrics[k] = "N/A"
            for k in ["tp", "fp", "fn"]:
                l_metrics[k] = "N/A"

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
        if legacy_evaluable:
            legacy_totals["tp"] += l_metrics["tp"]
            legacy_totals["fp"] += l_metrics["fp"]
            legacy_totals["fn"] += l_metrics["fn"]
            legacy_totals["pred"] += l_metrics["tp"] + l_metrics["fp"]
            legacy_totals["true"] += len(true_times)
            legacy_totals["abs_count_error"] += l_metrics["abs_count_error"]

        v2_totals["tp"] += v_metrics["tp"]
        v2_totals["fp"] += v_metrics["fp"]
        v2_totals["fn"] += v_metrics["fn"]
        v2_totals["pred"] += v_metrics["tp"] + v_metrics["fp"]
        v2_totals["true"] += len(true_times)
        v2_totals["abs_count_error"] += v_metrics["abs_count_error"]

    # Calculate aggregated metrics
    for totals, key in [(legacy_totals, "legacy_aggregated"), (v2_totals, "v2_aggregated")]:
        if totals["true"] == 0 and totals["tp"] == 0 and totals["fp"] == 0:
             results[key] = {
                "tp": "N/A", "fp": "N/A", "fn": "N/A",
                "precision": "N/A", "recall": "N/A", "f1": "N/A",
                "abs_count_error": "N/A"
             }
             continue

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
