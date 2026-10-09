import json
import pandas as pd
import numpy as np
from pathlib import Path
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig
import traceback
import hashlib

def default_encoder(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

def main():
    config = BeatCounterConfig()
    files = [
        "240926_IMR90_1,2,3HZ individual transients result.xlsx",
        "25092026_IMR90_1,2,3HZ individual transients result.xlsx"
    ]

    baseline = []

    for file_path in files:
        print(f"Processing {file_path}...")

        try:
            with open(file_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            checksum = "unknown"

        excel_file = pd.ExcelFile(file_path)
        pixel_sheets = [k for k in excel_file.sheet_names if "PixelCorrelation Segment" in k]

        for sheet_name in pixel_sheets:
            try:
                bpm, count, events, meta = count_main_beats_from_excel(
                    file_path, sheet_name=sheet_name, config=config, show_plot=False
                )

                sample_id = meta.get("sample_id")
                qc_status = "UNKNOWN"
                if "qc_pass" in meta:
                    qc_status = "PASS" if meta["qc_pass"] else "REVIEW"
                if meta.get("hard_noise", False):
                    qc_status = "REJECT"
                if "hard_reject_reason" in meta and meta["hard_reject_reason"]:
                    qc_status = "REJECT"

                qc_reason = meta.get("qc_reason", "") or meta.get("hard_reject_reason", "")
                orientation_dict = meta.get("orientation", {})
                polarity = orientation_dict.get("invert")

                event_indices = []
                event_timestamps = []
                if events is not None and not events.empty:
                    acc = events[events["decision_status"] == "accepted"]
                    event_indices = acc["candidate_index"].tolist()

                    time_col = "Time_s" if "Time_s" in acc.columns else ("time_s" if "time_s" in acc.columns else None)
                    if time_col:
                        event_timestamps = acc[time_col].tolist()

                observed_coverage = None
                duration_s = meta.get("duration_s")
                recording_s = meta.get("recording_span_s")
                if duration_s is not None and recording_s is not None and recording_s > 0:
                    observed_coverage = duration_s / recording_s

                baseline.append({
                    "source_workbook": Path(file_path).name,
                    "source_sha256": checksum,
                    "sample_id": sample_id,
                    "segment_identifier": sheet_name,
                    "accepted_event_count": int(count) if not np.isnan(count) else None,
                    "event_sample_indices": event_indices,
                    "event_timestamps": event_timestamps,
                    "selected_polarity_invert": polarity,
                    "qc_status": qc_status,
                    "qc_reasons": qc_reason,
                    "recording_span_s": recording_s,
                    "analyzed_duration_s": duration_s,
                    "observed_coverage": observed_coverage
                })
            except Exception as e:
                print(f"Error in {sheet_name}: {e}")
                baseline.append({
                    "source_workbook": Path(file_path).name,
                    "source_sha256": checksum,
                    "segment_identifier": sheet_name,
                    "qc_status": "ERROR",
                    "qc_reasons": str(e)
                })

    # Serialize, avoiding NaN where not allowed
    json_str = json.dumps(baseline, indent=2, default=default_encoder, allow_nan=False)
    Path("docs/real_data_baseline.json").write_text(json_str)
    print("Baseline saved to docs/real_data_baseline.json")

    # Assert there are exactly 90 segments
    assert len(baseline) == 90, f"Expected 90 segments, got {len(baseline)}"

if __name__ == "__main__":
    main()
