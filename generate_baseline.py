import json
import pandas as pd
import numpy as np
from pathlib import Path
from pixel_counter.analysis import count_main_beats_from_excel
from pixel_counter.config import BeatCounterConfig

def default_encoder(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
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

        excel_file = pd.ExcelFile(file_path)
        pixel_sheets = [k for k in excel_file.sheet_names if "PixelCorrelation Segment" in k]

        for sheet_name in pixel_sheets:
            bpm, count, events, meta = count_main_beats_from_excel(
                file_path, sheet_name=sheet_name, config=config, show_plot=False
            )

            sample_id = meta.get("sample_id", "Unknown")
            qc_status = meta.get("quality_status", "UNKNOWN")
            polarity = meta.get("orientation", "UNKNOWN")

            event_indices = []
            event_timestamps = []
            if events is not None and not events.empty:
                acc = events[events["decision_status"] == "accepted"]
                event_indices = acc["candidate_index"].tolist()
                event_timestamps = acc["time_s"].tolist() if "time_s" in acc.columns else []

            baseline.append({
                "source_workbook": Path(file_path).name,
                "sample_id": sample_id,
                "segment_identifier": sheet_name,
                "accepted_event_count": count,
                "event_sample_indices": event_indices,
                "event_timestamps": event_timestamps,
                "selected_polarity": polarity,
                "qc_status": qc_status,
                "qc_reasons": meta.get("quality_reasons", [])
            })

    Path("docs/real_data_baseline.json").write_text(json.dumps(baseline, indent=2, default=default_encoder))
    print("Baseline saved to docs/real_data_baseline.json")

if __name__ == "__main__":
    main()
