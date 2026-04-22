from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .results import AFCEvent, AFCSegmentReviewDecision, AFCReviewSession

def load_cytocypher_excel(file_path: Union[str, pd.ExcelFile], sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], float, float]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    y_cols = [c for c in df.columns if str(c).startswith("y ")]
    if not y_cols:
        raise ValueError(f"No 'y ' columns in {sheet_name!r}")

    if "Sampling Frequency" in df.columns:
        s = pd.to_numeric(df["Sampling Frequency"], errors="coerce").dropna()
        fs = float(s.iloc[0]) if not s.empty else 250.0
    else:
        fs = 250.0

    if "Begin (seconds)" in df.columns:
        s = pd.to_numeric(df["Begin (seconds)"], errors="coerce").dropna()
        t0 = float(s.iloc[0]) if not s.empty else 0.0
    else:
        t0 = 0.0

    out = df.copy()
    if "Transientnumber" in out.columns:
        tn = pd.to_numeric(out["Transientnumber"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        tn = tn.fillna(pd.Series(np.arange(len(out)), index=out.index))
        out["_tn"] = tn.astype(int)
        out = out.sort_values("_tn")
    else:
        out["_tn"] = np.arange(len(out))
    return out, y_cols, fs, t0


def extract_sample_id_from_segment_sheet(file_path: str, sheet_name: str) -> Any:
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols=lambda c: str(c).strip().lower() == "sample id",
        )
    except Exception:
        return np.nan
    if df is None or df.empty:
        return np.nan
    col = next((c for c in df.columns if str(c).strip().lower() == "sample id"), None)
    if col is None:
        return np.nan
    s = df[col]
    s = s.dropna()
    if s.empty:
        return np.nan
    s = s[s.astype(str).str.strip() != ""]
    if s.empty:
        return np.nan
    # Prefer stable value if repeated; otherwise use the most frequent non-null.
    val = s.value_counts(dropna=True).index[0]
    try:
        vf = float(val)
        if np.isfinite(vf) and abs(vf - round(vf)) < 1e-9:
            return int(round(vf))
    except Exception:
        pass
    return val


def save_afc_review_session_json(path: str, session: AFCReviewSession) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=True)


def load_afc_review_session_json(path: str) -> AFCReviewSession:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AFCReviewSession.from_dict(data)


def export_afc_events_csv(path: str, afc_events: List[AFCEvent]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not afc_events:
        pd.DataFrame(columns=["segment_name", "segment_index", "main_peak_index", "time_s", "amplitude", "source", "review_id"]).to_csv(out_path, index=False)
        return
    rows = [x.to_dict() for x in afc_events]
    pd.DataFrame(rows).sort_values(["segment_index", "main_peak_index", "time_s"]).to_csv(out_path, index=False)


def export_afc_review_log_csv(path: str, decisions: List[AFCSegmentReviewDecision]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not decisions:
        pd.DataFrame(
            columns=[
                "segment_name",
                "segment_index",
                "afc_lower_left_value",
                "afc_lower_right_value",
                "afc_upper_left_value",
                "afc_upper_right_value",
                "x_start_s",
                "x_end_s",
                "manual_afc_times_s",
                "manual_afc_amps",
                "status",
            ]
        ).to_csv(out_path, index=False)
        return
    rows = []
    for d in decisions:
        row = d.to_dict()
        row.pop("main_peak_index", None)
        row.pop("accepted_times_s", None)
        row.pop("accepted_amps", None)
        row.pop("rejected_times_s", None)
        row.pop("rejected_amps", None)
        row.pop("manual_added_times_s", None)
        row.pop("manual_added_amps", None)
        row.pop("notes", None)
        if "lower_line" in row and "afc_left_value" not in row:
            row["afc_left_value"] = row.pop("lower_line")
        if "upper_line" in row and "afc_right_value" not in row:
            row["afc_right_value"] = row.pop("upper_line")
        if "afc_lower_left_value" not in row:
            row["afc_lower_left_value"] = row.get("afc_left_value", np.nan)
        if "afc_lower_right_value" not in row:
            row["afc_lower_right_value"] = row.get("afc_right_value", np.nan)
        if "afc_upper_left_value" not in row:
            row["afc_upper_left_value"] = row.get("afc_upper_cap", row.get("upper_line", np.nan))
        if "afc_upper_right_value" not in row:
            row["afc_upper_right_value"] = row.get("afc_upper_cap", row.get("upper_line", np.nan))
        row.pop("afc_left_value", None)
        row.pop("afc_right_value", None)
        row.pop("afc_upper_cap", None)
        row.pop("lower_line", None)
        row.pop("upper_line", None)
        if "window_start_s" in row and "x_start_s" not in row:
            row["x_start_s"] = row.pop("window_start_s")
        if "window_end_s" in row and "x_end_s" not in row:
            row["x_end_s"] = row.pop("window_end_s")
        manual_times = (
            list(d.manual_afc_times_s)
            if d.manual_afc_times_s
            else list(d.manual_added_times_s) + list(d.accepted_times_s)
        )
        manual_amps = (
            list(d.manual_afc_amps)
            if d.manual_afc_amps
            else list(d.manual_added_amps) + list(d.accepted_amps)
        )
        row["manual_afc_times_s"] = ",".join(f"{float(x):.6f}" for x in manual_times)
        row["manual_afc_amps"] = ",".join(f"{float(x):.6f}" for x in manual_amps)
        rows.append(row)
    pd.DataFrame(rows).sort_values(["segment_index"]).to_csv(out_path, index=False)


def export_peak_debug_csv(path: str, peak_debug_df: pd.DataFrame) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if peak_debug_df is None or peak_debug_df.empty:
        pd.DataFrame(
            columns=[
                "segment_name",
                "segment_index",
                "peak_index_raw",
                "time_s",
                "amplitude",
                "prominence",
                "width_s",
                "transient_index",
                "stage_first_seen",
                "survived_raw_filter",
                "survived_main_candidate_stage",
                "survived_dedup_stage",
                "survived_short_gap_prune",
                "survived_local_weak_prune",
                "survived_interbeat_tiny_filter",
                "survived_rescue_stage",
                "final_label",
                "rejection_reason",
                "notes",
            ]
        ).to_csv(out_path, index=False)
        return
    peak_debug_df.to_csv(out_path, index=False)


def export_peak_debug_xlsx(path: str, peak_debug_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if peak_debug_df is None or peak_debug_df.empty:
            export_df = pd.DataFrame(
                columns=[
                    "segment_name",
                    "segment_index",
                    "peak_index_raw",
                    "time_s",
                    "amplitude",
                    "prominence",
                    "width_s",
                    "transient_index",
                    "stage_first_seen",
                    "survived_raw_filter",
                    "survived_main_candidate_stage",
                    "survived_dedup_stage",
                    "survived_short_gap_prune",
                    "survived_local_weak_prune",
                    "survived_interbeat_tiny_filter",
                    "survived_rescue_stage",
                    "final_label",
                    "rejection_reason",
                    "notes",
                ]
            )
        else:
            export_df = peak_debug_df
        export_df.to_excel(writer, sheet_name="peak_debug", index=False)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="peak_debug_summary", index=False)
