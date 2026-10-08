from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .results import AFCEvent, AFCSegmentReviewDecision, AFCReviewSession

def load_cytocypher_excel(file_path: Union[str, pd.ExcelFile], sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], float, float]:
    if isinstance(file_path, pd.ExcelFile):
        df = file_path.parse(sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
    if isinstance(df, dict):
        # pd.read_excel returns a dict if sheet_name is None and there are multiple sheets.
        # But this function expects a single dataframe. Usually it is called per-sheet.
        if sheet_name is None:
            # Fallback to the first sheet if none specified and a dict is returned
            first_sheet = list(df.keys())[0]
            df = df[first_sheet]
            sheet_name = first_sheet

    # 1. Parse and validate y columns
    y_col_names = [c for c in df.columns if str(c).startswith("y ")]
    if not y_col_names:
        raise ValueError(f"No 'y ' columns in {sheet_name!r}")
    
    parsed_y_cols = []
    seen_offsets = set()
    for c in y_col_names:
        offset_str = str(c)[2:].strip()
        try:
            offset_val = float(offset_str)
        except ValueError:
            raise ValueError(f"Non-numeric y-offset found: {c!r}")
        if offset_val in seen_offsets:
            raise ValueError(f"Duplicate y-offset found for value {offset_val}")
        seen_offsets.add(offset_val)
        parsed_y_cols.append((offset_val, c))
    
    parsed_y_cols.sort(key=lambda x: x[0])
    y_cols = [c for _, c in parsed_y_cols]

    # 2. Parse and validate Sampling Frequency
    if "Sampling Frequency" in df.columns:
        s = pd.to_numeric(df["Sampling Frequency"], errors="coerce").dropna()
        if not s.empty:
            unique_fs = s.unique()
            if len(unique_fs) > 1:
                raise ValueError(f"Conflicting Sampling Frequency values found: {unique_fs}")
            fs = float(unique_fs[0])
            if not np.isfinite(fs) or fs <= 0.0:
                raise ValueError(f"Invalid Sampling Frequency: {fs}")
        else:
            fs = 250.0  # Documented metadata source fallback
    else:
        fs = 250.0

    # 3. Parse Sample ID
    sample_id_col = next((c for c in df.columns if str(c).strip().lower() == "sample id"), None)
    if sample_id_col is not None:
        sample_ids = df[sample_id_col].dropna().astype(str).str.strip()
        sample_ids = sample_ids[sample_ids != ""]
        if not sample_ids.empty:
            unique_sample_ids = sample_ids.unique()
            if len(unique_sample_ids) > 1:
                raise ValueError(f"Conflicting Sample IDs found: {unique_sample_ids}")

    # 4. Parse Begin (seconds)
    if "Begin (seconds)" in df.columns:
        s = pd.to_numeric(df["Begin (seconds)"], errors="coerce").dropna()
        t0 = float(s.iloc[0]) if not s.empty else 0.0
    elif "Begin" in df.columns:
        s = pd.to_numeric(df["Begin"], errors="coerce").dropna()
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


@contextlib.contextmanager
def atomic_file_path(target_path: Union[str, Path]) -> Iterator[Path]:
    """
    Context manager that provides a temporary path for writing.
    Upon successful exit, the temporary file is atomically moved to target_path.
    If an exception occurs, the temporary file is cleaned up and the target is untouched.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary file in the same directory to ensure atomic os.replace
    fd, tmp_path_str = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    
    try:
        yield tmp_path
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def build_output_path(input_path: Union[str, Path], suffix: str) -> str:
    """
    Safely build an output path replacing the original extension (.xlsx) 
    with the given suffix, without relying on simple string replace.
    """
    p = Path(input_path)
    if p.suffix.lower() == ".xlsx":
        return str(p.with_name(p.stem + suffix))
    else:
        return str(p.with_name(p.name + suffix))


def check_output_conflicts(input_path: Union[str, Path], output_paths: List[Union[str, Path]]) -> None:
    """
    Raises ValueError if any output_path conflicts with input_path.
    It checks using absolute resolved paths and samefile (where possible).
    """
    in_p = Path(input_path).resolve()
    for out in output_paths:
        if not out:
            continue
        out_p = Path(out).resolve()
        
        # simple path match (handles case sensitivity on Windows if paths are resolved properly)
        if in_p == out_p or str(in_p).lower() == str(out_p).lower():
            raise ValueError(f"Output path conflicts with input path: {in_p}")
        
        # If output file already exists, check samefile
        if in_p.exists() and out_p.exists():
            try:
                if in_p.samefile(out_p):
                    raise ValueError(f"Output path conflicts with input path (samefile): {in_p}")
            except OSError:
                pass


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
    s = df[col].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return np.nan
    
    unique_vals = s.unique()
    if len(unique_vals) > 1:
        raise ValueError(f"Conflicting Sample IDs found: {unique_vals}")
        
    val = unique_vals[0]
    try:
        vf = float(val)
        if np.isfinite(vf) and abs(vf - round(vf)) < 1e-9:
            return int(round(vf))
    except Exception:
        pass
    return val


def save_afc_review_session_json(path: str, session: AFCReviewSession) -> None:
    with atomic_file_path(path) as tmp_path:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=True)


def load_afc_review_session_json(path: str) -> AFCReviewSession:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AFCReviewSession.from_dict(data)


def export_afc_events_csv(path: str, afc_events: List[AFCEvent]) -> None:
    with atomic_file_path(path) as tmp_path:
        if not afc_events:
            pd.DataFrame(columns=["segment_name", "segment_index", "main_peak_index", "time_s", "amplitude", "source", "review_id"]).to_csv(tmp_path, index=False)
            return
        rows = [x.to_dict() for x in afc_events]
        pd.DataFrame(rows).sort_values(["segment_index", "main_peak_index", "time_s"]).to_csv(tmp_path, index=False)


def export_afc_review_log_csv(path: str, decisions: List[AFCSegmentReviewDecision]) -> None:
    with atomic_file_path(path) as tmp_path:
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
            ).to_csv(tmp_path, index=False)
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
        pd.DataFrame(rows).sort_values(["segment_index"]).to_csv(tmp_path, index=False)


def export_peak_debug_csv(path: str, peak_debug_df: pd.DataFrame) -> None:
    with atomic_file_path(path) as tmp_path:
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
            ).to_csv(tmp_path, index=False)
            return
        peak_debug_df.to_csv(tmp_path, index=False)


def export_peak_debug_xlsx(path: str, peak_debug_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None) -> None:
    with atomic_file_path(path) as tmp_path:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
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
