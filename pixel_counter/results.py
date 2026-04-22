from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


@dataclass
class AFCReviewItem:
    segment_name: str
    segment_index: int
    main_peak_index: int
    main_peak_time_s: float
    main_peak_amp: float
    window_start_s: float
    window_end_s: float
    lower_line: float
    upper_line: float
    auto_candidate_times_s: list[float] = field(default_factory=list)
    auto_candidate_amps: list[float] = field(default_factory=list)
    status: str = "pending"

    @property
    def review_id(self) -> str:
        return f"{self.segment_index}:{self.main_peak_index}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCReviewItem":
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            main_peak_index=int(data.get("main_peak_index", -1)),
            main_peak_time_s=float(data.get("main_peak_time_s", np.nan)),
            main_peak_amp=float(data.get("main_peak_amp", np.nan)),
            window_start_s=float(data.get("window_start_s", np.nan)),
            window_end_s=float(data.get("window_end_s", np.nan)),
            lower_line=float(data.get("lower_line", np.nan)),
            upper_line=float(data.get("upper_line", np.nan)),
            auto_candidate_times_s=[float(x) for x in data.get("auto_candidate_times_s", [])],
            auto_candidate_amps=[float(x) for x in data.get("auto_candidate_amps", [])],
            status=str(data.get("status", "pending")),
        )


@dataclass
class AFCEvent:
    segment_name: str
    segment_index: int
    main_peak_index: int
    time_s: float
    amplitude: float
    source: str
    prominence: float = np.nan
    width_s: float = np.nan
    review_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCEvent":
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            main_peak_index=int(data.get("main_peak_index", -1)),
            time_s=float(data.get("time_s", np.nan)),
            amplitude=float(data.get("amplitude", np.nan)),
            prominence=float(data.get("prominence", np.nan)),
            width_s=float(data.get("width_s", np.nan)),
            source=str(data.get("source", "")),
            review_id=str(data["review_id"]) if data.get("review_id") is not None else None,
        )


@dataclass
class AFCReviewDecision:
    segment_name: str
    segment_index: int
    main_peak_index: int
    lower_line: float
    upper_line: float
    window_start_s: float
    window_end_s: float
    accepted_times_s: list[float] = field(default_factory=list)
    rejected_times_s: list[float] = field(default_factory=list)
    manual_added_times_s: list[float] = field(default_factory=list)
    notes: str = ""
    status: str = "saved"

    @property
    def review_id(self) -> str:
        return f"{self.segment_index}:{self.main_peak_index}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCReviewDecision":
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            main_peak_index=int(data.get("main_peak_index", -1)),
            lower_line=float(data.get("lower_line", np.nan)),
            upper_line=float(data.get("upper_line", np.nan)),
            window_start_s=float(data.get("window_start_s", np.nan)),
            window_end_s=float(data.get("window_end_s", np.nan)),
            accepted_times_s=[float(x) for x in data.get("accepted_times_s", [])],
            rejected_times_s=[float(x) for x in data.get("rejected_times_s", [])],
            manual_added_times_s=[float(x) for x in data.get("manual_added_times_s", [])],
            notes=str(data.get("notes", "")),
            status=str(data.get("status", "saved")),
        )


@dataclass
class AFCSegmentReviewItem:
    segment_name: str
    segment_index: int
    afc_lower_left_value: float
    afc_lower_right_value: float
    afc_upper_left_value: float
    afc_upper_right_value: float
    x_start_s: float
    x_end_s: float
    status: str = "pending"
    main_peak_times_s: list[float] = field(default_factory=list)
    main_peak_amps: list[float] = field(default_factory=list)
    rescue_peak_times_s: list[float] = field(default_factory=list)
    rescue_peak_amps: list[float] = field(default_factory=list)
    helper_candidate_times_s: list[float] = field(default_factory=list)
    helper_candidate_amps: list[float] = field(default_factory=list)
    manual_afc_times_s: list[float] = field(default_factory=list)
    manual_afc_amps: list[float] = field(default_factory=list)
    notes: str = ""

    # Legacy compatibility fields kept to load old sessions safely.
    auto_candidate_times_s: list[float] = field(default_factory=list)
    auto_candidate_amps: list[float] = field(default_factory=list)
    accepted_times_s: list[float] = field(default_factory=list)
    accepted_amps: list[float] = field(default_factory=list)
    manual_added_times_s: list[float] = field(default_factory=list)
    manual_added_amps: list[float] = field(default_factory=list)
    rejected_times_s: list[float] = field(default_factory=list)
    rejected_amps: list[float] = field(default_factory=list)

    @property
    def review_id(self) -> str:
        return f"{self.segment_index}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCSegmentReviewItem":
        if "afc_lower_left_value" not in data:
            data = dict(data)
            data["afc_lower_left_value"] = data.get("afc_left_value", data.get("lower_line", np.nan))
        if "afc_lower_right_value" not in data:
            data = dict(data)
            data["afc_lower_right_value"] = data.get("afc_right_value", data.get("upper_line", np.nan))
        if "afc_upper_left_value" not in data:
            data = dict(data)
            data["afc_upper_left_value"] = data.get("afc_upper_cap", data.get("upper_line", np.nan))
        if "afc_upper_right_value" not in data:
            data = dict(data)
            data["afc_upper_right_value"] = data.get("afc_upper_cap", data.get("upper_line", np.nan))
        manual_times = data.get(
            "manual_afc_times_s",
            data.get("manual_added_times_s", data.get("accepted_times_s", data.get("manually_added_afc_times_s", []))),
        )
        manual_amps = data.get(
            "manual_afc_amps",
            data.get("manual_added_amps", data.get("accepted_amps", [])),
        )
        helper_times = data.get("helper_candidate_times_s", data.get("auto_candidate_times_s", []))
        helper_amps = data.get("helper_candidate_amps", data.get("auto_candidate_amps", []))
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            afc_lower_left_value=float(data.get("afc_lower_left_value", np.nan)),
            afc_lower_right_value=float(data.get("afc_lower_right_value", np.nan)),
            afc_upper_left_value=float(data.get("afc_upper_left_value", np.nan)),
            afc_upper_right_value=float(data.get("afc_upper_right_value", np.nan)),
            x_start_s=float(data.get("x_start_s", np.nan)),
            x_end_s=float(data.get("x_end_s", np.nan)),
            status=str(data.get("status", "pending")),
            main_peak_times_s=[float(x) for x in data.get("main_peak_times_s", [])],
            main_peak_amps=[float(x) for x in data.get("main_peak_amps", [])],
            rescue_peak_times_s=[float(x) for x in data.get("rescue_peak_times_s", [])],
            rescue_peak_amps=[float(x) for x in data.get("rescue_peak_amps", [])],
            helper_candidate_times_s=[float(x) for x in (helper_times or [])],
            helper_candidate_amps=[float(x) for x in (helper_amps or [])],
            manual_afc_times_s=[float(x) for x in (manual_times or [])],
            manual_afc_amps=[float(x) for x in (manual_amps or [])],
            notes=str(data.get("notes", "")),
            auto_candidate_times_s=[float(x) for x in data.get("auto_candidate_times_s", [])],
            auto_candidate_amps=[float(x) for x in data.get("auto_candidate_amps", [])],
            accepted_times_s=[float(x) for x in data.get("accepted_times_s", data.get("accepted_afc_times_s", []))],
            accepted_amps=[float(x) for x in data.get("accepted_amps", [])],
            manual_added_times_s=[float(x) for x in data.get("manual_added_times_s", data.get("manually_added_afc_times_s", []))],
            manual_added_amps=[float(x) for x in data.get("manual_added_amps", [])],
            rejected_times_s=[float(x) for x in data.get("rejected_times_s", data.get("rejected_afc_times_s", []))],
            rejected_amps=[float(x) for x in data.get("rejected_amps", [])],
        )

    @classmethod
    def from_legacy_mainpeak_item(cls, data: Dict) -> "AFCSegmentReviewItem":
        main_t = float(data.get("main_peak_time_s", np.nan))
        main_a = float(data.get("main_peak_amp", np.nan))
        main_times = [main_t] if np.isfinite(main_t) else []
        main_amps = [main_a] if np.isfinite(main_a) else []
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            afc_lower_left_value=float(data.get("lower_line", np.nan)),
            afc_lower_right_value=float(data.get("upper_line", np.nan)),
            afc_upper_left_value=float(data.get("upper_line", np.nan)),
            afc_upper_right_value=float(data.get("upper_line", np.nan)),
            x_start_s=float(data.get("window_start_s", np.nan)),
            x_end_s=float(data.get("window_end_s", np.nan)),
            main_peak_times_s=main_times,
            main_peak_amps=main_amps,
            helper_candidate_times_s=[float(x) for x in data.get("auto_candidate_times_s", [])],
            helper_candidate_amps=[float(x) for x in data.get("auto_candidate_amps", [])],
            manual_afc_times_s=[float(x) for x in data.get("manual_added_times_s", [])],
            manual_afc_amps=[float(x) for x in data.get("manual_added_amps", [])],
            notes=str(data.get("notes", "")),
            auto_candidate_times_s=[float(x) for x in data.get("auto_candidate_times_s", [])],
            auto_candidate_amps=[float(x) for x in data.get("auto_candidate_amps", [])],
            accepted_times_s=[],
            accepted_amps=[],
            manual_added_times_s=[],
            manual_added_amps=[],
            rejected_times_s=[],
            rejected_amps=[],
            status=str(data.get("status", "pending")),
        )


@dataclass
class AFCSegmentReviewDecision:
    segment_name: str
    segment_index: int
    afc_lower_left_value: float
    afc_lower_right_value: float
    afc_upper_left_value: float
    afc_upper_right_value: float
    x_start_s: float
    x_end_s: float
    manual_afc_times_s: list[float] = field(default_factory=list)
    manual_afc_amps: list[float] = field(default_factory=list)
    notes: str = ""
    status: str = "saved"

    # Legacy compatibility fields kept to load old sessions safely.
    accepted_times_s: list[float] = field(default_factory=list)
    accepted_amps: list[float] = field(default_factory=list)
    rejected_times_s: list[float] = field(default_factory=list)
    rejected_amps: list[float] = field(default_factory=list)
    manual_added_times_s: list[float] = field(default_factory=list)
    manual_added_amps: list[float] = field(default_factory=list)

    @property
    def review_id(self) -> str:
        return f"{self.segment_index}"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCSegmentReviewDecision":
        if "afc_lower_left_value" not in data:
            data = dict(data)
            data["afc_lower_left_value"] = data.get("afc_left_value", data.get("lower_line", np.nan))
        if "afc_lower_right_value" not in data:
            data = dict(data)
            data["afc_lower_right_value"] = data.get("afc_right_value", data.get("upper_line", np.nan))
        if "afc_upper_left_value" not in data:
            data = dict(data)
            data["afc_upper_left_value"] = data.get("afc_upper_cap", data.get("upper_line", np.nan))
        if "afc_upper_right_value" not in data:
            data = dict(data)
            data["afc_upper_right_value"] = data.get("afc_upper_cap", data.get("upper_line", np.nan))
        manual_times = data.get("manual_afc_times_s", data.get("manual_added_times_s", data.get("accepted_times_s", [])))
        manual_amps = data.get("manual_afc_amps", data.get("manual_added_amps", data.get("accepted_amps", [])))
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            afc_lower_left_value=float(data.get("afc_lower_left_value", np.nan)),
            afc_lower_right_value=float(data.get("afc_lower_right_value", np.nan)),
            afc_upper_left_value=float(data.get("afc_upper_left_value", np.nan)),
            afc_upper_right_value=float(data.get("afc_upper_right_value", np.nan)),
            x_start_s=float(data.get("x_start_s", np.nan)),
            x_end_s=float(data.get("x_end_s", np.nan)),
            manual_afc_times_s=[float(x) for x in (manual_times or [])],
            manual_afc_amps=[float(x) for x in (manual_amps or [])],
            notes=str(data.get("notes", "")),
            status=str(data.get("status", "saved")),
            accepted_times_s=[float(x) for x in data.get("accepted_times_s", [])],
            accepted_amps=[float(x) for x in data.get("accepted_amps", [])],
            rejected_times_s=[float(x) for x in data.get("rejected_times_s", [])],
            rejected_amps=[float(x) for x in data.get("rejected_amps", [])],
            manual_added_times_s=[float(x) for x in data.get("manual_added_times_s", [])],
            manual_added_amps=[float(x) for x in data.get("manual_added_amps", [])],
        )

    @classmethod
    def from_legacy_mainpeak_decision(cls, data: Dict) -> "AFCSegmentReviewDecision":
        return cls(
            segment_name=str(data.get("segment_name", "")),
            segment_index=int(data.get("segment_index", -1)),
            afc_lower_left_value=float(data.get("lower_line", np.nan)),
            afc_lower_right_value=float(data.get("upper_line", np.nan)),
            afc_upper_left_value=float(data.get("upper_line", np.nan)),
            afc_upper_right_value=float(data.get("upper_line", np.nan)),
            x_start_s=float(data.get("window_start_s", np.nan)),
            x_end_s=float(data.get("window_end_s", np.nan)),
            manual_afc_times_s=[float(x) for x in data.get("manual_added_times_s", data.get("accepted_times_s", []))],
            manual_afc_amps=[float(x) for x in data.get("manual_added_amps", data.get("accepted_amps", []))],
            notes=str(data.get("notes", "")),
            status=str(data.get("status", "saved")),
            accepted_times_s=[float(x) for x in data.get("accepted_times_s", [])],
            accepted_amps=[float(x) for x in data.get("accepted_amps", [])],
            rejected_times_s=[float(x) for x in data.get("rejected_times_s", [])],
            rejected_amps=[float(x) for x in data.get("rejected_amps", [])],
            manual_added_times_s=[float(x) for x in data.get("manual_added_times_s", [])],
            manual_added_amps=[float(x) for x in data.get("manual_added_amps", [])],
        )


@dataclass
class AFCReviewSession:
    input_workbook: str
    created_at: str
    updated_at: str
    items: list[AFCSegmentReviewItem] = field(default_factory=list)
    decisions: list[AFCSegmentReviewDecision] = field(default_factory=list)
    schema_version: str = "afc_segment_manual_v3"

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "input_workbook": self.input_workbook,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "items": [x.to_dict() for x in self.items],
            "decisions": [x.to_dict() for x in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AFCReviewSession":
        items_raw = data.get("items", [])
        decisions_raw = data.get("decisions", [])
        items: List[AFCSegmentReviewItem] = []
        decisions: List[AFCSegmentReviewDecision] = []
        for x in items_raw:
            if "x_start_s" in x or "main_peak_times_s" in x:
                items.append(AFCSegmentReviewItem.from_dict(x))
            else:
                items.append(AFCSegmentReviewItem.from_legacy_mainpeak_item(x))
        for x in decisions_raw:
            if "x_start_s" in x:
                decisions.append(AFCSegmentReviewDecision.from_dict(x))
            else:
                decisions.append(AFCSegmentReviewDecision.from_legacy_mainpeak_decision(x))
        return cls(
            input_workbook=str(data.get("input_workbook", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            items=items,
            decisions=decisions,
            schema_version=str(data.get("schema_version", "afc_segment_manual_v3")),
        )


def _format_events_df_for_report(events_df: pd.DataFrame, max_rows: int = 120) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=["Time_s", "Amp", "Prom", "Width_s", "Transient"])
    out = events_df.copy()
    cols = ["Time_s", "Amp", "Prom", "Width_s", "Transient"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    if "Time_s" in out.columns:
        out["Time_s"] = pd.to_numeric(out["Time_s"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    for c in ["Amp", "Prom", "Width_s"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{float(x):.5f}")
    if "Transient" in out.columns:
        out["Transient"] = pd.to_numeric(out["Transient"], errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(x)))
    return out.head(int(max_rows)).reset_index(drop=True)

def _format_rescue_df_for_report(
    rescue_data: Union[Sequence[float], Sequence[Dict], pd.DataFrame],
    max_rows: int = 80,
) -> pd.DataFrame:
    cols = ["Time_s", "Amp", "Prom", "Width_s"]
    if rescue_data is None:
        return pd.DataFrame(columns=cols)

    if isinstance(rescue_data, pd.DataFrame):
        out = rescue_data.copy()
    else:
        try:
            data_list = list(rescue_data)
        except Exception:
            data_list = []
        if not data_list:
            return pd.DataFrame(columns=cols)
        if isinstance(data_list[0], dict):
            out = pd.DataFrame(data_list)
        else:
            vals: List[float] = []
            for x in data_list:
                try:
                    xf = float(x)
                except Exception:
                    continue
                if np.isfinite(xf):
                    vals.append(float(xf))
            if not vals:
                return pd.DataFrame(columns=cols)
            out = pd.DataFrame({"Time_s": sorted(vals)})

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].head(int(max_rows)).copy()
    out["Time_s"] = pd.to_numeric(out["Time_s"], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    for c in ["Amp", "Prom", "Width_s"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.5f}")
    return out.reset_index(drop=True)

def _make_summary_dataframe(segment_results: List[Dict], stim_hz: float, recording_s: float) -> pd.DataFrame:
    rows: List[Dict] = []
    expected = float(stim_hz * recording_s) if stim_hz > 0 and recording_s > 0 else np.nan
    for item in segment_results:
        meta = item["meta"]
        n_total = int(item["n_main"])
        n_primary = int(item.get("n_main_primary", n_total))
        n_rescue = int(item.get("n_rescue", 0))
        bpm_user_exact = float((n_total / recording_s) * 60.0) if recording_s > 0 else 0.0
        row = {
            "Segment": item["sheet_name"],
            "QC": "PASS" if bool(meta.get("qc_pass", False)) else "REJECT",
            "QC reason": str(meta.get("qc_reason", "unknown")),
            "BPM (using user duration)": float(bpm_user_exact),
            # Explicit, unambiguous breakdown for human-readable report tables.
            "Primary main beats": n_primary,
            "Rescue peaks": n_rescue,
            "Total detected events": n_total,
            # Explicit machine-friendly columns for downstream filtering/pivoting in XLSX.
            "primary_main_beats": n_primary,
            "rescue_peaks": n_rescue,
            "total_detected_events": n_total,
            # Backward-compatibility legacy name; equal to total detected events.
            "Main beats": n_total,
        }
        if stim_hz > 0 and np.isfinite(expected):
            ratio = float(n_total / expected) if expected > 0 else np.nan
            row["Expected beats (Hz x seconds)"] = float(expected)
            row["Detected Total/Expected ratio"] = ratio
            row["expected_beats"] = float(expected)
            row["detected_total_expected_ratio"] = ratio
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "Segment",
                "QC",
                "QC reason",
                "BPM (using user duration)",
                "Primary main beats",
                "Rescue peaks",
                "Total detected events",
                "primary_main_beats",
                "rescue_peaks",
                "total_detected_events",
                "Main beats",
                "Expected beats (Hz x seconds)",
                "Detected Total/Expected ratio",
                "expected_beats",
                "detected_total_expected_ratio",
            ]
        )
    return pd.DataFrame(rows)
