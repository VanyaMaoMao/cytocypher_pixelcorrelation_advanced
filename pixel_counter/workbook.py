from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .afc_review import launch_afc_review_session
from .analysis import (
    build_afc_segment_review_items,
    count_main_beats_from_excel,
    merge_afc_segment_decisions_with_results,
)
from .config import AFCReviewConfig, BeatCounterConfig
from .io_utils import (
    export_afc_events_csv,
    export_afc_review_log_csv,
    export_peak_debug_csv,
    export_peak_debug_xlsx,
    save_afc_review_session_json,
)
from .plotting import plot_events
from .reporting import build_arrhythmia_summary_workbook, build_raw_cytocypher_docx_report
from .results import _make_summary_dataframe

logger = logging.getLogger(__name__)


def list_supported_raw_segment_sheets(file_path: Union[str, pd.ExcelFile]) -> List[Tuple[str, int, str]]:
    xl = file_path if isinstance(file_path, pd.ExcelFile) else pd.ExcelFile(file_path)
    pat = re.compile(r"^(PixelCorrelation)\s+Segment\s+(\d+)$")
    out = []
    for s in xl.sheet_names:
        m = pat.match(str(s).strip())
        if m:
            out.append((m.group(1), int(m.group(2)), s))
    out.sort(key=lambda x: x[1])
    return out


def _run_auto_segment_analysis(
    *,
    raw_xlsx_path: str,
    config: BeatCounterConfig,
    diagnostics_dir: Optional[str],
    diagnostics_segments: Optional[Sequence[int]],
    debug: bool,
    show_plots: bool,
    need_docx_plot_png: bool,
) -> List[Dict]:
    xl = pd.ExcelFile(raw_xlsx_path)
    sheets = list_supported_raw_segment_sheets(xl)
    if not sheets:
        raise ValueError("No PixelCorrelation Segment sheets found.")

    diag_set = set(int(x) for x in diagnostics_segments) if diagnostics_segments is not None else None
    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)

    segment_results: List[Dict] = []
    for _, seg_no, sheet_name in sheets:
        try:
            bpm_file, n_main, events_df, meta = count_main_beats_from_excel(
                raw_xlsx_path,
                sheet_name=sheet_name,
                config=config,
                show_plot=False,
                debug=debug,
                xl=xl,
            )
        except Exception as exc:
            logger.exception("Failed on %s in %s", sheet_name, raw_xlsx_path)
            bpm_file = 0.0
            n_main = 0
            events_df = pd.DataFrame(columns=["Time_s", "Type", "Amp", "Prom", "Width_s", "Transient"])
            meta = {
                "qc_pass": False,
                "qc_reason": f"runtime_error_{type(exc).__name__}",
                "snr": np.nan,
                "strong_thr": np.nan,
                "weak_thr": np.nan,
                "n_main_rescue": 0,
                "n_main_primary": 0,
            }
        meta = dict(meta or {})
        sample_id = meta.get("sample_id", np.nan)
        meta["sample_id"] = sample_id

        result = {
            "sheet_name": sheet_name,
            "sample_id": sample_id,
            "segment_no": int(seg_no),
            "bpm_file": float(bpm_file),
            "n_main": int(n_main),
            "n_rescue": int(meta.get("n_main_rescue", 0)),
            "n_main_primary": int(meta.get("n_main_primary", n_main)),
            "events": events_df,
            "meta": meta,
        }

        need_plot_object = bool(need_docx_plot_png or diagnostics_dir or show_plots)
        plot_generation_s = 0.0
        if need_plot_object and "_sig_plot" in meta and "_time_plot" in meta:
            t_plot = perf_counter()
            fig = plot_events(
                time=np.asarray(meta["_time_plot"], dtype=float),
                sig=np.asarray(meta["_sig_plot"], dtype=float),
                events=events_df,
                file_name=sheet_name,
                bpm=float(bpm_file),
                n_main=int(n_main),
                n_rescue=int(meta.get("n_main_rescue", 0)),
                rescue_peaks=meta.get("_rescue_peaks_plot", []),
                snr=float(meta.get("snr", np.nan)),
                strong_thr=float(meta.get("strong_thr", np.nan)),
                weak_thr=float(meta.get("weak_thr", np.nan)),
                config=config,
                show=show_plots,
                return_fig=True,
                meta=meta,
            )
            if fig is not None:
                if need_docx_plot_png:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                    result["plot_png"] = buf.getvalue()
                if diagnostics_dir and (diag_set is None or int(seg_no) in diag_set):
                    diag_png = os.path.join(diagnostics_dir, f"segment_{int(seg_no):03d}.png")
                    fig.savefig(diag_png, dpi=180, bbox_inches="tight")
                plt.close(fig)
            plot_generation_s = float(perf_counter() - t_plot)
        if debug and need_plot_object:
            timing_out = dict(meta.get("timing", {}) if isinstance(meta.get("timing"), dict) else {})
            timing_out["report_plot_generation_s"] = float(plot_generation_s)
            meta["timing"] = timing_out

        segment_results.append(result)
    return segment_results


def analyze_workbook_auto_only(
    *,
    raw_xlsx_path: str,
    stim_hz: float,
    recording_s: float,
    config: BeatCounterConfig = BeatCounterConfig(),
    output_docx: Optional[str] = None,
    output_summary_xlsx: Optional[str] = None,
    diagnostics_dir: Optional[str] = None,
    diagnostics_segments: Optional[Sequence[int]] = None,
    debug: bool = False,
    debug_peak_trace: bool = False,
    show_plots: bool = False,
    return_segment_results: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict]]]:
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    if output_docx is None:
        output_docx = raw_xlsx_path.replace(".xlsx", "_arrhythmia_report.docx")
    if output_summary_xlsx is None:
        output_summary_xlsx = raw_xlsx_path.replace(".xlsx", "_arrhythmia_summary.xlsx")

    segment_results = _run_auto_segment_analysis(
        raw_xlsx_path=raw_xlsx_path,
        config=config,
        diagnostics_dir=diagnostics_dir,
        diagnostics_segments=diagnostics_segments,
        debug=debug,
        show_plots=show_plots,
        need_docx_plot_png=bool(output_docx),
    )

    summary_df = _make_summary_dataframe(segment_results, stim_hz=stim_hz, recording_s=recording_s)
    summary_df.to_excel(output_summary_xlsx, index=False)
    if diagnostics_dir:
        summary_df.to_excel(os.path.join(diagnostics_dir, "pixelcorr_summary.xlsx"), index=False)

    if output_docx:
        build_raw_cytocypher_docx_report(
            raw_xlsx_path=raw_xlsx_path,
            stim_hz=stim_hz,
            recording_s=recording_s,
            segment_results=segment_results,
            output_docx=output_docx,
        )

    if bool(debug_peak_trace):
        peak_debug_paths = _default_peak_debug_paths(raw_xlsx_path)
        peak_debug_df = _collect_peak_debug_dataframe(segment_results)
        peak_debug_summary_df = _build_peak_debug_summary(peak_debug_df)
        export_peak_debug_csv(peak_debug_paths["csv"], peak_debug_df)
        export_peak_debug_xlsx(peak_debug_paths["xlsx"], peak_debug_df, peak_debug_summary_df)
        logger.info("Peak debug exports written: %s | %s", peak_debug_paths["xlsx"], peak_debug_paths["csv"])

    if return_segment_results:
        return summary_df, segment_results
    return summary_df


def _default_afc_paths(raw_xlsx_path: str) -> Dict[str, str]:
    base = str(Path(raw_xlsx_path).with_suffix(""))
    return {
        "session_json": f"{base}_afc_review_session.json",
        "events_csv": f"{base}_afc_events.csv",
        "review_log_csv": f"{base}_afc_review_log.csv",
        "plots_dir": f"{base}_afc_review_plots",
    }


def _default_peak_debug_paths(raw_xlsx_path: str) -> Dict[str, str]:
    base = str(Path(raw_xlsx_path).with_suffix(""))
    return {
        "xlsx": f"{base}_peak_debug.xlsx",
        "csv": f"{base}_peak_debug.csv",
    }


def _collect_peak_debug_dataframe(segment_results: Sequence[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for seg in segment_results:
        seg_name = str(seg.get("sheet_name", ""))
        seg_idx = int(seg.get("segment_no", -1))
        meta = seg.get("meta", {}) or {}
        seg_rows = meta.get("_peak_debug_rows", []) or []
        if not seg_rows:
            continue
        for row in seg_rows:
            row_out = dict(row)
            row_out["segment_name"] = str(row_out.get("segment_name", seg_name))
            row_out["segment_index"] = int(row_out.get("segment_index", seg_idx))
            rows.append(row_out)
    if not rows:
        return pd.DataFrame(
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
    df = pd.DataFrame(rows)
    if "segment_index" in df.columns:
        df["segment_index"] = pd.to_numeric(df["segment_index"], errors="coerce")
    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    if "peak_index_raw" in df.columns:
        df["peak_index_raw"] = pd.to_numeric(df["peak_index_raw"], errors="coerce")
    if "rejection_reason" not in df.columns:
        df["rejection_reason"] = ""
    df["rejection_reason"] = df["rejection_reason"].fillna("").astype(str)
    if "notes" not in df.columns:
        df["notes"] = ""
    else:
        # Notes column is retained for compatibility but intentionally left blank.
        df["notes"] = ""
    sort_cols = [c for c in ["segment_index", "peak_index_raw", "time_s"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _build_peak_debug_summary(peak_debug_df: pd.DataFrame) -> pd.DataFrame:
    if peak_debug_df is None or peak_debug_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_name",
                "segment_index",
                "raw_peaks",
                "main_candidates",
                "final_main",
                "final_rescue",
                "dropped",
                "raw_only",
            ]
        )
    df = peak_debug_df.copy()
    grp_cols = ["segment_name", "segment_index"]
    out_rows: List[Dict] = []
    for (seg_name, seg_idx), g in df.groupby(grp_cols, dropna=False):
        labels = g["final_label"].astype(str).str.lower() if "final_label" in g.columns else pd.Series([], dtype=str)
        out_rows.append(
            {
                "segment_name": str(seg_name),
                "segment_index": int(seg_idx) if pd.notna(seg_idx) else -1,
                "raw_peaks": int(len(g)),
                "main_candidates": int(pd.to_numeric(g.get("survived_main_candidate_stage", False), errors="coerce").fillna(0).astype(bool).sum()),
                "final_main": int((labels == "main").sum()),
                "final_rescue": int((labels == "rescue").sum()),
                "dropped": int((labels == "dropped").sum()),
                "raw_only": int((labels == "raw_only").sum()),
            }
        )
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["segment_index"]).reset_index(drop=True)
    return out


def analyze_workbook_with_afc_review(
    *,
    raw_xlsx_path: str,
    stim_hz: float,
    recording_s: float,
    config: BeatCounterConfig = BeatCounterConfig(),
    afc_config: AFCReviewConfig = AFCReviewConfig(enabled=True),
    output_docx: Optional[str] = None,
    output_summary_xlsx: Optional[str] = None,
    diagnostics_dir: Optional[str] = None,
    diagnostics_segments: Optional[Sequence[int]] = None,
    debug: bool = False,
    debug_peak_trace: bool = False,
    show_plots: bool = False,
    interactive_review: bool = True,
) -> pd.DataFrame:
    if not bool(afc_config.enabled):
        return analyze_workbook_auto_only(
            raw_xlsx_path=raw_xlsx_path,
            stim_hz=stim_hz,
            recording_s=recording_s,
            config=config,
            output_docx=output_docx,
            output_summary_xlsx=output_summary_xlsx,
            diagnostics_dir=diagnostics_dir,
            diagnostics_segments=diagnostics_segments,
            debug=debug,
            debug_peak_trace=debug_peak_trace,
            show_plots=show_plots,
        )

    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    if output_docx is None:
        output_docx = raw_xlsx_path.replace(".xlsx", "_arrhythmia_report.docx")
    if output_summary_xlsx is None:
        output_summary_xlsx = raw_xlsx_path.replace(".xlsx", "_arrhythmia_summary.xlsx")
    afc_paths = _default_afc_paths(raw_xlsx_path)

    segment_results = _run_auto_segment_analysis(
        raw_xlsx_path=raw_xlsx_path,
        config=config,
        diagnostics_dir=diagnostics_dir,
        diagnostics_segments=diagnostics_segments,
        debug=debug,
        show_plots=show_plots,
        need_docx_plot_png=bool(output_docx),
    )

    # Manual-first AFC workflow: one review item per analyzable segment.
    review_items = build_afc_segment_review_items(segment_results=segment_results, afc_config=afc_config)
    # `afc_config.resume_existing_session` controls fresh-start vs explicit resume behavior.
    session = launch_afc_review_session(
        input_workbook=raw_xlsx_path,
        review_items=review_items,
        segment_results=segment_results,
        afc_config=afc_config,
        session_json_path=afc_paths["session_json"],
        review_plots_dir=afc_paths["plots_dir"] if bool(afc_config.save_review_png) else None,
        interactive=interactive_review,
    )

    merged_results, afc_events, review_log_df = merge_afc_segment_decisions_with_results(segment_results, session.decisions)
    summary_df = _make_summary_dataframe(merged_results, stim_hz=stim_hz, recording_s=recording_s)
    build_arrhythmia_summary_workbook(
        output_xlsx=output_summary_xlsx,
        summary_df=summary_df,
        segment_results=merged_results,
        afc_events=afc_events,
        review_log_df=review_log_df,
    )

    if output_docx:
        build_raw_cytocypher_docx_report(
            raw_xlsx_path=raw_xlsx_path,
            stim_hz=stim_hz,
            recording_s=recording_s,
            segment_results=merged_results,
            output_docx=output_docx,
            afc_events=afc_events,
            review_log_df=review_log_df,
            afc_review_plots_dir=afc_paths["plots_dir"] if bool(afc_config.save_review_png) else None,
        )

    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)
        build_arrhythmia_summary_workbook(
            output_xlsx=os.path.join(diagnostics_dir, "pixelcorr_summary.xlsx"),
            summary_df=summary_df,
            segment_results=merged_results,
            afc_events=afc_events,
            review_log_df=review_log_df,
        )

    if bool(afc_config.save_review_json):
        save_afc_review_session_json(afc_paths["session_json"], session)
    if bool(afc_config.save_review_csv):
        export_afc_events_csv(afc_paths["events_csv"], afc_events)
        export_afc_review_log_csv(afc_paths["review_log_csv"], session.decisions)

    if bool(debug_peak_trace):
        peak_debug_paths = _default_peak_debug_paths(raw_xlsx_path)
        peak_debug_df = _collect_peak_debug_dataframe(merged_results)
        peak_debug_summary_df = _build_peak_debug_summary(peak_debug_df)
        export_peak_debug_csv(peak_debug_paths["csv"], peak_debug_df)
        export_peak_debug_xlsx(peak_debug_paths["xlsx"], peak_debug_df, peak_debug_summary_df)
        logger.info("Peak debug exports written: %s | %s", peak_debug_paths["xlsx"], peak_debug_paths["csv"])

    return summary_df


def analyze_raw_cytocypher_workbook(
    *,
    raw_xlsx_path: str,
    stim_hz: float,
    recording_s: float,
    config: BeatCounterConfig = BeatCounterConfig(),
    output_docx: Optional[str] = None,
    output_summary_xlsx: Optional[str] = None,
    diagnostics_dir: Optional[str] = None,
    diagnostics_segments: Optional[Sequence[int]] = None,
    debug: bool = False,
    debug_peak_trace: bool = False,
    show_plots: bool = False,
) -> pd.DataFrame:
    return analyze_workbook_auto_only(
        raw_xlsx_path=raw_xlsx_path,
        stim_hz=stim_hz,
        recording_s=recording_s,
        config=config,
        output_docx=output_docx,
        output_summary_xlsx=output_summary_xlsx,
        diagnostics_dir=diagnostics_dir,
        diagnostics_segments=diagnostics_segments,
        debug=debug,
        debug_peak_trace=debug_peak_trace,
        show_plots=show_plots,
    )
