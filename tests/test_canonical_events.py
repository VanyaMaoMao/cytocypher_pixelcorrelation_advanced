import pytest
import numpy as np
import pandas as pd
from pixel_counter.analysis import build_events_dataframe, _build_peak_debug_rows

def test_canonical_fields_in_events_dataframe():
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])
    main_peaks = np.array([1])
    main_proms = np.array([1.0])
    main_widths = np.array([0.1])
    main_tids = np.array([0])
    
    rescue_peaks = np.array([0])
    rescue_audit = {0: {"prominence": 0.5, "width_s": 0.05, "rescue_type": "gap_fill"}}
    
    df = build_events_dataframe(
        time=time,
        sig=sig,
        main_peaks=main_peaks,
        main_proms=main_proms,
        main_widths=main_widths,
        main_tids=main_tids,
        segment_name="TestSeg",
        segment_index=42,
        sample_id="ID123",
        rescue_peaks=rescue_peaks,
        rescue_audit_by_peak=rescue_audit
    )
    
    assert "event_id" in df.columns
    assert "segment_name" in df.columns
    assert "segment_index" in df.columns
    assert "sample_id" in df.columns
    assert "candidate_index" in df.columns
    assert "detection_source" in df.columns
    assert "decision_status" in df.columns
    assert "decision_reasons" in df.columns
    
    assert len(df) == 2
    
    # Check that rescue and main beats are correctly formatted
    main_evt = df[df["Type"] == "Main Beat"].iloc[0]
    assert main_evt["segment_name"] == "TestSeg"
    assert main_evt["segment_index"] == 42
    assert main_evt["sample_id"] == "ID123"
    assert main_evt["detection_source"] == "main_peak_pipeline"
    assert main_evt["decision_status"] == "accepted"
    
    res_evt = df[df["Type"] == "Rescue"].iloc[0]
    assert res_evt["segment_name"] == "TestSeg"
    assert res_evt["segment_index"] == 42
    assert res_evt["sample_id"] == "ID123"
    assert res_evt["detection_source"] == "rescue_pipeline"
    assert res_evt["decision_status"] == "accepted"
    assert res_evt["decision_reasons"] == "gap_fill"


def test_canonical_fields_in_peak_debug_rows():
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])
    
    rows = _build_peak_debug_rows(
        segment_name="TestSeg",
        segment_index=42,
        sample_id="ID123",
        time=time,
        sig=sig,
        raw_peaks_all=np.array([1]),
        raw_proms_all=np.array([1.0]),
        raw_widths_all=np.array([0.1]),
        raw_tids_all=np.array([0]),
        raw_survivors=set([1]),
        main_candidates=set([1]),
        main_after_dedup=set([1]),
        main_after_short_gap=set([1]),
        main_after_local=set([1]),
        main_after_interbeat=set([1]),
        final_main=set([1]),
        rescue=set(),
        removed_by_rescue=set(),
        promoted_gap=set(),
        promoted_transient=set(),
        rejected_in_stitched_gap=set(),
        strong_thr=0.8
    )
    
    assert len(rows) == 1
    row = rows[0]
    
    assert "candidate_id" in row
    assert "segment_name" in row
    assert "segment_index" in row
    assert "sample_id" in row
    assert "candidate_index" in row
    
    assert row["segment_name"] == "TestSeg"
    assert row["segment_index"] == 42
    assert row["sample_id"] == "ID123"
