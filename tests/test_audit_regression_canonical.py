import pytest
import numpy as np
from pixel_counter.analysis import build_events_dataframe, _build_peak_debug_rows

@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Duplicate candidates when in both main and rescue.")
def test_canonical_event_duplicate_identity():
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])

    # Candidate index 1 is in both main and rescue
    main_peaks = np.array([1])
    main_proms = np.array([1.0])
    main_widths = np.array([0.1])
    main_tids = np.array([0])

    rescue_peaks = np.array([1])
    rescue_audit = {1: {"prominence": 1.0, "width_s": 0.1, "rescue_type": "gap_fill"}}

    df = build_events_dataframe(
        time=time, sig=sig, main_peaks=main_peaks, main_proms=main_proms,
        main_widths=main_widths, main_tids=main_tids, segment_name="TestSeg",
        segment_index=42, sample_id="ID123", rescue_peaks=rescue_peaks,
        rescue_audit_by_peak=rescue_audit
    )

    assert len(df) == 1, f"Expected 1 unique event row, got {len(df)}"
    assert df["event_id"].nunique() == len(df), "Event IDs are not unique across rows"


def test_canonical_event_missing_rescue_audit():
    """
    Test whether rescue-only events make it into the debug rows.
    Actually this passed, so it appears rescue items ARE included.
    We just assert it directly.
    """
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])

    # Candidate 2 is rescue only
    rows = _build_peak_debug_rows(
        segment_name="TestSeg", segment_index=42, sample_id="ID123",
        time=time, sig=sig, raw_peaks_all=np.array([1, 2]),
        raw_proms_all=np.array([1.0, 0.5]), raw_widths_all=np.array([0.1, 0.05]),
        raw_tids_all=np.array([0, 0]), raw_survivors=set([1, 2]),
        main_candidates=set([1]), main_after_dedup=set([1]),
        main_after_short_gap=set([1]), main_after_local=set([1]),
        main_after_interbeat=set([1]), final_main=set([1]),
        rescue=set([2]), removed_by_rescue=set(),
        promoted_gap=set([2]), promoted_transient=set(),
        rejected_in_stitched_gap=set(), strong_thr=0.8
    )

    assert len(rows) == 2, f"Expected 2 debug rows, got {len(rows)}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Rejected preliminary events incorrectly listed as final accepted.")
def test_canonical_event_rejected_preliminary():
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])

    df = build_events_dataframe(
        time=time, sig=sig, main_peaks=np.array([1]), main_proms=np.array([1.0]),
        main_widths=np.array([0.1]), main_tids=np.array([0]), segment_name="TestSeg",
        segment_index=42, sample_id="ID123", rescue_peaks=np.array([]),
        rescue_audit_by_peak={}
    )
    assert "segment_status" in df.columns, "Expected segment_status column to indicate rejection"


def test_canonical_event_sample_id_propagation():
    """
    Test that Sample ID is populated in the events dataframe.
    """
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])
    df = build_events_dataframe(
        time=time, sig=sig, main_peaks=np.array([1]), main_proms=np.array([1.0]),
        main_widths=np.array([0.1]), main_tids=np.array([0]), segment_name="TestSeg",
        segment_index=42, sample_id="ID123", rescue_peaks=np.array([]),
        rescue_audit_by_peak={}
    )

    assert "sample_id" in df.columns
    assert df["sample_id"].iloc[0] == "ID123"
