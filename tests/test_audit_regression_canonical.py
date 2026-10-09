import pytest
import numpy as np
from pixel_counter.analysis import build_events_dataframe, _build_peak_debug_rows

@pytest.mark.xfail(strict=True, reason="Missing unified event registry. Rescue events missing from debug; duplicate candidate IDs present.")
def test_canonical_event_identity():
    """
    Test that every candidate is uniquely identified, and one candidate in both main and rescue
    does not duplicate in the accepted events dataframe. Also tests that rescue-only
    candidates appear in the overall debug rows.
    """
    time = np.array([0.1, 0.2, 0.3])
    sig = np.array([1.0, 2.0, 1.0])

    # Simulate a candidate present in both main and rescue
    # Candidate index = 1
    main_peaks = np.array([1])
    main_proms = np.array([1.0])
    main_widths = np.array([0.1])
    main_tids = np.array([0])

    rescue_peaks = np.array([1])
    rescue_audit = {1: {"prominence": 1.0, "width_s": 0.1, "rescue_type": "gap_fill"}}

    # 1. Test build_events_dataframe for duplicate generation
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

    # We expect 1 unique accepted event, not 2.
    # Currently it creates 2 rows with the same event_id.
    assert len(df) == 1, f"Expected 1 unique event row, got {len(df)}"
    assert df["event_id"].nunique() == len(df), "Event IDs are not unique across rows"

    # 2. Test _build_peak_debug_rows for missing rescue events
    # Simulate candidate 1 as main, candidate 2 as rescue.
    rows = _build_peak_debug_rows(
        segment_name="TestSeg",
        segment_index=42,
        sample_id="ID123",
        time=time,
        sig=sig,
        raw_peaks_all=np.array([1, 2]),
        raw_proms_all=np.array([1.0, 0.5]),
        raw_widths_all=np.array([0.1, 0.05]),
        raw_tids_all=np.array([0, 0]),
        raw_survivors=set([1, 2]),
        main_candidates=set([1]), # only 1 went to main
        main_after_dedup=set([1]),
        main_after_short_gap=set([1]),
        main_after_local=set([1]),
        main_after_interbeat=set([1]),
        final_main=set([1]),
        rescue=set([2]), # 2 is rescue only
        removed_by_rescue=set(),
        promoted_gap=set([2]),
        promoted_transient=set(),
        rejected_in_stitched_gap=set(),
        strong_thr=0.8
    )

    # We expect 2 debug rows (one for candidate 1, one for candidate 2).
    # Currently _build_peak_debug_rows only loops over main_candidates, so it yields 1.
    assert len(rows) == 2, f"Expected 2 debug rows, got {len(rows)}"
