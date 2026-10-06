import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from pixel_counter import (
    analyze_raw_cytocypher_workbook,
    analyze_workbook_with_afc_review,
    BeatCounterConfig,
    AFCReviewConfig
)


@patch("pixel_counter.workbook.analyze_workbook_auto_only")
def test_analyze_raw_cytocypher_workbook_mock(mock_auto):
    dummy_df = pd.DataFrame([{"segment_index": 1, "test": "val"}])
    mock_auto.return_value = dummy_df
    
    res = analyze_raw_cytocypher_workbook(
        raw_xlsx_path="dummy.xlsx",
        stim_hz=1.0,
        recording_s=10.0,
    )
    assert res.equals(dummy_df)


@patch("pixel_counter.workbook._run_auto_segment_analysis")
@patch("pixel_counter.workbook.build_afc_segment_review_items")
@patch("pixel_counter.workbook.launch_afc_review_session")
@patch("pixel_counter.workbook.merge_afc_segment_decisions_with_results")
@patch("pixel_counter.workbook._make_summary_dataframe")
@patch("pixel_counter.workbook.build_arrhythmia_summary_workbook")
@patch("pixel_counter.workbook.build_raw_cytocypher_docx_report")
@patch("pixel_counter.workbook.save_afc_review_session_json")
@patch("pixel_counter.workbook.export_afc_events_csv")
@patch("pixel_counter.workbook.export_afc_review_log_csv")
@patch("pixel_counter.workbook.os.makedirs")
def test_analyze_workbook_with_afc_review_mock(
    mock_makedirs,
    mock_export_log, mock_export_events, mock_save_session,
    mock_build_docx, mock_build_xlsx,
    mock_make_summary, mock_merge, mock_launch, mock_build_items, mock_run_auto
):
    dummy_df = pd.DataFrame([{"segment_index": 1, "test": "val"}])
    mock_run_auto.return_value = []
    mock_build_items.return_value = []
    
    mock_session = MagicMock()
    mock_session.decisions = []
    mock_launch.return_value = mock_session
    
    mock_merge.return_value = ([], [], pd.DataFrame())
    mock_make_summary.return_value = dummy_df

    res = analyze_workbook_with_afc_review(
        raw_xlsx_path="dummy.xlsx",
        stim_hz=1.0,
        recording_s=10.0,
        afc_config=AFCReviewConfig(enabled=True)
    )
    assert res.equals(dummy_df)

