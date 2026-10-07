import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import subprocess
import sys
import numpy as np

from pixel_counter import (
    analyze_raw_cytocypher_workbook,
    analyze_workbook_with_afc_review,
    BeatCounterConfig,
    AFCReviewConfig
)

from pixel_counter.results import _make_summary_dataframe

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


def test_make_summary_dataframe_status_fields():
    # Pass case
    segment_results_pass = [{
        "sheet_name": "Pass_Sheet",
        "n_main": 10,
        "n_main_primary": 10,
        "n_rescue": 0,
        "meta": {"qc_pass": True, "qc_reason": "ok"}
    }]
    df_pass = _make_summary_dataframe(segment_results_pass, stim_hz=1.0, recording_s=10.0)
    assert df_pass.iloc[0]["Status"] == "PASS"
    assert df_pass.iloc[0]["Accepted BPM"] == 60.0
    assert df_pass.iloc[0]["Accepted events"] == 10

    # Reject case
    segment_results_reject = [{
        "sheet_name": "Reject_Sheet",
        "n_main": 5,
        "n_main_primary": 5,
        "n_rescue": 0,
        "meta": {"qc_pass": False, "qc_reason": "low_snr"}
    }]
    df_reject = _make_summary_dataframe(segment_results_reject, stim_hz=1.0, recording_s=10.0)
    assert df_reject.iloc[0]["Status"] == "REJECT"
    assert np.isnan(df_reject.iloc[0]["Accepted BPM"])
    assert np.isnan(df_reject.iloc[0]["Accepted events"])
    assert df_reject.iloc[0]["Diagnostic BPM"] == 30.0
    assert df_reject.iloc[0]["Diagnostic events"] == 5

    # Error case
    segment_results_error = [{
        "sheet_name": "Error_Sheet",
        "n_main": 0,
        "n_main_primary": 0,
        "n_rescue": 0,
        "meta": {"qc_pass": False, "qc_reason": "runtime_error_ValueError"}
    }]
    df_error = _make_summary_dataframe(segment_results_error, stim_hz=1.0, recording_s=10.0)
    assert df_error.iloc[0]["Status"] == "ERROR"
    assert np.isnan(df_error.iloc[0]["Accepted BPM"])
    assert np.isnan(df_error.iloc[0]["Accepted events"])
    assert df_error.iloc[0]["Diagnostic BPM"] == 0.0
    assert df_error.iloc[0]["Diagnostic events"] == 0


def test_cli_exit_code_missing_args():
    res = subprocess.run([sys.executable, "run_pixel_analysis.py"], capture_output=True)
    assert res.returncode == 2


@patch("run_pixel_analysis.analyze_raw_cytocypher_workbook")
@patch("run_pixel_analysis.analyze_workbook_with_afc_review")
def test_cli_exit_code_1_on_error(mock_afc, mock_raw):
    # Mocking a summary dataframe with an ERROR status
    df_error = pd.DataFrame([{"Segment": "Seg1", "Status": "ERROR", "Diagnostic BPM": 0.0}])
    mock_raw.return_value = df_error
    
    # Needs to be called via subprocess or directly importing main and patching sys.argv
    import run_pixel_analysis
    with patch.object(sys, 'argv', ['run_pixel_analysis.py', '--input', 'dummy.xlsx']):
        with pytest.raises(SystemExit) as e:
            run_pixel_analysis.main()
        assert e.value.code == 1

@patch("run_pixel_analysis.analyze_raw_cytocypher_workbook")
@patch("run_pixel_analysis.analyze_workbook_with_afc_review")
def test_cli_exit_code_0_on_success(mock_afc, mock_raw):
    # Mocking a summary dataframe with a PASS and REJECT status, but NO ERROR
    df_pass_reject = pd.DataFrame([
        {"Segment": "Seg1", "Status": "PASS", "Diagnostic BPM": 60.0},
        {"Segment": "Seg2", "Status": "REJECT", "Diagnostic BPM": 30.0}
    ])
    mock_raw.return_value = df_pass_reject
    
    import run_pixel_analysis
    with patch.object(sys, 'argv', ['run_pixel_analysis.py', '--input', 'dummy.xlsx']):
        try:
            run_pixel_analysis.main()
        except SystemExit as e:
            pytest.fail(f"SystemExit raised with code {e.code} when 0 was expected.")
