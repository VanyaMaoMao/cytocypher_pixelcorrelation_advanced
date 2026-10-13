import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open

from run_v2_review_utility import process_workbook, generate_html

def test_v2_review_utility_timestamp_conversion_and_mapping():
    # We will mock load_cytocypher_excel and others to return deterministic dummy data

    # Dummy signal and time config
    fs = 100.0
    t0 = 5.0
    sig_corr = np.array([0.0, 1.0, 0.0, 0.5, 0.0])

    # Mock behavior of analysis
    mock_candidates = {
        "time_s": np.array([0.01, 0.03]), # segment relative
        "amplitude": np.array([1.0, 0.5]),
        "status": np.array(["PASS", "REVIEW"]),
        "dedup_audit": [{"suppressed_id": 4, "competing_id": 1, "reason": "too close"}]
    }

    mock_v2_res = {
        "candidates": mock_candidates,
        "metadata": {"qc_status": "PASS"}
    }

    with patch("run_v2_review_utility.pd.ExcelFile") as mock_xl:
        mock_xl_inst = mock_xl.return_value
        mock_xl_inst.sheet_names = ["PixelCorrelation Segment 1"]

        with patch("run_v2_review_utility.load_cytocypher_excel") as mock_load:
            # df, y_cols, fs, t0
            mock_load.return_value = (pd.DataFrame(), ["y 0"], fs, t0)

            with patch("run_v2_review_utility.extract_sample_id_from_segment_sheet", return_value="Cell_A"):
                with patch("run_v2_review_utility.build_concatenated_signal", return_value=(sig_corr, [], {})):
                    with patch("run_v2_review_utility.analyze_v2_pipeline", return_value=mock_v2_res):
                        segments_data, source_name = process_workbook("dummy.xlsx")

                        assert len(segments_data) == 1
                        seg = segments_data[0]
                        assert seg["t0"] == t0
                        assert seg["qc_status"] == "PASS"

                        # Timestamps should be absolute: 0.01 + 5.0 = 5.01, 0.03 + 5.0 = 5.03
                        assert seg["candidates"]["time_s"][0] == 5.01
                        assert seg["candidates"]["time_s"][1] == 5.03
                        assert seg["candidates"]["status"][0] == "PASS"
                        assert seg["candidates"]["status"][1] == "REVIEW"

                        # Suppressed candidate (index 4) should be recovered: time = 4/100 + 5.0 = 5.04
                        assert len(seg["candidates"]["time_s"]) == 3
                        assert seg["candidates"]["time_s"][2] == 5.04
                        assert seg["candidates"]["amplitude"][2] == 0.0
                        assert seg["candidates"]["status"][2] == "REJECT"

@patch("urllib.request.urlopen")
def test_v2_review_utility_html_offline_generation(mock_urlopen):
    # Mock the urlopen to return a simple script body
    import io
    mock_urlopen.return_value.__enter__.return_value = io.BytesIO(b"console.log('offline plotly');")

    segments_data = [{"dummy": "data"}]

    with patch("builtins.open", mock_open()) as m:
        generate_html(segments_data, "test.xlsx", "out.html")

        # Verify that HTML was written and the plotly string was embedded
        written_content = "".join(call[0][0] for call in m().write.call_args_list)
        assert "console.log('offline plotly');" in written_content
        assert "https://cdn.plot.ly" not in written_content
        assert "Reviewed" in written_content
