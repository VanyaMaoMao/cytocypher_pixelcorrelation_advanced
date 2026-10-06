import os
import pandas as pd
from pixel_counter import analyze_raw_cytocypher_workbook

def test_analyze_dummy_workbook():
    raw_path = "dummy_input.xlsx"
    assert os.path.exists(raw_path)
    
    # Run full analysis
    summary_df = analyze_raw_cytocypher_workbook(
        raw_xlsx_path=raw_path,
        stim_hz=1.0,
        recording_s=10.0,
        output_docx="dummy_report.docx",
        output_summary_xlsx="dummy_summary.xlsx"
    )
    
    assert isinstance(summary_df, pd.DataFrame)
    assert not summary_df.empty
    assert "Segment" in summary_df.columns or "segment" in summary_df.columns
