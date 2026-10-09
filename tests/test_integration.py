import os
import pandas as pd
from pixel_counter import analyze_raw_cytocypher_workbook

def test_analyze_dummy_workbook(tmp_path):
    raw_path = str(tmp_path / "dummy_input.xlsx")
    out_docx = str(tmp_path / "dummy_report.docx")
    out_xlsx = str(tmp_path / "dummy_summary.xlsx")

    # Generate a dummy sheet so it doesn't fail
    df = pd.DataFrame({"y 0": [1, 2, 3]})
    df.to_excel(raw_path, sheet_name="PixelCorrelation Segment 1", index=False)

    assert os.path.exists(raw_path)
    
    # Run full analysis
    summary_df = analyze_raw_cytocypher_workbook(
        raw_xlsx_path=raw_path,
        stim_hz=1.0,
        recording_s=10.0,
        output_docx=out_docx,
        output_summary_xlsx=out_xlsx
    )
    
    assert isinstance(summary_df, pd.DataFrame)
    assert not summary_df.empty
    assert "Segment" in summary_df.columns or "segment" in summary_df.columns
