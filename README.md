# Cytocypher PixelCorrelation Advanced

Python tools for automated contraction frequency analysis on Cytocypher PixelCorrelation segment workbooks.

## What this project does

- Reads Excel workbooks containing sheets named like `PixelCorrelation Segment N`
- Runs automatic peak/event analysis per segment
- Produces:
  - segment-level DOCX report (`*_contraction_report.docx`)
  - workbook summary Excel (`*_contraction_summary.xlsx`)
- Optionally exports peak-trace debug files (`*_peak_debug.xlsx`, `*_peak_debug.csv`)

## Project structure

- `pixel_counter/` - core package (analysis, preprocessing, QC, reporting)
- `run_pixel_analysis.py` - command-line entry script
- `tests/` - test files
- `tested files/` - sample/test input workbooks (if present)

## Requirements

- Python 3.10+ (recommended)
- See `requirements.txt`

## Setup (Windows PowerShell)

```powershell
cd "C:\path\to\cytocypher_pixelcorrelation_advanced"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

### Automatic mode

```powershell
python .\run_pixel_analysis.py --input ".\tested files\SCRAMBLED ISO individual transients result.xlsx"
```

### Optional flags

- `--debug-peak-trace` : export peak debug files (`*_peak_debug.xlsx`, `*_peak_debug.csv`)

## Typical outputs

For an input like `my_workbook.xlsx`, outputs are written next to the input file:

- `my_workbook_contraction_report.docx`
- `my_workbook_contraction_summary.xlsx`
- (optional) `my_workbook_peak_debug.xlsx`
- (optional) `my_workbook_peak_debug.csv`

## Notes

- AFC review mode and associated flags have been removed.
- In the current runner script, auto mode already enables debug tracing outputs by default.
