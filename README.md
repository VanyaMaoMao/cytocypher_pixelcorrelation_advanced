# Cytocypher PixelCorrelation Advanced

Python tools for automated beat/event detection on Cytocypher PixelCorrelation segment workbooks, with optional AFC review workflow and report generation.

## What this project does

- Reads Excel workbooks containing sheets named like `PixelCorrelation Segment N`
- Runs automatic peak/event analysis per segment
- Produces:
  - segment-level DOCX report (`*_arrhythmia_report.docx`)
  - workbook summary Excel (`*_arrhythmia_summary.xlsx`)
- Optionally runs AFC review mode and exports AFC review artifacts
- Optionally exports peak-trace debug files (`*_peak_debug.xlsx`, `*_peak_debug.csv`)

## Project structure

- `pixel_counter/` - core package (analysis, preprocessing, QC, reporting, AFC review integration)
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

### Automatic mode (default)

```powershell
python .\run_pixel_analysis.py --input ".\tested files\SCRAMBLED ISO individual transients result.xlsx"
```

### Optional flags

- `--afc-review` : enable AFC review pipeline
- `--afc-interactive` : open interactive AFC review UI (requires `--afc-review`)
- `--afc-resume` : resume existing AFC review session (requires `--afc-review`)
- `--debug-peak-trace` : export peak debug files (`*_peak_debug.xlsx`, `*_peak_debug.csv`)

Example with AFC review:

```powershell
python .\run_pixel_analysis.py --input ".\tested files\SCRAMBLED ISO individual transients result.xlsx" --afc-review
```

## Typical outputs

For an input like `my_workbook.xlsx`, outputs are written next to the input file:

- `my_workbook_arrhythmia_report.docx`
- `my_workbook_arrhythmia_summary.xlsx`
- (optional) `my_workbook_peak_debug.xlsx`
- (optional) `my_workbook_peak_debug.csv`
- (optional AFC) `my_workbook_afc_review_session.json`, `my_workbook_afc_events.csv`, `my_workbook_afc_review_log.csv`

## Notes

- The runner keeps AFC review disabled by default.
- In the current runner script, auto mode already enables debug tracing outputs by default; `--debug-peak-trace` is primarily relevant when using AFC review mode.
- Interactive AFC mode requires a GUI-capable environment.
