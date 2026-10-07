# Refactor Progress

## 00. Fix initial state
- Verified current state. Current commit: `176e490bfa0710ee01d1daa5e1c1a2fa16377edb`.
- Created working branch `task/refactor-00-01`.

## 01. Create a reproducible test suite
- Installed dependencies and set up basic pytest environment.
- Created `tests/__init__.py` and `tests/test_basic.py` with placeholder tests.

## 02. Ensure input files are never overwritten
- Replaced `.replace('.xlsx')` string replacement logic with a safer `build_output_path` using `Path`.
- Implemented `check_output_conflicts` to verify output paths don't conflict with inputs.
- Created an `atomic_file_path` context manager to write output safely to a temporary location before atomically swapping.
- Protected outputs in `io_utils.py` and `reporting.py` with `atomic_file_path`.
- Added IO regression tests in `tests/test_io.py`.

## 03. ERROR and REJECT are not equal to 0 BPM
- Modified `_make_summary_dataframe` in `pixel_counter/results.py` to add explicit `Status` field (`PASS`, `REJECT`, `ERROR`).
- Created separate fields for accepted results (`Accepted BPM`, `Accepted events`) and diagnostic results (`Diagnostic BPM`, `Diagnostic events`). Accepted fields are set to `NaN` when status is not `PASS`.
- Updated `run_pixel_analysis.py` to return exit code 1 if any `ERROR` status is found in the summary dataframe, ensuring batch processing reports partial failures correctly.
- Added regression tests in `tests/test_run_pixel_analysis.py` to verify exit codes and dataframes formats.
