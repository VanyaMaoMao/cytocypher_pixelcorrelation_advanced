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

## 04. Fix crash missing Begin fallback
- Identified that `build_concatenated_signal` caused a `NameError` on fallback processing when no `Begin` column exists, because of a missing `re` import.
- Added missing `re` import to `pixel_counter/preprocessing.py`.
- Added test in `tests/test_preprocessing.py` to ensure fallback processing paths resolve without crashes.
- Fixed a secondary crash in `pixel_counter/qc.py::_row_corr_median` which occurs during `np.corrcoef` calculation if the input has fewer than 2 elements.

## 05. Remove user-facing AFC workflow
- Removed AFC review arguments (`--afc-review`, `--afc-interactive`, `--afc-resume`) from `run_pixel_analysis.py`, returning a clear error if used.
- Removed unused imports and references from CLI and `pixel_counter/__init__.py`.
- Updated output report naming in CLI from `_arrhythmia_*` to `_contraction_*`.
- Updated `README.md` to reflect the removal of AFC and the new output file names.
- Updated `test_run_pixel_analysis.py` to assert correct exit codes and tests without the legacy mock `analyze_workbook_with_afc_review`.

## 07. Remove dead rules without changing method
- Verified call graph and public API. Identified `prune_short_gap_weak_mains`, `prune_local_weak_mains`, `prune_interbeat_tiny_bumps`, and `_candidate_orientation_sanity_score` as uncalled functions in `pixel_counter/analysis.py`.
- Removed these dead functions from `pixel_counter/analysis.py`.
- Removed their associated configuration fields from `BeatCounterConfig` in `pixel_counter/config.py`, specifically `orientation_sanity_rescue_ratio_penalty`, `orientation_sanity_promoted_fail_penalty`, and all fields starting with `main_local_weak_`, `main_local_tiny_`, `main_short_gap_`, and `main_interbeat_tiny_`.
- Ran regression tests to verify that these removals don't alter current functionality. All tests pass successfully.

## 08. Input Excel contract and sampling frequency
- Documented input schema in `docs/input_schema.md` covering Begin, End, Sample ID, Sampling Frequency, y-offsets, overlaps, and conflicts.
- Refactored `load_cytocypher_excel` in `pixel_counter/io_utils.py` to correctly parse and sort numeric `y ` column offsets.
- Added validation to raise errors on duplicate, non-numeric `y ` offsets or missing `y ` columns.
- Added explicit parsing and validation of `Sampling Frequency` array to throw errors on missing/NaN, zero, negative, or multiple conflicting frequency values.
- Updated `extract_sample_id_from_segment_sheet` and `load_cytocypher_excel` to explicitly error on multiple conflicting Sample IDs on a single sheet, removing the previous silent fallback (majority vote).
- Authored tests in `test_io_excel.py`, `test_io_excel_validation.py`, and `test_extract_sample_id.py`.
