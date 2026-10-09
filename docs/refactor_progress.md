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

## 09. NaN does not drop time position
- Modified `build_concatenated_signal` in `pixel_counter/preprocessing.py` to stop using `tr[~np.isnan(tr)]` when parsing raw traces. This ensures that internal missing values do not shift the timestamps of subsequent events and gaps are preserved.
- Removed the gap interpolation logic so that gaps remain as NaNs, maintaining invalid representation instead of fabricating data points.
- Refactored orientation-finding features like `_row_dominant_direction`, `choose_orientation_make_peaks_positive`, and `_row_corr_median` to properly handle NumPy arrays that have `NaN` elements by switching to nan-safe routines or properly masking inputs prior to calculation.
- Added tests asserting that missing values in the form of NaNs do not discard the element position.

## 10. Coverage, gaps, and absolute timeline
- Verified `build_concatenated_signal` evaluates the valid overlap/gaps dynamically using `np.isnan` tracking correctly mapped against `Begin/End` coordinates. Gaps appropriately remain `NaN`.
- Refactored `build_transient_id_vector` to assign a deterministic sentinel `-1` instead of allocating an uninitialized `np.empty` structure. This ensures sections without measurements appropriately fall back rather than emitting invalid memory traces or arbitrarily combining with unassociated overlaps.
- Extended regression tests specifically checking overlap blending intervals and testing that sentinel IDs properly apply in missing data fields across boundaries without error.

## 11. Canonical events, full debug, and Sample ID
- Updated `build_events_dataframe` and `_build_peak_debug_rows` in `pixel_counter/analysis.py` to output a fully unified event schema.
- Added explicit tracking and reporting of new standard fields: `event_id`/`candidate_id`, `segment_name`, `segment_index`, `sample_id`, `candidate_index`, `detection_source`, and `decision_status`.
- Integrated `rescue` peaks directly within the `build_events_dataframe` pass, avoiding redundant and error-prone retroactive building from metadata blocks during reporting.
- Simplified `_collect_main_events_table` in `pixel_counter/reporting.py` to seamlessly aggregate the standardized and canonical output arrays.
- Ensured sample ID properly cascades directly from analysis stages directly into the compiled dataframes.
- Validated modifications using newly added integration tests inside `tests/test_canonical_events.py`.

### Audit Remediation (Task 01: Regression Tests)
**Status**: COMPLETE
**Details**:
- Validated current HEAD and established regression test baseline for steps 01-11 issues detailed in the independent audit.
- Created `tests/test_audit_regression_row_partition.py` reproducing the 40-peak counterexample showing partition dependence (marked strict xfail for steps 14-18).
- Created `tests/test_audit_regression_missing_data.py` proving NaN presence falsely generates hundreds of artifacts instead of properly ignoring gaps (marked strict xfail).
- Created `tests/test_audit_regression_timing_overlap.py` verifying legacy padding and median overlap correction silently drops and shifts data incorrectly (marked strict xfail).
- Created `tests/test_audit_regression_canonical.py` reproducing duplicate event IDs across rescue/main boundaries and missing rescue events in peak debug outputs (marked strict xfail).
- Created `tests/test_audit_regression_io.py` testing for un-validated silent 250Hz missing sampling frequency and output-collision issues, including missing y-offset columns and conflicting sampling frequencies handling.
- Created `tests/test_audit_regression_bpm.py` testing for denominator discrepancy between results layer and reporting layers, as well as REJECT instances incorrectly reporting 0 BPM rather than NaN (marked strict xfail).
- Added `generate_baseline.py` script to export real-world test results per segment to `docs/real_data_baseline.json` as a regression snapshot prior to code modifications.
