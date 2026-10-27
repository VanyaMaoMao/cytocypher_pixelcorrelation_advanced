# Forensic Scientific Integrity & Anti-Overfitting Audit Report

**Repository**: `https://github.com/VanyaMaoMao/cytocypher_pixelcorrelation_advanced`  
**Audited Baseline Branch**: `stable/v2-2026-10-20`  
**Audited Baseline Commit**: `ccdc1c8` (`ccdc1c8740f073b7ad91f962d6b09e28440b4fac`)  
**Audit Date**: 2026-10-26  

---

## 1. Executive Summary

A comprehensive, line-by-line forensic audit was conducted across all production modules (`pixel_counter/*.py`, `run_v2_review_utility.py`, `run_pixel_analysis.py`) and test suites (`tests/*.py`) at baseline commit `ccdc1c8`.

While no production `if` statement literally branched on a string workbook filename or segment number variable (e.g., `if segment_id == 16`), **the audit uncovered severe indirect dataset-specific overfitting**:
1. **Explicit segment-number targeting in production code comments and compound thresholds**: Production files (`pixel_counter/sarcomere_qc.py`, `pixel_counter/v2_signal.py`, `pixel_counter/calcium_detector.py`) contained explicit comments naming target segments (`A Seg 1, A Seg 3, A Seg 16`, `B Seg 7, B Seg 12, B Seg 32, B Seg 42`, `A28, A29, A39, A41, B29, B39, B49`, `Seg 25`, `3 Hz`) immediately above multi-variable conjunctive/disjunctive thresholds engineered to carve out those exact recordings.
2. **Count-dependent special rules**: In `pixel_counter/sarcomere_qc.py` (lines 458–466), the condition `if 0 < n_accepted <= 12 and n_rec_ivs >= 8 and (n_rec_ivs / n_accepted >= 0.75) and rail_frac >= 0.008:` was specifically constructed to reject `CAR DISH1` Segments 1 and 3 (which have 10 accepted beats) while sparing longer recordings with >12 beats.
3. **Unconditional rejection of clean quiescent sarcomere recordings**: In `pixel_counter/sarcomere_qc.py` (lines 420–426), `if n_accepted == 0:` unconditionally returned `"REJECT", "noise_dominated"`, making it impossible for a clean, low-noise quiescent cardiomyocyte recording to pass QC with 0 contractions.
4. **Sampling-rate-dependent derivative and window constants**: In `pixel_counter/sarcomere_qc.py` (lines 123–190) and `pixel_counter/sarcomere_detector.py` (line 438), derivative jump thresholds (`0.028`, `0.032`, `0.052` µm/sample) and window sizes (`7`, `14`, `21` samples) were hardcoded assuming `fs = 250 Hz` rather than scaled by physical velocity (`µm/s / fs`) and duration (`s * fs`).
5. **Frequency-band-specific polarity cascades**: In `pixel_counter/v2_signal.py` (lines 472–614), discrete frequency gates (`1.5 <= freq <= 2.4` for 2 Hz and `tc <= 0.40` with five sub-rules 4A–4E for 3 Hz) were tuned to match specific segments in `25092026_IMR90_1,2,3HZ` and `240926_IMR90_1,2,3HZ`.
6. **Three redundant/conflicting classification passes and dead code in `sarcomere_detector.py`**: Lines 476–506 computed candidate status and immediately overwrote it; lines 577–631 and `classify_sarcomere_contractions` (lines 665–893) ran three conflicting classification passes where Pass 2 rejected `SECONDARY` events at `pos == 0` or `dt > 1.25 s` and Pass 3 subsequently overwrote Pass 2 using `pct_shortening` alone.
7. **Dead 1,700-line duplicate HTML template in `pixel_counter/sarcomere_review.py`**: `SARCOMERE_HTML_TEMPLATE` (lines 432–2137) was completely unused because `generate_sarcomere_html` delegated directly to `run_v2_review_utility.generate_html`.

---

## 2. Confirmed Dataset-Specific Conditions

| File & Lines | Exact Code / Comment Evidence | Target Recordings Fitted | Scientific Flaw | Corrective Action |
| :--- | :--- | :--- | :--- | :--- |
| `pixel_counter/sarcomere_qc.py` L458–466 | `# 4. Excessive reconstruction dependence in short/medium recordings:`<br>`if n_accepted <= 12 and n_accepted > 0:`<br>`    if n_rec_ivs >= 8 and (n_rec_ivs / max(n_accepted, 1) >= 0.75) and rail_frac >= 0.008:` | `CAR DISH1` Seg 1 & Seg 3 (10 beats each) | Arbitrary beat-count ceiling (`n_accepted <= 12`) has no physical basis; a 15-beat recording with 100% reconstructed intervals would bypass this rule. | **REMOVE_AS_OVERFIT**: Replace with a count-invariant uncorrupted event fraction rule across all recordings. |
| `pixel_counter/sarcomere_qc.py` L427–455 | `# Single-contraction positive controls (A28, A29, A39, A41, B29, B39, B49) have:`<br>`# Noise-only segments (A4, A8, A10, A11, A21, A23-26, A30, A32-34, A36-38, A42-58) have:`<br>`# Sub-case B: Moderate-count noise traces (like Seg 9: 4 false peaks...)` | `CAR DISH1` Segs 4, 8, 9, 10, 11, 21, 23–26, 28–30, 32–39, 41–58; `ISO DISH1` Segs 29, 39, 49 | Multi-clause thresholds (`asym < 1.65 and med_row_corr < 0.75 and (dn_p98 < 0.045 or med_snr < 12.0 or asym < 1.35)`, `n_accepted >= 3 and dn_p98 < 0.016...`) reverse-engineered to partition the CAR/ISO segment table. | **REPLACE_WITH_GENERAL_SIGNAL_MODEL**: Replace with general waveform SNR and downward-excursion asymmetry criteria. |
| `pixel_counter/sarcomere_qc.py` L400–411 | `# 1. Severe tracking artifact domination (vertical lines / rails / loss of tracking)`<br>`# Catches B Seg 7 (rail_frac=0.041), B Seg 12 (rail_frac=0.025, 13 clusters),`<br>`# B Seg 32 (art_frac=0.228), B Seg 42 (max_art_dur=0.90s, art_frac=0.149)` | `ISO DISH1` Seg 7, Seg 12, Seg 32, Seg 42 | Thresholds (`0.035`, `0.020 and >=10`, `0.14`, `0.80`) chosen just below the empirical values of those 4 segments (`0.041`, `0.025 & 13`, `0.228`, `0.90`). | **SIMPLIFY_AND_MERGE**: Use clean, orthogonal signal-coverage bounds (`max_rail_fraction`, `max_artifact_fraction`, `max_contiguous_artifact_s`). |
| `pixel_counter/v2_signal.py` L495–526 | `# Rule 2: Overwhelming onset kinetics with curvature confirmation.`<br>`# At intermediate pacing (e.g. 2 Hz), broadened contractions or after-contraction shoulders (e.g. Seg 25)...`<br>`if 1.5 <= freq <= 2.4:` | `25092026_IMR90` Seg 25 (2 Hz) | Hard frequency gate `1.5 <= freq <= 2.4` creates a discontinuous jump at 1.49 Hz vs 1.50 Hz and 2.40 Hz vs 2.41 Hz. | **REPLACE_WITH_GENERAL_SIGNAL_MODEL**: Weight dwell vs kinetic/curvature evidence continuously by cycle duty ratio. |
| `pixel_counter/v2_signal.py` L539–614 | `# Rule 4: High-frequency (e.g. 3 Hz, cycle <= 0.40s) cycle morphology resolution`<br>Rules 4A, 4B, 4C, 4D, 4E | `240926_IMR90` & `25092026_IMR90` 3 Hz segments (30, 33, 35, 36, 39, 40, 41, 44, 45) | Five cascading `elif` branches with 18 distinct numeric cutoffs tailored to individual 3 Hz segments. | **SIMPLIFY_AND_MERGE**: Replace with a single unified multi-cue score and abstain (`UNDETERMINED`) when cues conflict. |
| `pixel_counter/qc_v2.py` L107–121 | Narrow parameter-box exclusion in `detect_incompatible_contraction_directions`: `if 0.30 <= h_ratio <= 0.60 and 0.11 <= med_wu <= 0.20 and 0.08 <= med_wd <= 0.16 and med_wd < med_wu and 1.25 <= ratio_count <= 2.50:` | Specific neonatal/iPSC segments with biphasic rebound | Arbitrary 6-dimensional hyper-rectangle carve-out. | **SIMPLIFY_AND_MERGE**: Replace with a general physiological check testing whether minor opposite-polarity peaks are time-locked post-relaxation rebounds following dominant contractions. |

---

## 3. Suspected Dataset-Driven Thresholds

1. **`pixel_counter/qc_v2.py` (`evaluate_signal_interpretability_qc`, lines 418–483)**:
   - Contains 5 disjunctive branches for `is_noise_dominated` when `n_accepted == 0` and 6 disjunctive branches for `is_noise_dominated_with_acc` when `n_accepted > 0`, with slightly different noise cutoffs (`0.0050`, `0.0042`, `0.0028`, `0.0020`, `0.0030`) and SNR cutoffs (`8.5`, `11.5`, `12.0`, `14.0`, `20.0`).
   - **Audit conclusion**: While each variable (`sn_raw`, `drift_ratio`, `med_acc_snr`, `acc_fraction`) is physically meaningful, the proliferation of 11 sub-clauses reflects iterative patching around `SCRAMBLED ISO` Segments 2 & 6 and `NICONICO` Segments 70 & 84.
2. **`pixel_counter/calcium_detector.py` (line 524)**:
   - `noise_mult = params.min_prominence_sigma * 0.43` multiplies the user-facing parameter `min_prominence_sigma = 4.2` by a hidden constant `0.43` to produce `1.806`, which matches the comment in `tests/test_calcium_analyzer.py` line 947 (`cand_prom < 1.8 * raw_noise_sigma` for `DISH1NICOTINE` Segment 44).
   - **Audit conclusion**: Hidden scaling of a named parameter is scientifically opaque and must be replaced by an explicit parameter `raw_prominence_sigma`.
3. **`pixel_counter/calcium_detector.py` (`evaluate_calcium_qc`, lines 978–981)**:
   - `if mode == "adult": if n_accepted > 0 and n_main == 0 and n_accepted <= 2 and (noise_sigma > 0.012 or snr < 6.5):` was added specifically for `CAR DISH1` Segment 46.
   - **Audit conclusion**: Should be expressed as a general low-SNR false-peak QC criterion rather than an `n_main == 0 and n_accepted <= 2` count rule.

---

## 4. Frequency-Specific Heuristics

1. **PixelCorrelation Polarity (`pixel_counter/v2_signal.py`)**:
   - Hard branches at `1.5 <= freq <= 2.4` Hz and `tc <= 0.40` s (`freq >= 2.5` Hz).
   - **Why this is problematic**: Cardiomyocyte contraction kinetics vary continuously with beat rate and duty cycle (`FWHM / IBI`), not via step functions at `1.5 Hz`, `2.4 Hz`, and `2.5 Hz`.
2. **Sarcomere Detector (`pixel_counter/sarcomere_detector.py`, line 871)**:
   - In `classify_sarcomere_contractions`, a candidate in `[T_SCE, T_MAIN)` was rejected if `dt > 1.25` s or `pos == 0` (first event in trace). That meant a genuine secondary event occurring after a >1.25 s pause or at the start of a recording window was rejected by Pass 2 (and then inconsistently resurrected by Pass 3).

---

## 5. Circular Validation Risks & Dataset Provenance

All experimental workbooks present in the repository have been inspected during prior development turns:
- `sarcomere shortening/(MI) CGP+CAR+ROLI+IBMX DISH1  individual transients result.xlsx` (`CAR DISH1`, 52 segments)
- `sarcomere shortening/(MI) CGP+ISO+ROLI+IBMX DISH1  individual transients result.xlsx` (`ISO DISH1`, 50 segments)
- `all data/25092026_IMR90_1,2,3HZ individual transients result.xlsx`
- `all data/240926_IMR90_1,2,3HZ individual transients result.xlsx`
- `all data/13-12-24.4D.DISH1CONTROL.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx`
- `all data/13-12-24.4D.DISH2ICI.SEARCH.BASE.NICONICO.ISOISO individual transients result.xlsx`
- `all data/DISH1NICOTINE individual transients result.xlsx`
- `all data/SCRAMBLED ISO individual transients result.xlsx`
- `all data/05-12-24.10D.CGP.SEARCH.BASE.NICONICO.ISOISO.5ISOMISTAKE individual transients result.xlsx`

**Mandatory Audit Finding**:
- **Zero untouched independent holdout datasets exist in the workspace.**
- All evaluations on the above workbooks represent **DEVELOPMENT REGRESSION RESULTS**, NOT independent scientific validation.
- *"Independent scientific validation cannot yet be completed with the available previously inspected recordings."*

---

## 6. Tests That Encode Predetermined Segment Outcomes

The following test functions in `tests/` encode segment-specific expected counts or statuses from the development workbooks:
1. `tests/test_sarcomere_analyzer.py`:
   - `test_isolated_single_contraction_segments`: Asserts `PASS` and `n_accepted == 1` on CAR Segs `[28, 29, 39, 41]` and ISO Segs `[39, 49]`.
   - `test_noise_and_tracking_failure_qc_rejection`: Asserts `REJECT` on 26 specific CAR segment numbers and 14 specific ISO segment numbers.
   - `test_vertical_artifact_handling_and_corrupted_trough_nan`: Asserts `n_accepted == 10` and `n_corrupted_events == 2` on CAR Seg 16.
   - `test_high_frequency_irregular_cardiomyocyte_beats`: Asserts count ranges on ISO Segs `2, 17, 21, 23, 27, 28, 30`.
   - `test_experimental_segment_33_sequence_classification`, `test_experimental_segment_35_classification`, `test_experimental_segment_41_classification`: Assert specific sequences/counts on ISO Segs 33, 35, 41.
   - `test_car_dish1_segment1_recovery_regression`: Asserts `qc_reason == "excessive_reconstruction_dependence"` on CAR Seg 1.
2. `tests/test_calcium_analyzer.py`:
   - `test_case_a_imr90_segment_6_false_positive_rejection` through `test_noise_rejection_dish1nicotine_seg44`: Assert exact segment counts (`4`, `5`, `10`, `9`, `7`, `2`, `11`, `12`) and specific timestamps (`172.103s`, `304.91s`, `255.43s`, `1208.134s`, `919.550s`) on IMR90, DISH1CONTROL, DISH2ICI, and DISH1NICOTINE.
3. `tests/test_v2_robustness_improvements.py`:
   - `test_real_workbooks_problems_a_to_f`, `test_scrambled_iso_segment_2_noise_dominated_rejection`, `test_clean_and_borderline_records_preserved_pass`: Assert exact counts and statuses on `SCRAMBLED ISO` and `05-12-24` segments.
4. `tests/test_v2_polarity_qc.py`:
   - `test_dev_raw_polarity_references`, `test_dev_high_frequency_safe_abstention`, `test_dev_conflicting_segments_fail_safe_undetermined`, `test_calcium_ecc_real_problematic_segments`, `test_sept24_3hz_resolved_segments`: Assert segment polarity outcomes on IMR90 workbooks.

**Audit Action on Development Workbook Tests**:
- Distinguish **general scientific invariant / synthetic / metamorphic tests** (which define algorithm correctness) from **development dataset regression tests** (which track behavior on previously inspected recordings without forcing overfit production rules).

---

## 7. Deleted Test Files Audit (`git status`: `D tests/test_*.py`)

Five test files were marked deleted (`D`) in the working tree relative to `HEAD` (`ccdc1c8`):
1. `tests/test_detect_events_v2.py`
2. `tests/test_polarity_regression.py`
3. `tests/test_qc_v2.py`
4. `tests/test_step16_refinement.py`
5. `tests/test_step17_polarity.py`

**Forensic Inspection of Deleted Files**:
- Inspection via `git show HEAD:<file>` confirmed that all 5 files imported and tested a **superseded prototype v2 API** (`from pixel_counter.analysis import detect_events_v2, deduplicate_events_v2, auto_select_polarity_v2` and `from pixel_counter.qc_v2 import evaluate_qc_v2` expecting `"REVIEW"` status and `stability_count_high`/`v2_two_pass_enabled`).
- That prototype API was intentionally removed in commit `316b9cc` (`Stable Cytocypher v2 - validated detector, QC and interactive review`) when `pixel_counter/v2_detector.py`, `pixel_counter/v2_signal.py`, `pixel_counter/v2_pipeline.py`, and binary `PASS`/`REJECT` `pixel_counter/qc_v2.py` were introduced. In fact, `tests/test_v2_polarity_qc.py` (lines 92–103, `test_sensitivity_analysis_removed`) explicitly asserts that `detect_events_v2`, `v2_two_pass_enabled`, and `stability_*` no longer exist in `qc_v2` or `BeatCounterConfig`!
- All valid scientific invariants from those 5 legacy files (synthetic 1/2/4 Hz detection, missing-data NaN gaps, row-boundary & row-partition invariance, deep-valley doublet separation, polarity inversion antisymmetry, and technical QC rejection) are covered in `tests/test_v2_detector.py`, `tests/test_v2_polarity_qc.py`, `tests/test_audit_regression_*.py`, and the new `tests/test_scientific_integrity_metamorphic.py`.

---

## 8. Parameter Values Without Provenance & Sampling-Rate Dependencies

1. **Sampling-rate-dependent constants in `pixel_counter/sarcomere_qc.py` (lines 123–190)**:
   - Single-sample step threshold `0.028` µm/sample corresponds to `7.0 µm/s` at 250 Hz, but at 500 Hz it would require `14.0 µm/s` and at 100 Hz only `2.8 µm/s`.
   - Similarly, `0.024`, `0.032`, `0.040`, `0.044`, `0.045`, `0.052`, `0.090` µm/sample and window lengths `7`, `8`, `12`, `21` samples must all be parameterized in physical units (`µm/s` and `seconds`) and scaled by `fs`.
2. **Core artifact sample count in `pixel_counter/sarcomere_detector.py` (line 438)**:
   - `n_art_core > 14` (samples) -> must be `int(round(0.056 * fs))` (`56 ms`).
3. **Time-origin (`t0`) omission in `pixel_counter/sarcomere_artifact_recovery.py` (lines 98–99, 114–115, 141–142, 156–157, 166–167)**:
   - `reconstruct_narrow_tracking_artifacts` recorded `start_time_s = c0 / fs` (relative time) instead of `t0 + c0 / fs` (absolute time), causing a coordinate mismatch whenever `t0 != 0`.
4. **Ignored `params` argument in `pixel_counter/sarcomere_pipeline.py` (lines 233–236 & 502)**:
   - `analyze_sarcomere_pipeline` accepted `params: Optional[SarcomereDetectorParams]`, ignored it during preset execution, and crashed with `KeyError` if `params.preset_name` was not in `{"default", "permissive", "strict"}`.
5. **Miscounted `n_directly_measured` and `n_valid_amplitude` in `pixel_counter/sarcomere_pipeline.py` (lines 489–492)**:
   - In `mode="sarcomere"`, `match_info` did not populate `n_directly_measured`, so `analyze_sarcomere_pipeline` fell back to `reportable_acc`, counting reconstructed events with `NaN` amplitude as directly measured valid-amplitude events.

---

## 9. Instrument / File-Format Assumptions

1. **IonOptix FFT Sarcomere Rail Bounds (`1.28 µm` to `2.18 µm`)**:
   - The IonOptix CytoCypher FFT algorithm computes striation spacing within a hardware/optical spatial frequency window (~1.12 µm to ~2.2335 µm). When tracking fails, the output locks onto the upper (~2.2335 µm) or lower (~1.1168 µm) FFT rail.
   - Instead of hardcoding `1.28` and `2.18` inside `detect_sarcomere_artifacts`, these bounds are now encapsulated in a configurable `SarcomereMeasurementProfile` (`fft_rail_min_um=1.28`, `fft_rail_max_um=2.18`) and paired with signal-derived clipping/discontinuity detection so that smooth physiological extremes are not misclassified as rail lock.
2. **Sampling Frequency Metadata (`pixel_counter/io_utils.py`)**:
   - `load_cytocypher_excel` now records `declared_fs`, `implied_fs`, and `fs_reconciliation_reason` in `df.attrs` so any discrepancy between the declared column and `Begin`/`End` timestamps is explicitly documented in segment metadata.

---

## 10. Summary of Rules Retained, Parameterized, Simplified, and Removed

- **Removed as Overfit (`REMOVE_AS_OVERFIT`)**:
  - `SARC_QC_CAR_SEG1_SEG3_SPECIAL_RULE` (`n_accepted <= 12 and n_rec_ivs >= 8 ...` in `sarcomere_qc.py`).
- **Replaced with General Signal Models (`REPLACE_WITH_GENERAL_SIGNAL_MODEL`)**:
  - `SARC_QC_DERIVATIVE_SPIKE_THRESHOLDS`: Converted all per-sample step thresholds and sample windows in `sarcomere_qc.py` and `sarcomere_detector.py` to physical velocity (`µm/s`) and duration (`seconds`) scaled by `fs`.
  - `SARC_QC_ZERO_EVENTS_ALWAYS_REJECT`: Enabled clean quiescent cardiomyocyte recordings (`n_accepted == 0`, low `sigma_hf`, intact tracking) to `PASS` QC as `clean_quiescent_no_contractions`.
  - `SARC_QC_NOISE_DOMINATED_SEGMENT_GATES`: Replaced the CAR/ISO segment-fitted 3-branch tree with a general SNR, downward-asymmetry, and noise-rejection ratio check.
  - `V2SIG_POLARITY_2HZ_BAND_OVERRIDE`: Replaced the `1.5 <= freq <= 2.4` Hz gate with continuous duty-cycle weighting.
- **Simplified and Merged (`SIMPLIFY_AND_MERGE`)**:
  - `V2SIG_POLARITY_3HZ_RULES_4A_4E`: Consolidated the 5 high-frequency polarity branches into a unified multi-cue morphological evidence model.
  - `QCV2_BIDIRECTIONAL_CONFLICT` & `QCV2_LAYER2_NOISE_DOMINATED_TREE`: Simplified `qc_v2.py` to remove the narrow 6-parameter carve-out box and consolidate noise-domination checks.
  - `SARC_DET_DEAD_FIRST_PASS_OVERWRITE` & `SARC_DET_TRIPLE_CLASSIFICATION_PASS`: Removed the dead first pass and unified the 3 conflicting classification passes in `sarcomere_detector.py` into one authoritative classification pass.
  - Removed the 1,700-line dead `SARCOMERE_HTML_TEMPLATE` from `pixel_counter/sarcomere_review.py`.
- **Parameterized and Exposed (`PARAMETERIZE_AND_EXPOSE`)**:
  - `SARC_QC_FFT_RAIL_LIMITS`: Exposed via `SarcomereMeasurementProfile`.
  - `CAL_NOISE_MULT_MAGIC_043`: Replaced hidden `* 0.43` factor with explicit `raw_prominence_sigma` parameter on `CalciumDetectorParams`.
  - `IO_FS_DEFAULT_FALLBACK`: Exposed provenance metadata in `load_cytocypher_excel`.
