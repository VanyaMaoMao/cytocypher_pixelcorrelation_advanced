# Unresolved Issues, Validation Limitations, and Risk Register

This document records all remaining scientific, experimental, and engineering limitations identified during the project-wide scientific integrity audit, ranked by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).

---

## 1. Summary Table

| Issue ID | Severity | Domain | Title | Status |
|---|---|---|---|---|
| `U001` | **CRITICAL** | External Validation | Lack of Untouched Independent Holdout Datasets | Open (Requires New Experimental Acquisition) |
| `U002` | **HIGH** | Ground Truth | Single-Observer Manual Reference Annotations Without Blinded Inter-Rater Consensus | Open (Requires Multi-Observer Blinded Protocol) |
| `U003` | **HIGH** | Sarcomere Physics | Unobservable True Peak Amplitude During Systolic FFT Tracking Dropout | Mitigated by Design (`NaN` Amplitude / `TIMING_ONLY_UNMEASURED_PEAK`) |
| `U004` | **MEDIUM** | Optical Calibration | Hardware-Specific IonOptix FFT Search-Window Rail Bounds (`1.28–2.18 um`) | Mitigated via Configurable `SarcomereMeasurementProfile` |
| `U005` | **MEDIUM** | PixelCorrelation Physics | Polarity Ambiguity Under Near-Symmetric High-Frequency Sinusoidal Motion (`>= 3.5 Hz`) | Mitigated via `UNDETERMINED` Fail-Safe & Manual Review |
| `U006` | **LOW** | Legacy Prototype Tests | Five Deleted Pre-`316b9cc` Prototype Test Files in Working Tree (`git status`) | Documented & Superseded by Current Suite |

---

## 2. Detailed Issue Analysis and Recommended Next Steps

### `U001` [CRITICAL] — Lack of Untouched Independent Holdout Datasets
- **Scientific Problem**: Every experimental workbook currently present in the repository (`CAR DISH1`, `ISO DISH1`, `IMR90 25092026` 1 Hz / 2 Hz / 3 Hz, `DISH1CONTROL`, `DISH2ICI`, `DISH1NICOTINE`, `240926_test`, `DISH1TEST`) was previously inspected, discussed in iterative feedback prompts, or used for diagnostic error analysis during algorithm development.
- **Why Internal Regression Cannot Substitute for Independent Validation**: Even after removing all dataset-specific heuristics, magic numbers, and segment-number carve-outs, testing on previously inspected recordings measures **development regression stability**, not out-of-sample generalization error.
- **Explicit Statement**: *Independent scientific validation cannot yet be completed with the available previously inspected recordings.*
- **Required Next Step**:
  1. Acquire a sealed prospective validation cohort of at least $N \ge 3$ independent dishes across distinct pharmacological conditions (e.g., untreated control, $\beta$-adrenergic stimulation, and pathological arrhythmia/heart-failure models) and pacing frequencies ($0.5\text{–}3\text{ Hz}$).
  2. Freeze all production algorithm code and parameters (`SHA-256` hash of `pixel_counter/`) **prior** to unblinding the holdout workbooks.
  3. Execute the batch pipeline once on the locked holdout cohort and report sensitivity, positive predictive value (PPV), amplitude bias, and binary QC accuracy without any post-hoc parameter adjustment.

---

### `U002` [HIGH] — Single-Observer Manual Reference Annotations Without Blinded Inter-Rater Consensus
- **Scientific Problem**: Reference expectations for MAIN vs. SECONDARY (SCE) contractions and binary segment QC (`PASS` vs. `REJECT`) in the development workbooks originated from unblinded single-expert visual inspection of algorithm outputs during iterative development. Near the boundary between large secondary contractions and reduced-amplitude primary contractions ($0.50\text{–}0.65\,A_{\text{ref}}$), human observers frequently disagree on whether an event is a `MAIN` beat or a `SECONDARY` after-contraction.
- **Required Next Step**:
  1. Use the interactive HTML review utility to collect blinded annotations from at least two independent cardiac electrophysiologists/cell physiologists on raw traces (with algorithm markers initially hidden).
  2. Quantify inter-rater agreement using Cohen's $\kappa$ for binary segment QC and MAIN/SCE event classification, and intraclass correlation (ICC) for contraction amplitudes.

---

### `U003` [HIGH] — Unobservable True Peak Amplitude During Systolic FFT Tracking Dropout
- **Scientific Problem**: In adult cardiomyocyte sarcomere recordings with vigorous contractions, rapid striation translation during peak systole can momentarily cause the IonOptix real-time FFT tracker to lose lock right at the contraction nadir ($L_{\min}$).
- **Current Scientific Mitigation**: Rather than fabricating a peak shortening amplitude from spline interpolation (`reconstruct_narrow_tracking_artifacts`), any accepted contraction whose trough nadir overlaps a tracking artifact (`peak_corrupted == True`) is assigned `l_min_um = NaN`, `abs_shortening_um = NaN`, `pct_shortening = NaN`, and `amplitude_source = "TIMING_ONLY_UNMEASURED_PEAK"`. It contributes to event timing and frequency (`n_accepted`) when flanked by genuine shortening shoulders, but is strictly excluded from amplitude means (`n_directly_measured`, `n_valid_amplitude`).
- **Residual Limitation**: If tracking dropout systematically occurs on the strongest contractions within a segment, the segment's mean shortening amplitude (computed over the uncorrupted beats) may underestimate the cell's maximum contractile capacity. The exported summary explicitly reports `n_directly_measured` and `n_lacking_amplitude` so analysts can filter or weight segments accordingly.

---

### `U004` [MEDIUM] — Hardware-Specific IonOptix FFT Search-Window Rail Bounds (`1.28–2.18 um`)
- **Scientific Problem**: The default FFT rail clamping thresholds (`fft_rail_min_um = 1.28`, `fft_rail_max_um = 2.18`) correspond to standard IonOptix SarcLen optical ROI spatial-frequency bounds for mammalian ventricular cardiomyocytes ($L_0 \approx 1.70\text{–}1.95\ \mu\text{m}$). If a user acquires recordings with a modified FFT search window (e.g., hypercontracted cells at $1.20\ \mu\text{m}$ or stretched skeletal myofibrils at $2.40\ \mu\text{m}$), the default profile would misclassify valid lengths outside $[1.28, 2.18]\ \mu\text{m}$ as rail artifacts.
- **Current Scientific Mitigation**: Encapsulated all physical and optical bounds inside `SarcomereMeasurementProfile` (`pixel_counter/sarcomere_qc.py`), allowing custom rail bounds and velocity limits to be passed programmatically without modifying detector code.

---

### `U005` [MEDIUM] — Polarity Ambiguity Under Near-Symmetric High-Frequency Sinusoidal Motion (`>= 3.5 Hz`)
- **Scientific Problem**: In PixelCorrelation recordings lacking an auxiliary calcium channel, automatic polarity relies on physiological asymmetries between systole and diastole (resting dwell plateau, upstroke vs. relaxation velocity, extremum curvature, and envelope stability). At very high pacing rates ($\ge 3.5\text{ Hz}$) or in severely myopathic cells with quasi-sinusoidal motion, both duty cycle and derivative skewness approach symmetry.
- **Current Scientific Mitigation**: When cues conflict or lack decisive asymmetry, `determine_raw_polarity` refuses to guess and returns `polarity = 0` (`status = "UNDETERMINED"`), prompting manual polarity verification in the interactive review utility or resolution via excitation–contraction coupling latency (`_calcium_ecc_polarity`) when paired calcium is available.

---

### `U006` [LOW] — Five Deleted Pre-`316b9cc` Prototype Test Files in Working Tree
- **Engineering Context**: Five legacy test files (`tests/test_detect_events_v2.py`, `tests/test_polarity_regression.py`, `tests/test_qc_v2.py`, `tests/test_step16_refinement.py`, `tests/test_step17_polarity.py`) were deleted in the working tree prior to this audit because they imported removed prototype symbols (`detect_events_v2`, `auto_select_polarity_v2`, `v2_two_pass_enabled`, and `"REVIEW"` QC states) superseded by commit `316b9cc`.
- **Current Status**: Per the strict non-destructive Git policy, working-tree deletions were left untouched (`git status` preserved), and all underlying scientific invariants are now tested by `tests/test_v2_detector.py`, `tests/test_v2_polarity_qc.py`, `tests/test_v2_robustness_improvements.py`, and `tests/test_scientific_integrity_metamorphic.py`.
