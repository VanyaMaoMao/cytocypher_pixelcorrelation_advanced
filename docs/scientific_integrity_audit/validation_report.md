# Algorithm Validation & Development Regression Report

> [!IMPORTANT]
> **SCIENTIFIC INTEGRITY & VALIDATION SCOPE STATEMENT**
> **Independent scientific validation cannot yet be completed with the available previously inspected recordings.**
> Every experimental workbook currently present in the workspace (`CAR DISH1`, `ISO DISH1`, `IMR90 25092026`, `DISH1CONTROL`, `DISH2ICI`, `DISH1NICOTINE`, `240926_test`, `DISH1TEST`) was previously inspected during iterative development and error analysis. All results on these workbooks are therefore reported strictly as **DEVELOPMENT REGRESSION RESULTS**, while out-of-sample algorithmic generalization is evaluated via the **Synthetic & Metamorphic Property Test Suite (`tests/test_scientific_integrity_metamorphic.py`)**.

---

## 1. Executive Summary of Validation & Regression Outcomes

| Validation Tier | Dataset / Suite | Classification | Total Items | Agreement / Pass Rate | Notes |
|---|---|---|---|---|---|
| **Tier 1: Metamorphic & Property Suite** | `tests/test_scientific_integrity_metamorphic.py` | **Synthetic / Perturbation Generalization** | 18 test cases (15 categories) | **18 / 18 PASS (100%)** | Tests time-origin, row-partition, amplitude scale, baseline offset, `fs` resampling (`125/250/500 Hz`), quiescent vs. noise QC, PCHIP amplitude honesty, NaN safety, polarity antisymmetry, channel independence, filename/segment-ID invariance, preset monotonicity, and `SarcomereMeasurementProfile`. |
| **Tier 2: Full Automated Pytest Suite** | `tests/` (28 test modules) | **Unit, Integration & Regression** | 288 collected | **279 PASS, 9 XFAIL, 0 FAIL** | Includes all 261 baseline tests plus 18 new scientific integrity metamorphic tests. |
| **Tier 3: Development Regression — Dataset A** | `(MI) CGP+CAR+ROLI+IBMX DISH1` | **DEVELOPMENT REGRESSION RESULTS** | 52 Sarcomere + 52 Calcium segments | **52 / 52 (100% QC & Beat Count Agreement)** | 44 PASS / 8 REJECT (Sarcomere); 46 PASS / 6 REJECT (Adult Calcium). |
| **Tier 3: Development Regression — Dataset B** | `(MI) CGP+ISO+ROLI+IBMX DISH1` | **DEVELOPMENT REGRESSION RESULTS** | 50 Sarcomere + 50 Calcium segments | **50 / 50 (100% QC & Beat Count Agreement)** | 32 PASS / 18 REJECT (Sarcomere); 44 PASS / 6 REJECT (Adult Calcium). |
| **Tier 3: Development Regression — Dataset C** | `25092026_IMR90_1,2,3HZ` | **DEVELOPMENT REGRESSION RESULTS** | 45 PixelCorrelation + 45 Calcium segments | **45 / 45 (100% QC, Polarity & Beat Count Agreement)** | 27 PASS / 18 REJECT (PixelCorrelation); 37 PASS / 8 REJECT (Standard Calcium). |

---

## 2. Metamorphic & Invariance Property Test Results (`tests/test_scientific_integrity_metamorphic.py`)

| Category # | Test Function | Mathematical / Physical Invariant Verified | Status |
|---|---|---|---|
| 1 | `test_1_time_origin_invariance_all_analyzers` | Shifting `t0 -> t0 + 500.25 s` shifts all event timestamps and reconstructed intervals by `+500.25 s` (`< 1e-9 s` error) while preserving event counts, amplitudes, and QC across PixelCorrelation, Calcium, and Sarcomere. | **PASS** |
| 2 | `test_2_row_partition_invariance[125, 250, 500]` | Partitioning a multi-beat recording into 125-, 250-, or 500-sample `Transient` rows produces bit-for-bit identical stitched signals, timestamps, and amplitudes. | **PASS (3/3)** |
| 3 | `test_3_amplitude_scale_behavior` | Positive scaling invariance in PixelCorrelation (`c * x`), affine scaling (`c * R`) invariance of `Delta R / R0` in Calcium, and linear scaling of `pct_shortening` in Sarcomere. | **PASS** |
| 4 | `test_4_baseline_offset_behavior` | Additive constant shift invariance (`x + c`) in PixelCorrelation, expected denominator shift in Calcium `Delta R / R0`, and inverse baseline shift in Sarcomere `100 * Delta L / (L0 + c)`. | **PASS** |
| 5 | `test_5_sampling_rate_consistency[125.0, 250.0, 500.0]` | Resampling identical physiological waveforms at `125 Hz`, `250 Hz`, and `500 Hz` yields identical QC, beat counts, MAIN/SCE classifications, and shortening amplitudes (`< 0.15%` tolerance). | **PASS (3/3)** |
| 6 | `test_6_clean_quiescent_vs_noise_dominated_all_analyzers` | Clean quiescent traces (`0` contractions, low noise floor) receive `PASS` with `0` events across all three analyzers, whereas high-noise peakless traces receive `REJECT`. | **PASS** |
| 7 | `test_7_artifact_injection_and_direct_amplitude_accounting` | Injecting a 12 ms FFT rail spike at a contraction nadir marks `peak_corrupted = True`, sets `pct_shortening = NaN` (`TIMING_ONLY_UNMEASURED_PEAK`), and sets `n_directly_measured = 3`, `n_lacking_amplitude = 1` without fabricating amplitude. | **PASS** |
| 8 | `test_8_nan_injection_safety` | Short inter-beat NaN gaps do not crash or fabricate events (`PASS`, `3` beats); `> 60%` NaN segments cleanly `REJECT`. | **PASS** |
| 9 | `test_9_input_order_independence` | Analyzing recordings in forward vs. reverse order yields bit-for-bit identical results (zero hidden global state). | **PASS** |
| 10 | `test_10_polarity_inversion_antisymmetry` | Negating a PixelCorrelation trace (`x -> -x`) across `1 Hz`, `2 Hz`, and `3 Hz` pacing strictly negates automatic polarity (`+1 <-> -1`). | **PASS** |
| 11 | `test_11_calcium_channel_independence` | Running Sarcomere analysis with no calcium, normal calcium, or inverted calcium produces identical sarcomere QC, beat counts, and `mean_pct_shortening` (`< 1e-12`). | **PASS** |
| 12 & 13 | `test_12_and_13_filename_and_segment_id_invariance` | Renaming a workbook from `(MI) CGP+CAR+ROLI+IBMX DISH1...xlsx` to `SCRAMBLED_NOISE_ANONYMOUS_HOLDOUT_999.xlsx` and `Segment 1` to `Segment 987` produces identical outputs. | **PASS** |
| 14 | `test_14_preset_monotonicity_and_custom_params_honored` | Passing custom `SarcomereDetectorParams` or `CalciumDetectorParams` with custom `preset_name` and elevated floors is honored without `KeyError` and enforces the custom thresholds. | **PASS** |
| 15 | `test_15_configurable_measurement_profile_and_sidecar_roundtrip` | Customizing `SarcomereMeasurementProfile(fft_rail_max_um=2.35)` reconfigures rail detection as expected, and polarity/rejection JSON sidecars round-trip cleanly. | **PASS** |

---

## 3. Computational Performance & Memory Profiling

Measured on Python 3.14.5 (`win32`) using `time.perf_counter()` and `tracemalloc` across full workbook batch runs:

| Workbook | Mode | Segments Analyzed | Total Runtime (s) | Mean Time per Segment (ms) | Peak Traced Memory (MB) |
|---|---|---|---|---|---|
| `CAR_DISH1` | Sarcomere + Adult Calcium (3 presets each) | 52 | 85.68 s | 1647.6 ms | 18.15 MB |
| `ISO_DISH1` | Sarcomere + Adult Calcium (3 presets each) | 50 | 97.08 s | 1941.6 ms | 12.66 MB |
| `IMR90_25092026` | PixelCorrelation v2 + Standard Calcium | 45 | 12.62 s | 280.4 ms | 4.56 MB |

---

## 4. Development Regression Results: Before (`ccdc1c8`) vs. After Anti-Overfitting Refactor

> [!WARNING]
> **HONEST LABELING NOTICE**: The tables below represent **DEVELOPMENT REGRESSION RESULTS** on previously inspected workbooks. They verify that replacing ad-hoc rules and dataset-specific constants with principled signal-processing formulations preserves regression stability on known recordings, and must **not** be interpreted as independent holdout validation.

### 4.1 QC Confusion Matrix (Before `ccdc1c8` vs. After Refactor)

| Dataset & Channel | `PASS -> PASS` | `REJECT -> REJECT` | `PASS -> REJECT` | `REJECT -> PASS` | Total Segments | Agreement |
|---|---|---|---|---|---|---|
| `CAR DISH1` — Sarcomere | 44 | 8 | 0 | 0 | 52 | **100.0%** |
| `CAR DISH1` — Adult Calcium | 46 | 6 | 0 | 0 | 52 | **100.0%** |
| `ISO DISH1` — Sarcomere | 32 | 18 | 0 | 0 | 50 | **100.0%** |
| `ISO DISH1` — Adult Calcium | 44 | 6 | 0 | 0 | 50 | **100.0%** |
| `IMR90 25092026` — PixelCorrelation | 27 | 18 | 0 | 0 | 45 | **100.0%** |
| `IMR90 25092026` — Standard Calcium | 37 | 8 | 0 | 0 | 45 | **100.0%** |

### 4.2 Development Regression Table — Dataset A (`CAR DISH1`, 52 Segments)

| Seg | Sarc QC (Before -> After) | Sarc Reason | Sarc Accepted (`Main / SCE`) Before | Sarc Accepted (`Main / SCE`) After | Mean `% Short` Before | Mean `% Short` After | Cal QC (Before -> After) | Cal Accepted (`Main / SCE`) Before -> After |
|---|---|---|---|---|---|---|---|---|
| 1 | `REJECT -> REJECT` | `excessive_reconstruction_dependence` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 2 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 1.936% | 1.936% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 3 | `REJECT -> REJECT` | `excessive_reconstruction_dependence` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 4 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 0.657% | 0.657% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 5 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 4.222% | 4.222% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 6 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 3.496% | 3.496% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 7 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 3.105% | 3.105% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 8 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 1.130% | 1.130% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 9 | `PASS -> PASS` | `valid_contractions` | 10 (`6/4`) | 10 (`6/4`) | 0.701% | 0.701% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 10 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 0.472% | 0.472% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 11 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 13 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 4.046% | 4.046% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 15 | `PASS -> PASS` | `valid_contractions` | 9 (`9/0`) | 9 (`9/0`) | 7.975% | 7.975% | `PASS -> PASS` | 9 (`9/0`) -> 9 (`9/0`) |
| 16 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 8.143% | 8.143% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 21 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 22 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 23 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 24 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 25 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 26 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 27 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 28 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 2.751% | 2.751% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 29 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 2.758% | 2.758% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 30 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 31 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 32 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 33 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 34 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 35 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 36 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 37 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 38 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 39 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 3.584% | 3.584% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 40 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 41 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 1.971% | 1.971% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 42 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 43 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 44 | `REJECT -> REJECT` | `insufficient_valid_measurements` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 45 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 46 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 47 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 48 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 49 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 50 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 51 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 52 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 53 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 54 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 55 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 56 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 57 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 58 | `REJECT -> REJECT` | `noise_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |

### 4.3 Development Regression Table — Dataset B (`ISO DISH1`, 50 Segments)

| Seg | Sarc QC (Before -> After) | Sarc Reason | Sarc Accepted (`Main / SCE`) Before | Sarc Accepted (`Main / SCE`) After | Mean `% Short` Before | Mean `% Short` After | Cal QC (Before -> After) | Cal Accepted (`Main / SCE`) Before -> After |
|---|---|---|---|---|---|---|---|---|
| 2 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 14.031% | 14.031% | `PASS -> PASS` | 10 (`8/2`) -> 10 (`8/2`) |
| 3 | `PASS -> PASS` | `valid_contractions` | 9 (`9/0`) | 9 (`9/0`) | 9.499% | 9.499% | `PASS -> PASS` | 9 (`8/1`) -> 9 (`8/1`) |
| 4 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 4.498% | 4.498% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 5 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 9.819% | 9.819% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 6 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 5.343% | 5.343% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 7 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 8 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 2.908% | 2.908% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 9 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 3.612% | 3.612% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 10 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 2.523% | 2.523% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 11 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 10.170% | 10.170% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 12 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 13 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 12.779% | 12.779% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 14 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 15 | `PASS -> PASS` | `valid_contractions` | 10 (`10/0`) | 10 (`10/0`) | 14.378% | 14.378% | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 16 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 43 (`43/0`) -> 43 (`43/0`) |
| 17 | `PASS -> PASS` | `valid_contractions` | 35 (`35/0`) | 35 (`35/0`) | 7.222% | 7.222% | `PASS -> PASS` | 36 (`36/0`) -> 36 (`36/0`) |
| 18 | `PASS -> PASS` | `valid_contractions` | 39 (`39/0`) | 39 (`39/0`) | 6.069% | 6.069% | `PASS -> PASS` | 39 (`39/0`) -> 39 (`39/0`) |
| 19 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 10 (`10/0`) -> 10 (`10/0`) |
| 20 | `PASS -> PASS` | `valid_contractions` | 23 (`22/1`) | 23 (`22/1`) | 8.702% | 8.702% | `PASS -> PASS` | 22 (`13/9`) -> 22 (`13/9`) |
| 21 | `PASS -> PASS` | `valid_contractions` | 16 (`10/6`) | 16 (`10/6`) | 8.881% | 8.881% | `PASS -> PASS` | 11 (`11/0`) -> 11 (`11/0`) |
| 22 | `PASS -> PASS` | `valid_contractions` | 18 (`17/1`) | 18 (`17/1`) | 12.307% | 12.307% | `PASS -> PASS` | 15 (`12/3`) -> 15 (`12/3`) |
| 23 | `PASS -> PASS` | `valid_contractions` | 17 (`15/2`) | 17 (`15/2`) | 8.943% | 8.943% | `PASS -> PASS` | 11 (`11/0`) -> 11 (`11/0`) |
| 24 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 25 | `PASS -> PASS` | `valid_contractions` | 18 (`15/3`) | 18 (`15/3`) | 9.179% | 9.179% | `PASS -> PASS` | 18 (`12/6`) -> 18 (`12/6`) |
| 26 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 24 (`24/0`) -> 24 (`24/0`) |
| 27 | `PASS -> PASS` | `valid_contractions` | 36 (`35/1`) | 36 (`35/1`) | 5.060% | 5.060% | `PASS -> PASS` | 35 (`35/0`) -> 35 (`35/0`) |
| 28 | `PASS -> PASS` | `valid_contractions` | 36 (`36/0`) | 36 (`36/0`) | 5.543% | 5.543% | `PASS -> PASS` | 36 (`36/0`) -> 36 (`36/0`) |
| 29 | `PASS -> PASS` | `valid_contractions` | 8 (`1/7`) | 8 (`1/7`) | 1.025% | 1.025% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 30 | `PASS -> PASS` | `valid_contractions` | 37 (`37/0`) | 37 (`37/0`) | 5.641% | 5.641% | `PASS -> PASS` | 37 (`37/0`) -> 37 (`37/0`) |
| 31 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 23 (`23/0`) -> 23 (`23/0`) |
| 32 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 12 (`12/0`) -> 12 (`12/0`) |
| 33 | `PASS -> PASS` | `valid_contractions` | 21 (`14/7`) | 21 (`14/7`) | 9.019% | 9.019% | `PASS -> PASS` | 13 (`13/0`) -> 13 (`13/0`) |
| 34 | `REJECT -> REJECT` | `no_valid_sarcomere_signal` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `UNAVAILABLE -> UNAVAILABLE` | 0 (`0/0`) -> 0 (`0/0`) |
| 35 | `PASS -> PASS` | `valid_contractions` | 19 (`13/6`) | 19 (`13/6`) | 8.580% | 8.580% | `PASS -> PASS` | 19 (`10/9`) -> 19 (`10/9`) |
| 36 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 8 (`5/3`) -> 8 (`5/3`) |
| 37 | `PASS -> PASS` | `valid_contractions` | 31 (`31/0`) | 31 (`31/0`) | 4.173% | 4.173% | `PASS -> PASS` | 31 (`31/0`) -> 31 (`31/0`) |
| 38 | `PASS -> PASS` | `valid_contractions` | 28 (`25/3`) | 28 (`25/3`) | 3.218% | 3.218% | `PASS -> PASS` | 24 (`22/2`) -> 24 (`22/2`) |
| 39 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 3.808% | 3.808% | `PASS -> PASS` | 1 (`1/0`) -> 1 (`1/0`) |
| 40 | `PASS -> PASS` | `valid_contractions` | 32 (`24/8`) | 32 (`24/8`) | 7.564% | 7.564% | `PASS -> PASS` | 27 (`20/7`) -> 27 (`20/7`) |
| 41 | `PASS -> PASS` | `valid_contractions` | 32 (`24/8`) | 32 (`24/8`) | 7.039% | 7.039% | `PASS -> PASS` | 13 (`11/2`) -> 13 (`11/2`) |
| 42 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 28 (`28/0`) -> 28 (`28/0`) |
| 43 | `PASS -> PASS` | `valid_contractions` | 21 (`14/7`) | 21 (`14/7`) | 5.932% | 5.932% | `PASS -> PASS` | 11 (`11/0`) -> 11 (`11/0`) |
| 44 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 45 | `PASS -> PASS` | `valid_contractions` | 22 (`14/8`) | 22 (`14/8`) | 6.915% | 6.915% | `PASS -> PASS` | 13 (`13/0`) -> 13 (`13/0`) |
| 46 | `REJECT -> REJECT` | `tracking_artifact_dominated` | 0 (`0/0`) | 0 (`0/0`) | NaN | NaN | `PASS -> PASS` | 7 (`6/1`) -> 7 (`6/1`) |
| 47 | `PASS -> PASS` | `valid_contractions` | 34 (`18/16`) | 34 (`18/16`) | 1.751% | 1.751% | `PASS -> PASS` | 23 (`21/2`) -> 23 (`21/2`) |
| 48 | `PASS -> PASS` | `valid_contractions` | 27 (`18/9`) | 27 (`18/9`) | 2.383% | 2.383% | `PASS -> PASS` | 23 (`20/3`) -> 23 (`20/3`) |
| 49 | `PASS -> PASS` | `valid_contractions` | 1 (`1/0`) | 1 (`1/0`) | 5.169% | 5.169% | `REJECT -> REJECT` | 0 (`0/0`) -> 0 (`0/0`) |
| 50 | `PASS -> PASS` | `valid_contractions` | 36 (`31/5`) | 36 (`31/5`) | 6.403% | 6.403% | `PASS -> PASS` | 32 (`27/5`) -> 32 (`27/5`) |
| 51 | `PASS -> PASS` | `valid_contractions` | 32 (`31/1`) | 32 (`31/1`) | 8.516% | 8.516% | `PASS -> PASS` | 18 (`14/4`) -> 18 (`14/4`) |

### 4.4 Development Regression Table — Dataset C (`IMR90 25092026`, 45 Segments)

| Seg | Pixel QC (Before -> After) | Polarity (Before -> After) | Polarity Method | Pixel Beats (Before -> After) | Cal QC (Before -> After) | Cal Beats (Before -> After) |
|---|---|---|---|---|---|---|
| 1 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 2 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 3 | `PASS -> PASS` | `-1 -> -1` | `extremum_curvature` | 4 -> 4 | `PASS -> PASS` | 5 -> 5 |
| 4 | `PASS -> PASS` | `-1 -> -1` | `resting_dwell` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 5 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 4 -> 4 | `PASS -> PASS` | 5 -> 5 |
| 6 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 4 -> 4 | `PASS -> PASS` | 4 -> 4 |
| 7 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 8 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 9 | `PASS -> PASS` | `-1 -> -1` | `resting_dwell` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 10 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 11 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 4 -> 4 | `PASS -> PASS` | 4 -> 4 |
| 12 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 4 -> 4 | `PASS -> PASS` | 5 -> 5 |
| 13 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 14 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 5 -> 5 | `PASS -> PASS` | 5 -> 5 |
| 15 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 4 -> 4 | `PASS -> PASS` | 4 -> 4 |
| 16 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 9 -> 9 | `PASS -> PASS` | 10 -> 10 |
| 17 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 9 -> 9 |
| 18 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 10 -> 10 |
| 19 | `PASS -> PASS` | `-1 -> -1` | `envelope_stability` | 9 -> 9 | `PASS -> PASS` | 10 -> 10 |
| 20 | `PASS -> PASS` | `-1 -> -1` | `extremum_curvature` | 9 -> 9 | `PASS -> PASS` | 9 -> 9 |
| 21 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 10 -> 10 | `PASS -> PASS` | 10 -> 10 |
| 22 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 9 -> 9 | `PASS -> PASS` | 9 -> 9 |
| 23 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 10 -> 10 | `PASS -> PASS` | 10 -> 10 |
| 24 | `PASS -> PASS` | `-1 -> -1` | `resting_dwell` | 9 -> 9 | `PASS -> PASS` | 9 -> 9 |
| 25 | `PASS -> PASS` | `-1 -> -1` | `mild_dwell_and_onset_consensus` | 10 -> 10 | `PASS -> PASS` | 10 -> 10 |
| 26 | `PASS -> PASS` | `-1 -> -1` | `extremum_curvature` | 9 -> 9 | `PASS -> PASS` | 10 -> 10 |
| 27 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 8 -> 8 | `PASS -> PASS` | 8 -> 8 |
| 28 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 9 -> 9 | `PASS -> PASS` | 9 -> 9 |
| 29 | `PASS -> PASS` | `-1 -> -1` | `overwhelming_onset_kinetics` | 9 -> 9 | `PASS -> PASS` | 9 -> 9 |
| 30 | `PASS -> PASS` | `1 -> 1` | `mild_dwell_and_onset_consensus` | 10 -> 10 | `PASS -> PASS` | 10 -> 10 |
| 31 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 13 -> 13 |
| 32 | `PASS -> PASS` | `1 -> 1` | `resting_dwell` | 8 -> 8 | `PASS -> PASS` | 15 -> 15 |
| 33 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 14 -> 14 |
| 34 | `PASS -> PASS` | `-1 -> -1` | `hf_dwell_and_variance_consensus` | 13 -> 13 | `PASS -> PASS` | 13 -> 13 |
| 35 | `PASS -> PASS` | `1 -> 1` | `hf_dwell_and_variance_consensus` | 10 -> 10 | `PASS -> PASS` | 10 -> 10 |
| 36 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 14 -> 14 |
| 37 | `PASS -> PASS` | `-1 -> -1` | `dwell_and_curvature_consensus` | 3 -> 3 | `REJECT -> REJECT` | 0 -> 0 |
| 38 | `PASS -> PASS` | `-1 -> -1` | `hf_dwell_and_variance_consensus` | 14 -> 14 | `PASS -> PASS` | 14 -> 14 |
| 39 | `REJECT -> REJECT` | `1 -> 1` | `dwell_and_curvature_consensus` | 0 -> 0 | `PASS -> PASS` | 15 -> 15 |
| 40 | `PASS -> PASS` | `1 -> 1` | `dwell_and_curvature_consensus` | 7 -> 7 | `PASS -> PASS` | 14 -> 14 |
| 41 | `PASS -> PASS` | `1 -> 1` | `high_freq_kinetic_consensus` | 14 -> 14 | `PASS -> PASS` | 14 -> 14 |
| 42 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 11 -> 11 |
| 43 | `PASS -> PASS` | `-1 -> -1` | `resting_dwell` | 9 -> 9 | `REJECT -> REJECT` | 0 -> 0 |
| 44 | `PASS -> PASS` | `1 -> 1` | `resting_dwell` | 21 -> 21 | `PASS -> PASS` | 15 -> 15 |
| 45 | `REJECT -> REJECT` | `0 -> 0` | `none` | 0 -> 0 | `PASS -> PASS` | 14 -> 14 |
