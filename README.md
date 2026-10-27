# Cytocypher PixelCorrelation, Calcium & Sarcomere Shortening Analyzer

Integrated scientific Python suite for automated cardiomyocyte contractility, sarcomere shortening, and intracellular $\text{Ca}^{2+}$ transient analysis on IonOptix CytoCypher segment workbooks.

---

## 1. Scientific Capabilities & Architecture

The application provides three independent, signal-processing-driven analysis pipelines unified under a single interactive HTML review and reporting framework (`run_v2_review_utility.py`):

1. **PixelCorrelation Contraction Analyzer (`pixel_counter/v2_signal.py`, `pixel_counter/v2_detector.py`, `pixel_counter/qc_v2.py`)**:
   - Automatically determines physiological contraction polarity ($+1$ upward vs. $-1$ downward) using antisymmetric multi-cue morphology (resting dwell plateau, upstroke/relaxation velocity asymmetry, extremum curvature, and envelope stability) with continuous cycle-compression scaling.
   - Safely abstains with `UNDETERMINED` (`polarity = 0`) when cues conflict, rather than guessing an inverted orientation.
   - Supports optional excitation–contraction coupling (`--use-calcium-polarity`) latency verification when paired `RatioMetricCalcium` sheets are available.
   - Evaluates 8 deterministic binary segment QC criteria (`PASS` / `REJECT`), including clean quiescent recording recognition (`PASS` with `0` events).

2. **Calcium Transient Analyzer (`pixel_counter/calcium_detector.py`, `pixel_counter/calcium_review.py`)**:
   - Processes ratiometric Fura-2 (`RatioMetricCalcium Segment N`) waveforms with automatic sensor-dropout masking, rolling lower-percentile diastolic baseline $F_0(t)$ estimation, and prominence-protected fractional normalization $\Delta R / \max(R_0, \epsilon)$.
   - Supports **Standard Mode** (single-threshold peak detection) and **Adult Cardiomyocyte Mode** (`MAIN` vs. `SECONDARY` / `SCE` dual-threshold hierarchy $T_{\text{MAIN,Ca}} \ge T_{\text{SCE,Ca}}$).

3. **Sarcomere Shortening Analyzer (`pixel_counter/sarcomere_detector.py`, `pixel_counter/sarcomere_qc.py`, `pixel_counter/sarcomere_artifact_recovery.py`, `pixel_counter/sarcomere_pipeline.py`)**:
   - Detects downward sarcomere shortening troughs ($L_0(t) \to L_{\min}$) using a two-pass rolling diastolic percentile baseline $L_0(t)$ and enforces a strict dual-threshold amplitude hierarchy ($T_{\text{MAIN}} \ge T_{\text{SCE}}$) classifying contractions into `MAIN` and `SECONDARY` (`SCE`).
   - Encapsulates hardware optical FFT bounds and physical velocity limits inside `SarcomereMeasurementProfile` (`fft_rail_min_um=1.28`, `fft_rail_max_um=2.18`, `jump_vel_um_per_s=8.75`, `extreme_vel_um_per_s=13.75`), scaled by sampling interval $\Delta t = 1/f_s$.
   - Performs conservative local Hermite (`PchipInterpolator`) narrow-gap reconstruction ($\le 80\text{ ms}$) for timing recovery while strictly refusing to fabricate peak amplitudes when a trough nadir is corrupted (`NaN` amplitude, `TIMING_ONLY_UNMEASURED_PEAK`).
   - Enforces **strict channel independence**: in `sarcomere-calcium` mode, the calcium trace never alters sarcomere detection, classification, amplitudes, or sarcomere QC.

---

## 2. Requirements & Installation

- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pandas`, `openpyxl`, `plotly`, `pytest`

```powershell
cd "C:\Users\talif\PyCharmProjects\cytocypher_v2_wt"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Command-Line Usage (`run_v2_review_utility.py`)

### 3.1 PixelCorrelation Analysis (Default Mode)
```powershell
# PixelCorrelation-only mode (default):
python run_v2_review_utility.py "path\to\workbook.xlsx" --output review_pixel_only.html

# Optional Calcium-assisted polarity mode:
python run_v2_review_utility.py "path\to\workbook.xlsx" --use-calcium-polarity --output review_calcium_assisted.html

# Console polarity inspection table only:
python run_v2_review_utility.py "path\to\workbook.xlsx" --inspect-polarity
```

### 3.2 Sarcomere Shortening & Paired Sarcomere–Calcium Analysis
```powershell
# Sarcomere-only analysis and interactive HTML review:
python run_v2_review_utility.py "path\to\sarcomere_workbook.xlsx" --mode sarcomere --output review_sarcomere.html

# Paired Sarcomere + Adult Calcium independent dual-channel analysis:
python run_v2_review_utility.py "path\to\sarcomere_workbook.xlsx" --mode sarcomere-calcium --output review_sarcomere_calcium.html
```

### 3.3 Calcium Transient Analysis
```powershell
# Standalone Calcium transient analysis:
python run_v2_review_utility.py "path\to\workbook.xlsx" --mode calcium --output review_calcium.html
```

### 3.4 Manual Polarity & QC Sidecar Overrides
```powershell
# Override individual segments from CLI and save sidecar JSON:
python run_v2_review_utility.py "workbook.xlsx" --override-polarity "17=-1,18=-1,45=1" --save-overrides my_overrides.json

# Load saved overrides JSON:
python run_v2_review_utility.py "workbook.xlsx" --overrides my_overrides.json
```

---

## 4. Scientific Integrity Audit & Documentation

Complete mathematical specifications, forensic audit inventories, refactor changelogs, and development regression reports are maintained in `docs/scientific_integrity_audit/`:

- [`docs/scientific_integrity_audit/rule_inventory.csv`](docs/scientific_integrity_audit/rule_inventory.csv): Complete 32-rule catalog across all production modules with physical units, physiological justifications, and audit classifications.
- [`docs/scientific_integrity_audit/overfitting_audit.md`](docs/scientific_integrity_audit/overfitting_audit.md): Forensic audit of dataset-specific heuristics, magic numbers, and circular tests identified at baseline `ccdc1c8`.
- [`docs/scientific_integrity_audit/scientific_methods.md`](docs/scientific_integrity_audit/scientific_methods.md): Mathematical and physiological specification of all algorithms, baselines, dual thresholds, and QC criteria.
- [`docs/scientific_integrity_audit/refactor_changes.md`](docs/scientific_integrity_audit/refactor_changes.md): File-by-file and rule-by-rule engineering report of the anti-overfitting refactor.
- [`docs/scientific_integrity_audit/validation_report.md`](docs/scientific_integrity_audit/validation_report.md): Development regression results, before/after comparisons, metamorphic test results, and runtime/memory profiling.
- [`docs/scientific_integrity_audit/unresolved_issues.md`](docs/scientific_integrity_audit/unresolved_issues.md): Ranked risk register and external holdout validation requirements.

---

## 5. Running the Test Suite

```powershell
# Run full test suite:
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.pytest_temp

# Run the 15-category Scientific Integrity & Metamorphic Property Suite:
.\.venv\Scripts\python.exe -m pytest tests/test_scientific_integrity_metamorphic.py -v --basetemp=.pytest_temp
```
