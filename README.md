# Cytocypher PixelCorrelation Advanced

Advanced scientific Python pipeline for automated cardiomyocyte contraction detection, physiological polarity determination, and contractility analysis on IonOptix CytoCypher segment workbooks.

---

## 1. What This Software Does

The pipeline independently analyzes mechanical cardiomyocyte motion waveforms exported from IonOptix CytoCypher workbooks (`PixelCorrelation Segment N`), transforming raw displacement traces into physiological, baseline-relative contraction amplitude signals:

- **Physiological Polarity Resolution**: Contraction excursions in raw PixelCorrelation may point downward or upward depending on tracking orientation. The software automatically identifies physiological contraction direction and inverts downward traces into positive-going contraction amplitude $A(t) = \text{polarity} \times (S_{\text{raw}} - B_{\text{raw}})$.
- **Default Mode (PixelCorrelation-Only)**: Operates strictly on mechanical motion morphology alone by default. Does not require or depend on auxiliary channels (`RatioMetricCalcium` or `PixelIntensity`).
- **Optional Calcium-Assisted Mode**: When synchronized `RatioMetricCalcium` is available and enabled via `--use-calcium-polarity`, evaluates excitation-contraction coupling (ECC) timing (systolic latency ~35–150 ms) to resolve ambiguous waveforms.
- **Robust Event Detection (`v2_detector`)**: Identifies genuine major contractions with strictness presets (`default`, `strict`, `lenient`), rejecting secondary fluctuations, double-peaking, and baseline noise.
- **Interactive Review Utility (`run_v2_review_utility.py`)**: Standalone, interactive HTML report with embedded Plotly visualizations, live polarity flipping, manual overrides, and CSV annotation export.

---

## 2. Requirements & Installation

- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pandas`, `openpyxl`, `plotly`, `pytest`

```powershell
cd "C:\path\to\cytocypher_pixelcorrelation_advanced"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Cytocypher v2 Review Utility

The primary tool for verifying and analyzing recordings is `run_v2_review_utility.py`.

### Default Mode: PixelCorrelation-Only

By default, the utility processes the workbook using **PixelCorrelation morphology alone**:

```powershell
python run_v2_review_utility.py "path\to\workbook.xlsx" --output review_pixel_only.html
```

- High-frequency pacing ($\ge 2.5\text{ Hz}$, $T_{\text{cycle}} \le 0.40\text{ s}$) automatically engages kinetic gating to avoid false reversals caused by prolonged systolic plateaus.
- Segments with conflicting or ambiguous cues safely abstain with status `UNDETERMINED` (`polarity = 0`) instead of making incorrect guesses.

### Optional Mode: Calcium-Assisted Polarity

When synchronized `RatioMetricCalcium` sheets are present and calcium-assisted polarity resolution is desired:

```powershell
python run_v2_review_utility.py "path\to\workbook.xlsx" --use-calcium-polarity --output review_calcium_assisted.html
```

In this mode:
- The utility loads companion `RatioMetricCalcium Segment N` sheets and evaluates systolic latency and timing consistency across beats.
- Unambiguously resolves high-frequency, symmetric, and challenged waveforms.
- If calcium is flat, absent, or noisy for a particular segment, it falls back seamlessly to PixelCorrelation morphology.

### Stage 1 Polarity Inspection Only

To inspect the polarity classification table in the console without generating full HTML plots:

```powershell
python run_v2_review_utility.py "path\to\workbook.xlsx" --inspect-polarity
```

---

## 4. PyCharm Run Configuration Setup

To configure PyCharm for one-click analysis and review:

1. **Open Run/Debug Configurations**:
   - Go to menu `Run` -> `Edit Configurations...` (or click the configuration dropdown next to the green Play button).
2. **Add New Python Configuration**:
   - Click `+` -> `Python`.
   - **Name**: `Cytocypher v2 Review (Pixel-Only Default)`
   - **Script path**: Select `run_v2_review_utility.py`
   - **Parameters**:
     ```text
     "25092026_IMR90_1,2,3HZ individual transients result.xlsx" --output v2_review_pixel_only.html
     ```
   - **Python interpreter**: Select the project virtualenv (`.venv`).
   - **Working directory**: Project root directory.
3. **Add Calcium-Assisted Configuration (Optional)**:
   - Duplicate the configuration or add a new one named `Cytocypher v2 Review (Calcium-Assisted)`.
   - **Parameters**:
     ```text
     "25092026_IMR90_1,2,3HZ individual transients result.xlsx" --use-calcium-polarity --output v2_review_calcium_assisted.html
     ```
4. Click **Apply** and **OK**. Click the green **Run** button to generate review files directly from PyCharm.

---

## 5. Manual Polarity Overrides

For challenging segments where automated morphology abstains or when visual verification requires an override:

### Via Command-Line
```powershell
# Override individual segments (e.g., Segments 17 and 18 to -1, Segment 45 to +1):
python run_v2_review_utility.py "workbook.xlsx" --override-polarity "17=-1,18=-1,45=1"

# Save active overrides to a reusable JSON file:
python run_v2_review_utility.py "workbook.xlsx" --override-polarity "17=-1,18=-1,45=1" --save-overrides my_overrides.json

# Load overrides from a JSON file:
python run_v2_review_utility.py "workbook.xlsx" --overrides my_overrides.json
```

### Via Interactive Review UI
1. Open the generated HTML file in any browser (Chrome, Edge, Firefox).
2. Use the **Polarity** dropdown to flip the orientation between `Auto`, `+1 (Upward Contraction)`, and `-1 (Downward Contraction)`. The waveform, baseline, and detected peaks update immediately.
3. Click **Export polarity overrides (JSON)** to download a JSON file containing all manual overrides.

---

## 6. Running Tests

The test suite validates detector precision, polarity determination invariances, and auxiliary signal independence:

```powershell
# Run full test suite:
pytest -q tests/

# Run polarity and QC validation:
pytest -q tests/test_v2_polarity_qc.py

# Run review utility tests:
pytest -q tests/test_v2_review_utility.py
```
