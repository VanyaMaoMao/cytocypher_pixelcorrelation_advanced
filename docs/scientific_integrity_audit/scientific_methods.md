# Scientific Methods & Mathematical Specification

## 1. Signal Acquisition, Concatenation, and Sampling-Frequency Provenance

### 1.1 Multi-Row IonOptix CytoCypher Structure
IonOptix CytoCypher Excel workbooks store time-series recordings across one or more row blocks (`Transient` rows) per worksheet (`Segment k`). Each row $i \in \{1, \dots, M\}$ contains:
- Metadata columns: `Sampling Frequency` ($f_{s,\text{decl}}$), `Begin` ($t_{\text{begin}, i}$), `End` ($t_{\text{end}, i}$), and `Transient`.
- Sample columns: $y_0, y_1, \dots, y_{N_i - 1}$.

### 1.2 Sampling-Frequency Provenance and Reconciliation
In `pixel_counter/io_utils.py` (`load_cytocypher_excel`), the sampling rate is determined and audited with explicit provenance in `df.attrs`:
1. **Declared Sampling Frequency ($f_{s,\text{decl}}$)**: Extracted from the `Sampling Frequency` column when present, finite, and $> 0$.
2. **Implied Sampling Frequency ($f_{s,\text{impl}}$)**: Computed from `Begin` and `End` timestamps and valid sample count $N_{\text{valid}}$:
   $$f_{s,\text{impl}} = \frac{N_{\text{valid}} - 1}{t_{\text{end}} - t_{\text{begin}}}$$
   when $\Delta t = t_{\text{end}} - t_{\text{begin}} > 0$ and $N_{\text{valid}} > 1$.
3. **Provenance Classification (`fs_provenance`)**:
   - `"declared"`: $f_{s,\text{decl}}$ is present and consistent with $f_{s,\text{impl}}$ within $10\%$ relative tolerance (or `Begin`/`End` are absent).
   - `"reconciled_declared_over_implied"`: Both $f_{s,\text{decl}}$ and $f_{s,\text{impl}}$ are finite, but $|f_{s,\text{decl}} - f_{s,\text{impl}}| / f_{s,\text{decl}} > 0.10$; $f_{s,\text{decl}}$ is retained as authoritative (`fs_reconciled = True`) and a diagnostic warning is logged.
   - `"inferred_from_begin_end"`: `Sampling Frequency` column is missing or invalid; $f_{s,\text{eff}} = f_{s,\text{impl}}$.
   - `"fallback_default"`: Neither declared nor implied frequency is available; defaults to $f_{\text{default}}$ (`250.0 Hz`) with explicit warning.

### 1.3 Time-Origin-Invariant Concatenation
In `pixel_counter/preprocessing.py` (`build_concatenated_signal`), row blocks are sorted by `Begin` time (if monotonic and non-overlapping) and concatenated into a single vector $\mathbf{x} \in \mathbb{R}^N$ with sample time vector:
$$t_k = t_0 + \frac{k}{f_s}, \quad k \in \{0, 1, \dots, N-1\}$$
where $t_0$ is the segment's initial `Begin` timestamp (defaulting to $0.0\text{ s}$ if absent). All downstream event timestamps (`time_s`, `t_onset_s`, `t_offset_s`) and reconstructed artifact intervals (`start_time_s`, `end_time_s`) are reported relative to $t_0 + k / f_s$, guaranteeing strict invariance under shifts of $t_0$ (`test_1_time_origin_invariance_all_analyzers`) and invariance under arbitrary partitioning of samples across `Transient` rows (`test_2_row_partition_invariance`).

---

## 2. PixelCorrelation Contraction Analyzer (`v2_signal.py`, `v2_detector.py`, `qc_v2.py`)

### 2.1 Physical Signal Characteristics
PixelCorrelation measures frame-to-frame optical correlation displacement of cardiomyocyte texture relative to a reference frame. Unlike sarcomere length or fluorescence ratio, the sign of the excursion depends on the arbitrary choice of reference frame and cellular motion axis:
- Upward-going contractions: polarity $p = +1$.
- Downward-going contractions: polarity $p = -1$.

### 2.2 Baseline Estimation and Cycle-Compression Scaling
1. **Dominant Cycle Estimation**: Autocorrelation of a bandpass-smoothed copy of $\mathbf{x}_{\text{raw}}$ over lags $\tau \in [0.22\text{ s}, 2.5\text{ s}]$ identifies the dominant inter-beat interval $\text{IBI}_{\text{est}}$ (`estimate_dominant_cycle_s`).
2. **Continuous Cycle-Compression Factor**: To prevent rolling quantile windows from bridging across closely spaced contractions during rapid pacing ($2\text{–}3\text{ Hz}$), the baseline window scale factor $\gamma_{\text{cycle}} \in [0.45, 1.0]$ varies continuously with $\text{IBI}_{\text{est}}$:
   $$\gamma_{\text{cycle}} = \text{clip}\!\left(\frac{\text{IBI}_{\text{est}}}{0.80\text{ s}},\, 0.45,\, 1.0\right)$$
   The effective rolling baseline window is $W_{\text{base}} = \max(0.55\text{ s},\, \gamma_{\text{cycle}} W_0)$.
3. **Directional Rest-Level Baseline**: For candidate polarity $p \in \{+1, -1\}$, the oriented signal $y_p(t) = p \cdot x_{\text{raw}}(t)$ is baseline-corrected using a rolling lower percentile ($q = 20\%$ under rapid cycling, $q = 25\%$ otherwise) smoothed with a Gaussian kernel ($\sigma = 0.25 W_{\text{base}}$):
   $$x_{\text{corr}, p}(t) = y_p(t) - B_p(t)$$

### 2.3 Antisymmetric Multi-Cue Polarity Selection
In `determine_raw_polarity`, candidate orientations $p = +1$ and $p = -1$ are evaluated symmetrically on five dimensionless physiological cues:
1. **Robust Skewness ($S_{\text{skew}}$)**: $(P_{95} - P_{50}) - (P_{50} - P_{5})$ normalized by $(P_{95} - P_{5})$.
2. **Upper-Tail Excursion Excess ($S_{\text{tail}}$)**: $(P_{98} - P_{50}) - (P_{50} - P_{2})$ normalized by MAD noise $\sigma_{\text{MAD}}$.
3. **Prominence-Weighted Candidate Support ($S_{\text{prom}}$)**: Sum of top-5 candidate peak prominences exceeding $2.8\sigma_{\text{MAD}}$ with physiologically valid width ($\text{FWHM} \ge 35\text{ ms}$).
4. **Waveform Shape Asymmetry ($S_{\text{shape}}$)**: Cardiomyocyte contractions exhibit faster systolic upstroke than diastolic relaxation ($t_{\text{rise}} < t_{\text{decay}}$) and longer diastolic rest plateaus than systolic peaks.
5. **Rapid-Cycle Rest-Plateau Asymmetry**: Under rapid pacing ($\text{IBI}_{\text{est}} \le 0.65\text{ s}$), when systolic and diastolic duty cycles approach $50\%$, orientation is disambiguated by comparing diastolic rest flatness (lower variance on the resting plateau than at peak turning points) and rise/fall derivative asymmetry ($|\min \dot{x}|$ vs. $|\max \dot{x}|$). Because every cue is strictly odd under $x \mapsto -x$, negating the input signal strictly negates the selected polarity (`test_10_polarity_inversion_antisymmetry`).

### 2.4 Event Detection and Binary Segment QC
- **Noise Scale**: Estimated via Median Absolute Deviation of first differences:
  $$\sigma_{\text{diff}} = \frac{\text{median}(|\Delta x_{\text{corr}}|)}{0.6745\sqrt{2}}$$
- **Candidate Selection & Template Scoring**: Peaks on $x_{\text{Smooth}}(t)$ are snapped to local maxima on $x_{\text{corr}}(t)$ within $\pm 35\text{ ms}$, scored by cosine similarity against a canonical bi-exponential contraction template, and filtered by preset-specific SNR, prominence, and width bounds (`permissive`, `default`, `conservative`).
- **Binary Segment QC (`evaluate_qc_v2`)**: Evaluates 8 deterministic failure modes (`step_discontinuity`, `baseline_drift`, `clipped_peaks`, `truncated_peaks`, `high_noise_floor`, `low_snr`, `incompatible_contraction_directions`, and `no_contractions_noise_dominated`). Clean quiescent recordings with low noise floor ($\sigma_{\text{diff}}$ within physiological rest limits and zero candidate peaks) receive `PASS` with $N_{\text{accepted}} = 0$, whereas noise-dominated peakless traces receive `REJECT`.

---

## 3. Calcium Transient Analyzer (`calcium_detector.py`)

### 3.1 Fura-2 Ratiometric Signal Model and Baseline Normalization
Intracellular $\text{Ca}^{2+}$ transients are recorded as Fura-2 excitation ratios $R(t) = F_{340}(t) / F_{380}(t)$ (or raw fluorescence intensity).
1. **Sensor Dropout Masking**: Non-finite samples and non-positive values ($R(t) \le 0$ or sudden downward dropouts $> 8\sigma_{\text{HF}}$ lasting $< 100\text{ ms}$) are masked prior to baseline and noise estimation.
2. **Diastolic Baseline $F_0(t)$**: Estimated via a rolling lower percentile ($P_{15}$ over a $2.0\text{ s}$ window) on the dropout-free signal, followed by Gaussian smoothing ($\sigma = 0.25\text{ s}$).
3. **Prominence-Protected Normalization**:
   $$x_{\text{norm}}(t) = \frac{R(t) - F_0(t)}{\max(F_0(t),\, \epsilon_{F_0})}, \quad \epsilon_{F_0} = 0.05$$
   This represents fractional change $\Delta R / R_0$ when $R_0 \ge 0.05$, while preventing division-by-zero blowup if baseline approaches zero.

### 3.2 Standard Mode vs. Adult MAIN/SCE Dual-Threshold Mode
- **Standard Mode (`calcium_mode="standard"`)**: Single-threshold peak detector parameterized by `CalciumDetectorParams` (`min_prominence_sigma`, `raw_prominence_sigma`, `min_norm_height`, `min_rise_vel_sigma`, `min_fwhm_s`, `max_fwhm_s`, `min_asym_ratio`).
- **Adult Mode (`calcium_mode="adult"`)**: Adult ventricular cardiomyocytes frequently exhibit both primary electrically stimulated $\text{Ca}^{2+}$ transients (`MAIN`) and spontaneous sarcoplasmic reticulum $\text{Ca}^{2+}$ release events / after-transients (`SECONDARY` / `SCE`).
  - **Reference Amplitude $A_{\text{ref,Ca}}$**: 75th percentile of clean candidate $\Delta R / R_0$ amplitudes with $\text{SNR} \ge 5.0$ and $\text{FWHM} \ge 40\text{ ms}$.
  - **Dual Thresholds**:
    $$T_{\text{SCE,Ca}} = \max\!\left(k_{\text{sce,Ca}} \sigma_{\text{norm}},\, h_{\min,\text{sce}}\right)$$
    $$T_{\text{MAIN,Ca}} = \max\!\left(\alpha_{\text{ref,Ca}} A_{\text{ref,Ca}},\, k_{\text{main,Ca}} \sigma_{\text{norm}},\, h_{\min,\text{main}},\, T_{\text{SCE,Ca}}\right)$$
  - **Strict Hierarchy**: Any valid calcium transient with $A_{\text{peak}} \ge T_{\text{MAIN,Ca}}$ is classified as `MAIN`; any valid transient with $T_{\text{SCE,Ca}} \le A_{\text{peak}} < T_{\text{MAIN,Ca}}$ is classified as `SECONDARY`; candidates below $T_{\text{SCE,Ca}}$ are rejected.

---

## 4. Sarcomere Shortening Analyzer (`sarcomere_qc.py`, `sarcomere_artifact_recovery.py`, `sarcomere_detector.py`, `sarcomere_pipeline.py`)

### 4.1 Physical Measurement Model and Artifact Detection (`SarcomereMeasurementProfile`)
IonOptix sarcomere length $L(t)$ (in $\mu\text{m}$) is computed in hardware/acquisition software via real-time Fast Fourier Transform (FFT) of striation periodicity within a region of interest. Because the optical FFT power spectrum is searched within a bounded spatial-frequency bandpass window:
- When striation contrast drops or motion shifts the cell out of focus, the FFT peak locks onto the lower or upper boundary of the search window ("rail clamping") or jumps discontinuously between harmonic peaks.
- `SarcomereMeasurementProfile` encapsulates these physical and optical bounds:
  - `abs_min_um = 1.0`, `abs_max_um = 3.0`: Absolute biological bounds on mammalian sarcomere length.
  - `fft_rail_min_um = 1.28`, `fft_rail_max_um = 2.18`: Default IonOptix FFT search-window rail thresholds (configurable for alternative optics or species).
  - `jump_vel_um_per_s = 8.75`, `extreme_vel_um_per_s = 13.75`: Maximum physiological sarcomere shortening/relengthening velocity bounds ($\mu\text{m/s}$), converted at runtime to per-sample step thresholds via $\Delta t = 1 / f_s$.

### 4.2 Conservative Local PCHIP Artifact Reconstruction
In `reconstruct_narrow_tracking_artifacts` (`pixel_counter/sarcomere_artifact_recovery.py`):
1. Contiguous artifact sample clusters $[c_0, c_1]$ are identified from the union of biological out-of-range samples, FFT rail-clamped samples, and velocity-jump spikes.
2. **Eligibility Gate**: Reconstruction is attempted only if total artifact coverage is $\le 12\%$ of the trace, cluster count $\le 40$, and the individual gap duration is $\le \tau_{\max}$ (`max_gap_s = 0.080 s`, i.e., $80\text{ ms}$).
3. **Local Hermite Interpolation**: Up to 4 valid anchor samples immediately preceding $c_0$ and following $c_1$ define a local Piecewise Cubic Hermite Interpolating Polynomial (`scipy.interpolate.PchipInterpolator`). Because PCHIP preserves monotonicity between anchor points, it cannot invent an interior extremum beyond the anchor values unless the anchors themselves bracket a boundary. Safety bounds reject any interpolant that excursions $> 0.20\ \mu\text{m}$ below the minimum anchor or $> 0.05\ \mu\text{m}$ above the maximum anchor.
4. **Reconciliation & Honest Amplitude Accounting (`reconcile_reconstructed_events`)**:
   - Candidate detection is run on both the raw trace $\mathbf{L}_{\text{raw}}$ and the reconstructed trace $\mathbf{L}_{\text{recon}}$.
   - An event detected on $\mathbf{L}_{\text{recon}}$ whose trough nadir was corrupted on $\mathbf{L}_{\text{raw}}$ is accepted **only** if supported by genuine non-artifact shortening shoulders on both sides of the gap (`left_shoulder_drop >= 0.012 um` and `right_shoulder_drop >= 0.012 um`).
   - Crucially, if the trough nadir was corrupted by a tracking artifact (`peak_corrupted == True`), **no interpolated amplitude is ever reported as a measured contraction amplitude**: `l_min_um`, `abs_shortening_um`, and `pct_shortening` are set to `NaN`, `amplitude_source` is marked `"TIMING_ONLY_UNMEASURED_PEAK"`, and the segment summary increments `n_lacking_amplitude` while excluding the event from `n_directly_measured` and `n_valid_amplitude` (`test_7_artifact_injection_and_direct_amplitude_accounting`).

### 4.3 Two-Pass Diastolic Baseline $L_0(t)$ and Dual-Threshold Hierarchy
1. **Two-Pass Rolling Diastolic Percentile $L_0(t)$**:
   - Pass 1 estimates a provisional upper envelope ($P_{82}$ over a $1.25\text{ s}$ rolling window on clean samples).
   - Samples depressed by $> 2.5\sigma_{\text{HF}}$ below the Pass-1 envelope (active systolic troughs) are masked out, and Pass 2 re-estimates the rolling $P_{82}$ envelope on the remaining diastolic samples, smoothed with a Gaussian kernel ($\sigma = 0.18\text{ s}$).
2. **Shortening Metrics**: For each downward trough at index $k_{\min}$ with local minimum length $L_{\min} = L(k_{\min})$ and local diastolic baseline $L_0 = L_0(k_{\min})$:
   $$\Delta L = L_0 - L_{\min}\ (\mu\text{m}), \qquad \%\,\text{Shortening} = 100 \times \frac{L_0 - L_{\min}}{L_0}\ (\%)$$
3. **Dual Scientific Amplitude Thresholds**:
   Let $\sigma_{\%}= 100 \cdot \sigma_{\text{HF}} / \max(\text{median}(L_0),\, 0.5)$ and let $A_{\text{ref},\%}$ be the 75th percentile of $\%$-shortening among uncorrupted candidates with $\text{SNR} \ge 5.0$, $\text{FWHM} \ge \text{FWHM}_{\min}$, and prominence $\ge 0.45 \Delta L$. Then:
   $$T_{\text{SCE}} = \max\!\left(k_{\text{sce}} \sigma_{\%},\, \%_{\min,\text{sce}},\, \frac{100 \cdot \Delta L_{\min,\text{sce}}}{\text{median}(L_0)}\right)$$
   $$T_{\text{MAIN}} = \max\!\left(\alpha_{\text{ref}} A_{\text{ref},\%},\, k_{\text{main}} \sigma_{\%},\, \%_{\min,\text{main}},\, \frac{100 \cdot \Delta L_{\min,\text{main}}}{\text{median}(L_0)},\, T_{\text{SCE}}\right)$$
   This guarantees the strict invariant $T_{\text{MAIN}} \ge T_{\text{SCE}}$ across all recordings and presets. Every valid contraction with $\%\,\text{Shortening} \ge T_{\text{MAIN}}$ is classified as `MAIN`; every valid contraction with $T_{\text{SCE}} \le \%\,\text{Shortening} < T_{\text{MAIN}}$ is classified as `SECONDARY`.

### 4.4 Strict Sarcomere–Calcium Channel Independence
In `sarcomere-calcium` mode (`pixel_counter/sarcomere_pipeline.py`), the sarcomere shortening pipeline and calcium transient pipeline execute completely independently on their respective channels. Post-hoc temporal matching (`match_sarcomere_and_calcium_events`) associates accepted sarcomere events with accepted calcium transients within a physiological electromechanical coupling window ($[-60\text{ ms}, +180\text{ ms}]$) for reporting and visualization only. The presence, absence, scaling, or corruption of the calcium trace has zero mathematical effect on sarcomere event detection, classification, amplitudes, or sarcomere QC (`test_11_calcium_channel_independence`).

---

## 5. Statistical Summaries, Frequency Denominators, and Limitations

### 5.1 Frequency and Inter-Beat Interval (IBI) Denominators
- **Event-Span Contraction Frequency (`contraction_frequency_hz`)**: Computed only when $N_{\text{accepted}} \ge 2$ from the mean inter-beat interval $\overline{\text{IBI}} = \frac{1}{N_{\text{accepted}} - 1} \sum_{j=1}^{N_{\text{accepted}}-1} (t_{j+1} - t_j)$:
  $$f_{\text{span}} = \frac{1}{\overline{\text{IBI}}}\ (\text{Hz}), \qquad \text{CV}_{\text{IBI}} = \frac{\text{SD}(\text{IBI})}{\overline{\text{IBI}}}$$
- **Segment-Duration Rate**: $N_{\text{accepted}} / T_{\text{segment}}$, where $T_{\text{segment}} = N / f_s$. Explicitly distinguished from $f_{\text{span}}$ when beats occur in bursts or only part of a segment is active.

### 5.2 Known Physiological and Instrumental Limitations
1. **Unmeasured Peak Amplitudes During Tracking Dropout**: When optical FFT tracking drops out across the systolic nadir, true minimum sarcomere length $L_{\min}$ is physically unobservable. The software refuses to fabricate $L_{\min}$ from spline interpolation (`NaN` amplitude, `TIMING_ONLY_UNMEASURED_PEAK`), which preserves amplitude integrity at the cost of reduced sample size $N_{\text{valid\_amplitude}} < N_{\text{accepted}}$.
2. **Optical FFT Rail Bounds**: Default FFT rail thresholds (`1.28–2.18 um`) reflect standard IonOptix adult cardiomyocyte ROI settings. Recordings acquired with modified optical magnification or non-standard ROI grating bounds should supply an explicit `SarcomereMeasurementProfile`.
3. **Rapid Pacing Duty-Cycle Symmetry**: Above $3.5\text{ Hz}$ in PixelCorrelation recordings, systolic and diastolic durations converge toward $50\%$ duty cycle; if upstroke and relaxation velocities are also nearly symmetric, automatic polarity confidence decreases and manual polarity review in the HTML utility is recommended.
