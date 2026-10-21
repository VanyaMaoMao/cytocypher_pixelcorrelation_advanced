# Cytocypher Scientific Validation Report

## ⚠️ WARNING: SYNTHETIC DATA FALLBACK ⚠️

This validation report was generated using **deterministic synthetic ground truth** because independently annotated real recordings are unavailable.

**Status: BLOCKED**

Scientific validation of the v2 detector is BLOCKED. The detector is NOT validated or ready to replace the legacy pipeline. Step 20 cannot proceed until real annotations are collected and evaluated against the predefined acceptance criteria.

## Evaluation Parameters
- **Tolerance (s):** 0.05
- **V2 Config:** `auto polarity, default strictness (alpha=0.5, rho=0.5, snr=8)`

## Overall Results Summary

| Metric | Legacy Pipeline | V2 Pipeline |
|---|---|---|
| f1 | 0.2500 | 1.0000 |
| precision | 0.2045 | 1.0000 |
| recall | 0.3214 | 1.0000 |
| abs_count_error | 18 | 0 |
| tp | 9 | 42 |
| fp | 35 | 0 |
| fn | 19 | 0 |

## Per-Recording Results

### Recording: synthetic_1hz_clean
- True Event Count: 9

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.5143 | 0.3462 | 1.0000 | 9 | 17 | 0 | 17 | 1.7000 | N/A | ERROR/REJECT |
| v2 | 1.0000 | 1.0000 | 1.0000 | 9 | 0 | 0 | 0 | 0.0000 | True | PASS |

### Recording: synthetic_2hz_inverted
- True Event Count: 19

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.0000 | 0.0000 | 0.0000 | 0 | 18 | 19 | -1 | -0.1000 | N/A | ERROR/REJECT |
| v2 | 1.0000 | 1.0000 | 1.0000 | 19 | 0 | 0 | 0 | 0.0000 | True | PASS |

### Recording: synthetic_gap_1hz
- True Event Count: 4

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_EVALUABLE |
| v2 | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 | 0 | 0.0000 | True | PASS |

### Recording: synthetic_irregular
- True Event Count: 10

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_EVALUABLE |
| v2 | 1.0000 | 1.0000 | 1.0000 | 10 | 0 | 0 | 0 | 0.0000 | True | PASS |

