# Cytocypher Scientific Validation Report

## ⚠️ WARNING: SYNTHETIC DATA FALLBACK ⚠️

This validation report was generated using **deterministic synthetic ground truth** because independently annotated real recordings are unavailable.

**Status: BLOCKED**

Scientific validation of the v2 detector is BLOCKED. The detector is NOT validated or ready to replace the legacy pipeline. Step 20 cannot proceed until real annotations are collected and evaluated against the predefined acceptance criteria.

## Evaluation Parameters
- **Tolerance (s):** 0.05
- **V2 Config:** `auto, no_two_pass`

## Overall Results Summary

| Metric | Legacy Pipeline | V2 Pipeline |
|---|---|---|
| f1 | 0.0000 | 0.9767 |
| precision | 0.0000 | 0.9545 |
| recall | 0.0000 | 1.0000 |
| abs_count_error | 42 | 2 |
| tp | 0 | 42 |
| fp | 0 | 2 |
| fn | 42 | 0 |

## Per-Recording Results

### Recording: synthetic_1hz_clean
- True Event Count: 9

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 9 | -9 | nan | N/A | ERROR/REJECT |
| v2 | 1.0000 | 1.0000 | 1.0000 | 9 | 0 | 0 | 0 | 0.0000 | True | PASS |

### Recording: synthetic_2hz_inverted
- True Event Count: 19

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 19 | -19 | nan | N/A | ERROR/REJECT |
| v2 | 1.0000 | 1.0000 | 1.0000 | 19 | 0 | 0 | 0 | 0.0000 | True | REVIEW |

### Recording: synthetic_gap_1hz
- True Event Count: 4

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 4 | -4 | nan | N/A | ERROR/REJECT |
| v2 | 1.0000 | 1.0000 | 1.0000 | 4 | 0 | 0 | 0 | 0.0000 | True | PASS |

### Recording: synthetic_irregular
- True Event Count: 10

| Version | F1 | Precision | Recall | TP | FP | FN | Count Error | Hz Error | Polarity Correct | QC |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 10 | -10 | nan | N/A | ERROR/REJECT |
| v2 | 0.9091 | 0.8333 | 1.0000 | 10 | 2 | 0 | 2 | 0.1000 | True | PASS |
