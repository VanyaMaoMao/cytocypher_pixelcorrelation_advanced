# Cytocypher Real Data Annotation Instructions

Scientific validation requires comparison of the algorithm against independent ground truth. Since the existing inputs are unannotated, this document provides the necessary format and steps for lab personnel to label contractions.

## Annotation Format

Create a new Excel or CSV file (e.g., `ground_truth_annotations.xlsx`) with the following strict columns:

1. **`source_workbook`**: Name of the original Excel recording file.
2. **`sheet_name`**: Name of the Excel sheet.
3. **`sample_id`**: The experimental Sample ID (e.g., Cell A).
4. **`time_s`**: The time of the true contraction peak in absolute seconds.
5. **`confidence`**: `1` for unambiguous/confident, `0` for ambiguous/questionable.
6. **`annotator_id`**: Initials of the person labeling.
7. **`invalid_interval_start_s`** (optional): If this row describes a missing/invalid data interval rather than a point.
8. **`invalid_interval_end_s`** (optional): End of the missing/invalid data interval.

## Rules
- **No AFCs:** Establish an operational definition of a genuine contraction event without classifying aftercontractions (AFC). Do not annotate AFCs.
- **Do not use existing outputs:** Do not copy the output of the legacy algorithm and call it ground truth. Manually inspect the raw signal (or use synchronous video if available).
- **Development / Holdout Split:** Label a wide range of files. Mark some as "development" (for us to verify thresholds) and some as "holdout" (which we will run ONCE as a final test).

## Next Steps
Once annotations are gathered:
1. Save the file.
2. We will run the validation infrastructure (now built in `pixel_counter.validation`) against these real files.
3. Step 20 of the migration will only proceed after these tests pass the required criteria.
