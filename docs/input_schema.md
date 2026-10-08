# Input Excel Contract

This document defines the strict schema and validation rules for input Excel workbooks processed by the `pixel_counter` package.

## Required Columns
The processor looks for specific columns based on their names. Names are typically parsed case-sensitively or explicitly lower-cased in some helper functions, but the parser strictly requires:

1. **`y <offset>` columns**: At least one column starting with the exact string `"y "`. The rest of the column name must be a parsable numeric value (the y-offset in µm). For example: `"y -1.0"`, `"y 0"`, `"y 1"`.
   - **Order and Sorting**: Columns are not assumed to be in numeric order in the Excel file. The parser validates that all `y ` columns have valid numeric offsets and sorts them from lowest to highest.
   - **Uniqueness**: Duplicate numeric offsets (e.g. `"y 1"` and `"y 1.0"`) are not allowed and will cause an error.

2. **`Sampling Frequency`** (optional but highly recommended):
   - Contains the sampling rate in Hz (e.g., `250.0`).
   - If present, the parser checks all non-empty values in the column. All values must be identical (no conflicting sampling frequencies allowed within a segment).
   - The value must be > 0, finite, and not NaN.
   - If missing, it defaults to a fallback value of `250.0` (as a documented metadata source convention).

3. **`Begin (seconds)`** (optional):
   - Contains the starting time of the recording block.
   - Used for the absolute time alignment.
   - Defaults to `0.0` if not provided.

4. **`Transientnumber`** (optional):
   - Integer identifier for transient window splits.
   - Parsed for consecutive numbers. If not present, an auto-incrementing index is used per row.

5. **`Sample ID`** (optional):
   - Identifies the biological or experimental sample.
   - Must be consistent within a single sheet. If multiple conflicting Sample IDs are found in the same sheet, an error is raised.

## Structure & Timing Overlaps
- Each row represents an individual observation or window (often a transient window).
- `Begin (seconds)` establishes the start of the row's temporal data.
- If rows overlap in their temporal spans (calculated via lengths of valid signal / `Sampling Frequency`), they are later stitched into a single continuous timeline.
- The temporal alignment relies strictly on the numeric constraints and validity of the time indices, preventing arbitrary scaling or crossfading without explicit logic.

## Rejections
The parser will explicitly throw an error (rather than silently fallback) if:
- No `"y "` columns are found.
- A `"y "` column cannot be parsed as a float.
- There are duplicate `"y "` offsets.
- There are conflicting `Sampling Frequency` values.
- `Sampling Frequency` is `<= 0`, infinite, or negative.
- There are conflicting `Sample ID` values on the same sheet.
