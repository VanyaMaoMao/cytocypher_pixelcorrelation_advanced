import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pixel_counter.io_utils import check_output_conflicts, load_cytocypher_excel

@pytest.mark.xfail(strict=True, reason="Missing or non-numeric sampling frequency silently assumes 250Hz. Expected ValueError.")
def test_missing_sampling_frequency_validation(tmp_path):
    """
    Test that invalid or missing sampling frequency raises a clear error rather than
    defaulting silently to 250 Hz.
    """
    df = pd.DataFrame({
        "Begin (seconds)": [0.0],
        "Sampling Frequency": ["oops"], # Invalid text
        "y 0": [1.0]
    })

    test_file = tmp_path / "test.xlsx"
    df.to_excel(test_file, sheet_name="PixelCorrelation Segment 1", index=False)

    # We expect a ValueError or explicit rejection.
    # Currently it defaults to 250.0.
    _, _, fs, _ = load_cytocypher_excel(str(test_file), "PixelCorrelation Segment 1")
    # The assert below will fail because it silently returns 250.0
    assert fs != 250.0, "Expected strict validation, but silently defaulted to 250Hz"


@pytest.mark.xfail(strict=True, reason="Output conflict check permits multiple identical output destinations.")
def test_output_path_collision():
    """
    Test that supplying identical paths to multiple output handlers triggers a conflict error
    rather than allowing race conditions or overwrite collisions.
    """
    input_path = Path("input.xlsx")
    outputs = [
        Path("output.xlsx"),
        Path("output.xlsx") # Duplicate!
    ]

    # check_output_conflicts currently only compares outputs vs input.
    # It does not check if outputs collide with each other.
    # We expect this to raise a ValueError.
    try:
        check_output_conflicts(input_path, outputs)
        raised = False
    except ValueError:
        raised = True

    assert raised, "Expected a ValueError for duplicate output paths, but none was raised."

def test_conflicting_sampling_frequencies(tmp_path):
    """
    Test that conflicting sampling frequencies across segments trigger validation errors.
    This was already properly handled in the implementation (it raises ValueError).
    So we don't xfail this, we just assert that it works.
    """
    df = pd.DataFrame({
        "Begin (seconds)": [0.0, 1.0],
        "Sampling Frequency": [250.0, 500.0],
        "y 0": [1.0, 2.0]
    })

    test_file = tmp_path / "test_conflict.xlsx"
    df.to_excel(test_file, sheet_name="PixelCorrelation Segment 1", index=False)

    with pytest.raises(ValueError, match="Conflicting Sampling Frequency values found"):
        load_cytocypher_excel(str(test_file), "PixelCorrelation Segment 1")

@pytest.mark.xfail(strict=True, reason="Missing y offsets are not validated.")
def test_missing_y_offsets(tmp_path):
    """
    Test that missing y-offsets trigger validation errors.
    """
    df = pd.DataFrame({
        "Begin (seconds)": [0.0],
        "Sampling Frequency": [250.0],
        "y 0": [1.0],
        "y 2": [2.0] # y 1 is missing
    })

    test_file = tmp_path / "test_missing_y.xlsx"
    df.to_excel(test_file, sheet_name="PixelCorrelation Segment 1", index=False)

    try:
        # Load and check if there is an error for missing y1
        df_loaded, y_cols, _, _ = load_cytocypher_excel(str(test_file), "PixelCorrelation Segment 1")
        raised = False
        # If it doesn't raise, we assert fail because it allows skipping columns silently
        assert raised, "Expected ValueError due to missing offset positions, but passed silently"
    except ValueError:
        pass

def test_output_path_collision_input():
    """
    Test that supplying an output path matching the input path triggers an error.
    This is already handled by check_output_conflicts, so we assert it works as intended without xfail.
    """
    input_path = Path("input.xlsx")
    outputs = [Path("output.xlsx"), input_path]
    with pytest.raises(ValueError, match="Output path conflicts with input path"):
        check_output_conflicts(input_path, outputs)
