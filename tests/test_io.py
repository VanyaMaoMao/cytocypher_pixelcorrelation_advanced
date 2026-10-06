import pytest
import os
import pandas as pd
from pathlib import Path
from pixel_counter.io_utils import atomic_file_path, build_output_path, check_output_conflicts

def test_atomic_file_path(tmp_path):
    target_file = tmp_path / "test_atomic.txt"
    with atomic_file_path(target_file) as temp_file:
        temp_file.write_text("hello")
    assert target_file.exists()
    assert target_file.read_text() == "hello"

def test_atomic_file_path_failure(tmp_path):
    target_file = tmp_path / "test_atomic_fail.txt"
    try:
        with atomic_file_path(target_file) as temp_file:
            temp_file.write_text("failed write")
            raise ValueError("simulated error")
    except ValueError:
        pass
    assert not target_file.exists()
    assert not temp_file.exists()

def test_build_output_path():
    assert build_output_path("test.xlsx", "_suffix.txt") == "test_suffix.txt"
    assert build_output_path("TEST.XLSX", "_suffix.txt") == "TEST_suffix.txt"
    assert build_output_path("test.data.xlsx", "_suffix.txt") == "test.data_suffix.txt"
    assert Path(build_output_path("my folder/test.xlsx", "_suffix.txt")) == Path("my folder/test_suffix.txt")
    assert build_output_path("test.txt", "_suffix") == "test.txt_suffix"

def test_check_output_conflicts(tmp_path):
    input_file = tmp_path / "input.xlsx"
    input_file.touch()

    # Exact match
    with pytest.raises(ValueError):
        check_output_conflicts(input_file, [input_file])

    # Case insensitive match
    with pytest.raises(ValueError):
        check_output_conflicts(input_file, [str(input_file).upper()])

    # Samefile match (using a symlink if available, otherwise skip)
    if hasattr(os, "symlink"):
        symlink_file = tmp_path / "input_link.xlsx"
        try:
            os.symlink(input_file, symlink_file)
            with pytest.raises(ValueError):
                check_output_conflicts(input_file, [symlink_file])
        except OSError:
            pass

    # No conflict
    other_file = tmp_path / "other.xlsx"
    check_output_conflicts(input_file, [other_file]) # Should not raise
