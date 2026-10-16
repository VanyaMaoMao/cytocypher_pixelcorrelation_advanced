"""Unit tests for polarity overrides management and fallback mechanisms."""
import json
import tempfile
from pathlib import Path

import pytest

from pixel_counter.polarity_overrides import (
    compute_file_sha256,
    default_overrides_path,
    load_polarity_overrides,
    parse_polarity_override_str,
    save_polarity_overrides,
)
import run_v2_review_utility as rv


def test_compute_file_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello cytocypher v2")
    # Expected SHA256 of "hello cytocypher v2"
    import hashlib
    expected = hashlib.sha256(b"hello cytocypher v2").hexdigest()
    assert compute_file_sha256(f) == expected


def test_default_overrides_path():
    wb = Path("/data/experiment1.xlsx")
    assert default_overrides_path(wb) == Path("/data/experiment1.polarity_overrides.json")


def test_parse_polarity_override_str():
    # Empty inputs
    assert parse_polarity_override_str("") == {}
    assert parse_polarity_override_str("   ") == {}

    # Basic formats
    res = parse_polarity_override_str("17=-1,18=-1,45=1")
    assert res == {17: -1, 18: -1, 45: 1}

    # Colon separators, varied casing, textual keywords
    res2 = parse_polarity_override_str("17:-1, 45:positive, 30:down, 16:downward, 35:up, 36:upward")
    assert res2 == {17: -1, 45: 1, 30: -1, 16: -1, 35: 1, 36: 1}

    # Reset / auto keywords
    res3 = parse_polarity_override_str("17:0, 18:auto, 19:none")
    assert res3 == {17: 0, 18: 0, 19: 0}

    # Invalid tokens
    with pytest.raises(ValueError, match="Expected 'segment=polarity'"):
        parse_polarity_override_str("invalid_token")

    with pytest.raises(ValueError, match="Invalid segment number"):
        parse_polarity_override_str("segmentA=1")

    with pytest.raises(ValueError, match="Invalid polarity value"):
        parse_polarity_override_str("17=unknown_polarity")


def test_save_and_load_overrides_roundtrip(tmp_path):
    wb = tmp_path / "mock_experiment.xlsx"
    wb.write_bytes(b"mock excel content for sha256 testing")
    ov_file = tmp_path / "mock_experiment.polarity_overrides.json"

    overrides_to_save = {17: -1, 18: -1, 45: 1, 99: 0}  # 0 should be pruned from saved active overrides
    saved_path = save_polarity_overrides(wb, overrides_to_save, ov_file)
    assert saved_path == ov_file
    assert ov_file.exists()

    with open(ov_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["workbook_name"] == "mock_experiment.xlsx"
    assert data["workbook_sha256"] == compute_file_sha256(wb)
    assert data["overrides"] == {"17": -1, "18": -1, "45": 1}

    # Load with strict hash
    loaded = load_polarity_overrides(wb, ov_file, strict_hash=True)
    assert loaded == {17: -1, 18: -1, 45: 1}


def test_load_overrides_mismatched_workbook(tmp_path):
    wb1 = tmp_path / "workbook_A.xlsx"
    wb1.write_bytes(b"data A")
    wb2 = tmp_path / "workbook_B.xlsx"
    wb2.write_bytes(b"data B")

    ov_file = save_polarity_overrides(wb1, {17: -1})

    # Attempting to apply wb1's overrides to wb2 must raise ValueError
    with pytest.raises(ValueError, match="Mismatched workbook"):
        load_polarity_overrides(wb2, ov_file)


def test_load_overrides_strict_sha256_mismatch(tmp_path):
    wb = tmp_path / "workbook_A.xlsx"
    wb.write_bytes(b"original data")
    ov_file = save_polarity_overrides(wb, {17: -1})

    # Modify workbook data
    wb.write_bytes(b"modified data")

    # Strict hash should raise ValueError
    with pytest.raises(ValueError, match="Mismatched workbook SHA256"):
        load_polarity_overrides(wb, ov_file, strict_hash=True)

    # Non-strict hash allows loading if name matches
    loaded = load_polarity_overrides(wb, ov_file, strict_hash=False)
    assert loaded == {17: -1}


def test_load_overrides_file_not_found(tmp_path):
    wb = tmp_path / "workbook.xlsx"
    wb.write_bytes(b"data")
    non_existent = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_polarity_overrides(wb, non_existent)


def test_review_utility_cli_overrides(capsys, monkeypatch):
    # Test that --override-polarity and --save-overrides work via main()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wb = tdp / "dummy_test.xlsx"
        wb.write_bytes(b"dummy")
        ov_out = tdp / "out_overrides.json"

        # Mock process_workbook to prevent actual Excel parsing
        monkeypatch.setattr(rv, "process_workbook", lambda *args, **kwargs: ([], "dummy_test.xlsx"))

        # Run inspect polarity with override saving
        rv.main([
            str(wb),
            "--override-polarity", "17=-1,45=1",
            "--save-overrides", str(ov_out),
            "--inspect-polarity",
        ])

        assert ov_out.exists()
        loaded = load_polarity_overrides(wb, ov_out)
        assert loaded == {17: -1, 45: 1}
