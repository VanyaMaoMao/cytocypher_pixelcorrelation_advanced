import io
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import run_v2_review_utility as rv
from pixel_counter.v2_detector import DetectorParams, STRICTNESS_PRESETS
from tests.v2_synth import FS, make_amplitude, to_raw


def _fake_process(raw, t0_stitch, t0_loader):
    meta = {"time_origin_s": t0_stitch, "stitch_overlap_conflicts": 0}
    with patch("pathlib.Path.exists", return_value=True):
        with patch("run_v2_review_utility.pd.ExcelFile") as xl:
            xl.return_value.sheet_names = ["PixelCorrelation Segment 1"]
            with patch("run_v2_review_utility.load_cytocypher_excel", return_value=(pd.DataFrame(), ["y 0"], FS, t0_loader)):
                with patch("run_v2_review_utility.extract_sample_id_from_segment_sheet", return_value="Cell_A"):
                    with patch("run_v2_review_utility.build_concatenated_signal", return_value=(raw, [], meta)) as bcs:
                        out = rv.process_workbook("dummy.xlsx")
                        return out, bcs


def test_review_uses_raw_signal_and_stitched_time_origin():
    t, a = make_amplitude([0.8, 1.8, 2.8, 3.8, 4.8])
    raw = to_raw(a, -1)
    (segments, source), bcs = _fake_process(raw, t0_stitch=44.2998, t0_loader=43.9598)
    # the signal is requested WITHOUT legacy inversion
    assert bcs.call_args.kwargs.get("force_invert") is False
    seg = segments[0]
    assert seg["t0"] == pytest.approx(44.2998)                 # stitched origin, not the segment header
    assert seg["raw_polarity"] == -1
    run = seg["runs"]["default"]
    acc = [i for i, s in enumerate(run["events"]["status"]) if s == "ACCEPTED"]
    assert len(acc) == 5
    for i in acc:
        idx = run["events"]["index"][i]
        assert run["events"]["time_s"][i] == pytest.approx(44.2998 + idx / FS)
    # all strictness presets are real detector runs
    assert set(seg["runs"]) == set(STRICTNESS_PRESETS)
    assert seg["runs"]["strict"]["params"]["min_rel_amplitude"] == STRICTNESS_PRESETS["strict"].min_rel_amplitude


def test_review_embeds_consistent_amplitude_and_raw():
    t, a = make_amplitude([0.8, 1.8, 2.8, 3.8, 4.8])
    raw = to_raw(a, 1)
    (segments, _), _ = _fake_process(raw, 10.0, 10.0)
    seg = segments[0]
    dec = lambda b: np.frombuffer(__import__("base64").b64decode(b), dtype=np.float32)
    amp, base, rawd = dec(seg["amplitude_b64"]), dec(seg["baseline_raw_b64"]), dec(seg["sig_raw_b64"])
    np.testing.assert_allclose(amp, seg["raw_polarity"] * (rawd - base), atol=1e-5)
    ev = seg["runs"]["default"]["events"]
    for i, s in enumerate(ev["status"]):
        if s == "ACCEPTED":
            assert ev["amplitude"][i] == pytest.approx(float(amp[ev["index"][i]]), abs=1e-4)
            assert ev["raw_value"][i] == pytest.approx(float(rawd[ev["index"][i]]), abs=1e-4)


def test_review_failure_segment_is_reported_without_events():
    rng = np.random.default_rng(1)
    raw = 0.5 + rng.normal(0, 1e-3, 1500)
    (segments, _), _ = _fake_process(raw, 0.0, 0.0)
    seg = segments[0]
    assert seg["qc_status"] == "REJECT" and seg["has_amplitude"] is False
    assert seg["runs"]["default"]["events"]["index"] == []


@patch("urllib.request.urlopen")
def test_review_html_offline_and_layout(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value = io.BytesIO(b"console.log('offline plotly');")
    t, a = make_amplitude([0.8, 1.8, 2.8])
    (segments, name), _ = _fake_process(to_raw(a, 1), 5.0, 5.0)
    written = {}
    out = "review_test_out.html"
    with patch("builtins.open", create=True) as m:
        m.return_value.__enter__.return_value.write.side_effect = lambda s: written.setdefault("html", s)
        rv.generate_html(segments, name, out)
    html = written["html"]
    assert "console.log('offline plotly');" in html and "https://cdn.plot.ly" not in html
    # no auxiliary overlay axes any more, three real strictness runs, params displayed
    assert "overlaying" not in html
    assert "min_rel_amplitude" in html and "min_rel_recovery" in html and "Show rejected candidates" in html
    assert "REVIEW" not in html.replace("review", "")          # no uncertain class in the review UI
    assert r'\n"' in html or '\\n' in html                       # CSV newline escapes survive as JS escapes
    data = json.loads(re.search(r"const segments_data = (.*?);\n", html, re.S).group(1))
    assert data[0]["runs"]["default"]["params"]["min_rel_amplitude"] == 0.5


def test_cli_custom_run():
    args = type("A", (), {"min_rel_amplitude": 0.4, "min_rel_recovery": None, "min_snr": None})()
    runs = rv.build_params_by_name(args)
    assert next(iter(runs)) == "custom" and runs["custom"] == DetectorParams(min_rel_amplitude=0.4)
    args = type("A", (), {"min_rel_amplitude": None, "min_rel_recovery": None, "min_snr": None})()
    assert "custom" not in rv.build_params_by_name(args)


def test_arbitrary_filename_and_windows_path_accepted():
    # Demonstrates that arbitrary filenames, including those containing '240926' and
    # absolute Windows paths, are accepted without artificial rejection.
    t, a = make_amplitude([0.8, 1.8, 2.8])
    raw = to_raw(a, 1)

    filenames_to_test = [
        "240926_IMR90_1,2,3HZ individual transients result.xlsx",
        r"C:\Users\talif\Desktop\240926_IMR90_1,2,3HZ individual transients result.xlsx",
        "custom_experiment_2026_test.xlsx",
    ]

    for fname in filenames_to_test:
        meta = {"time_origin_s": 0.0, "stitch_overlap_conflicts": 0}
        with patch("pathlib.Path.exists", return_value=True):
            with patch("run_v2_review_utility.pd.ExcelFile") as xl:
                xl.return_value.sheet_names = ["PixelCorrelation Segment 1"]
                with patch("run_v2_review_utility.load_cytocypher_excel", return_value=(pd.DataFrame(), ["y 0"], FS, 0.0)):
                    with patch("run_v2_review_utility.extract_sample_id_from_segment_sheet", return_value="Cell_A"):
                        with patch("run_v2_review_utility.build_concatenated_signal", return_value=(raw, [], meta)):
                            segments, name = rv.process_workbook(fname)
                            assert len(segments) == 1
                            assert segments[0]["segment_number"] == 1


def test_input_validation_informative_errors(tmp_path):
    # Non-existent file raises FileNotFoundError
    non_existent = tmp_path / "does_not_exist.xlsx"
    with pytest.raises(FileNotFoundError, match="Workbook not found"):
        rv.process_workbook(str(non_existent))

    # Excel file with no PixelCorrelation segment sheets raises ValueError
    empty_wb = tmp_path / "no_corr_sheets.xlsx"
    empty_wb.write_bytes(b"empty")
    with patch("pathlib.Path.exists", return_value=True):
        with patch("run_v2_review_utility.pd.ExcelFile") as xl:
            xl.return_value.sheet_names = ["Sheet1", "OtherData"]
            with pytest.raises(ValueError, match="No PixelCorrelation segment sheets found"):
                rv.process_workbook(str(empty_wb))


def test_navigation_buttons_and_js_present():
    """Verify Previous and Next buttons, styles, and event handling in HTML template."""
    html = rv.HTML_TEMPLATE
    assert 'id="btn_prev"' in html
    assert 'id="btn_next"' in html
    assert 'updateNavButtons()' in html
    assert 'btnPrev.disabled = (currentSegIdx <= 0)' in html
    assert 'btnNext.disabled = (currentSegIdx >= segments_data.length - 1)' in html
    assert 'ArrowLeft' in html and 'ArrowRight' in html
    assert 'getActiveQC' in html
    assert 'POLARITY INPUT REQUIRED' in html
    assert 'qc_status,qc_reason' in html


def test_qc_reconciliation_segments_17_and_20():
    """Verify that Segments 17 and 20 resolve to -1 with 9 events, and undetermined Segment 25 requires input."""
    wb_path = r"C:\Users\talif\Desktop\240926_IMR90_1,2,3HZ individual transients result.xlsx"
    if not Path(wb_path).exists():
        pytest.skip(f"Workbook {wb_path} not found on this machine.")

    from dataclasses import replace
    from pixel_counter.config import BeatCounterConfig

    # 1. Default mode: PixelCorrelation-only (calcium disabled by default)
    segs_default, _ = rv.process_workbook(wb_path, segments=[17, 20])
    assert segs_default[1]["auto_polarity"] == 0, "Seg 20 should be UNDETERMINED in default pixel-only mode"

    # 2. Calcium-assisted mode: Explicitly enabled
    cfg_ca = replace(BeatCounterConfig(), v2_use_calcium_polarity=True)
    segs, _ = rv.process_workbook(wb_path, config=cfg_ca, segments=[17, 20])
    assert len(segs) == 2

    for seg in segs:
        # Automatic orientation resolves to -1 via calcium ECC timing
        assert seg["auto_polarity"] == -1
        assert seg["effective_polarity"] == -1
        assert seg["qc_status"] == "PASS"

        # Precomputed orientation "-1" MUST be PASS with 9 contractions
        orient_neg = seg["orientations"]["-1"]
        assert orient_neg["qc_status"] == "PASS"
        assert orient_neg["qc_reason"] == "clean_signal"
        default_events = orient_neg["runs"]["default"]["events"]
        n_accepted = sum(1 for s in default_events["status"] if s == "ACCEPTED")
        assert n_accepted == 9, f"Expected 9 accepted events for Segment {seg['segment_number']} under -1, got {n_accepted}"

    # Segment 25 remains genuinely undetermined and requires manual input
    segs_25, _ = rv.process_workbook(wb_path, config=cfg_ca, segments=[25])
    assert len(segs_25) == 1
    seg25 = segs_25[0]
    assert seg25["auto_polarity"] == 0
    assert seg25["effective_polarity"] == 0
    assert seg25["qc_status"] == "REJECT"
    assert seg25["qc_reason"] == "polarity_undetermined"



