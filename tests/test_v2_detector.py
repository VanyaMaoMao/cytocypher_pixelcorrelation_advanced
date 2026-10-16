import numpy as np
import pytest

from pixel_counter.config import BeatCounterConfig
from pixel_counter.v2_detector import ACCEPTED, REJECTED, DetectorParams, detect_contractions
from pixel_counter.v2_pipeline import analyze_v2_pipeline
from pixel_counter.v2_signal import compute_amplitude_waveform
from tests.v2_synth import FS, gauss, make_amplitude, to_raw

CENTERS = [0.8, 1.8, 2.8, 3.8, 4.8]


def accepted(res):
    ev = res["events"]
    return ev[ev["status"] == ACCEPTED]


def run(a, params=None):
    return detect_contractions(a, FS, params)


def test_regular_contractions_localised_at_maxima():
    t, a = make_amplitude(CENTERS)
    acc = accepted(run(a))
    assert len(acc) == 5
    np.testing.assert_allclose(acc["time_rel_s"].to_numpy(), CENTERS, atol=0.03)
    # every accepted marker is a genuine local maximum of the plotted waveform
    for i in acc["index"]:
        lo, hi = max(0, i - 3), min(len(a), i + 4)
        assert a[i] == a[lo:hi].max()
        assert a[i] > 0.5          # never on a resting-state minimum or a slope near rest


def test_only_two_outcomes():
    t, a = make_amplitude(CENTERS, extra=lambda t: gauss(t, 1.3, 0.03, 0.08) + gauss(t, 2.3, 0.03, 0.08))
    ev = run(a)["events"]
    assert set(ev["status"]) <= {ACCEPTED, REJECTED}
    assert (ev.loc[ev["status"] == REJECTED, "reason"] != "").all()


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.0])
def test_positive_amplitude_scaling_invariance(scale):
    t, a = make_amplitude(CENTERS, noise=0.003)
    base = accepted(run(a))
    scaled = accepted(run(scale * a))
    assert len(scaled) == len(base)
    np.testing.assert_array_equal(scaled["index"].to_numpy(), base["index"].to_numpy())


@pytest.mark.parametrize("sigma", [0.02, 0.05, 0.12])
def test_different_contraction_widths(sigma):
    t, a = make_amplitude(CENTERS, sigma=sigma)
    acc = accepted(run(a))
    assert len(acc) == 5
    np.testing.assert_allclose(acc["time_rel_s"].to_numpy(), CENTERS, atol=0.04)


def test_genuine_lower_amplitude_contractions_are_kept():
    amps = [1.0, 0.65, 1.0, 0.6, 0.9]       # 60-65 % contractions are real and must survive defaults
    t, a = make_amplitude(CENTERS, amps=amps)
    assert len(accepted(run(a))) == 5


def test_small_secondary_fluctuations_are_rejected():
    extra = lambda t: sum(gauss(t, c + 0.4, 0.03, 0.12) for c in CENTERS)
    t, a = make_amplitude(CENTERS, extra=extra)
    res = run(a)
    assert len(accepted(res)) == 5
    rej = res["events"][res["events"]["status"] == REJECTED]
    assert len(rej) >= 5
    assert set(rej["reason"]) <= {"insufficient_amplitude", "secondary_morphological_feature", "duplicate_peak_within_cycle", "baseline_noise", "incomplete_excursion"}


def test_secondary_shoulder_does_not_create_event():
    extra = lambda t: sum(gauss(t, c + 0.11, 0.05, 0.35) for c in CENTERS)
    t, a = make_amplitude(CENTERS, sigma=0.06, extra=extra)
    assert len(accepted(run(a))) == 5


def test_double_peaked_contraction_is_one_event_first_peak_time():
    # two maxima 0.12 s apart with a shallow notch -> one contraction, timestamp = first maximum
    amps_t = []
    t = np.arange(int(6 * FS)) / FS
    a = np.zeros_like(t)
    for c in CENTERS:
        a += gauss(t, c, 0.04, 0.95) + gauss(t, c + 0.12, 0.04, 1.0)
    a += np.random.default_rng(1).normal(0, 0.003, t.size)
    res = run(a)
    acc = accepted(res)
    assert len(acc) == 5
    # time-to-peak: the earliest near-maximal maximum (first lobe at c, or the merged top)
    assert np.all(acc["time_rel_s"].to_numpy() < np.array(CENTERS) + 0.09)
    dup = res["events"][res["events"]["reason"] == "duplicate_peak_within_cycle"]
    assert len(dup) >= 1


def test_two_genuinely_separate_nearby_contractions():
    t, a = make_amplitude([1.0, 1.22, 3.0, 3.22])      # 0.22 s apart, full recovery in between
    assert len(accepted(run(a))) == 4


def test_recovery_parameter_controls_splitting():
    t = np.arange(int(4 * FS)) / FS
    a = gauss(t, 1.0, 0.04) + gauss(t, 1.14, 0.04, 0.9) + gauss(t, 3.0, 0.04) + gauss(t, 3.14, 0.04, 0.9)
    strict = len(accepted(run(a, DetectorParams(min_rel_recovery=0.7))))
    default = len(accepted(run(a)))
    permissive = len(accepted(run(a, DetectorParams(min_rel_recovery=0.1))))
    assert strict <= default <= permissive
    assert permissive == 4 and strict == 2


def test_amplitude_parameter_monotonic():
    amps = [1.0, 0.7, 0.45, 0.3, 1.0]
    t, a = make_amplitude(CENTERS, amps=amps)
    counts = [len(accepted(run(a, DetectorParams(min_rel_amplitude=x)))) for x in (0.2, 0.4, 0.6, 0.8)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 5 and counts[-1] <= 3


def test_baseline_noise_only_gives_no_contractions():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, int(5 * FS))
    res = run(a)
    assert len(accepted(res)) == 0


def test_nan_gap_does_not_break_detection():
    t, a = make_amplitude(CENTERS)
    a[int(2.2 * FS):int(2.5 * FS)] = np.nan          # gap between contractions
    acc = accepted(run(a))
    assert len(acc) == 5
    assert not np.isnan(a[acc["index"].to_numpy()]).any()


def test_invalid_parameters_rejected():
    for kw in ({"min_rel_amplitude": 0.0}, {"min_rel_amplitude": 1.2}, {"min_rel_recovery": 0.0}, {"min_snr": -1.0}):
        with pytest.raises(ValueError):
            DetectorParams(**kw).validate()


# ---------------------------------------------------------------------------------------------
# full pipeline: raw -> polarity -> amplitude -> events
# ---------------------------------------------------------------------------------------------
def _pipeline(raw, t0=0.0, **cfg):
    return analyze_v2_pipeline(raw, FS, BeatCounterConfig(**cfg), t0=t0, meta={"time_origin_s": t0})


@pytest.mark.parametrize("pol", [1, -1])
def test_pipeline_polarity_and_event_agreement(pol):
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, pol)
    res = _pipeline(raw)
    assert res["metadata"]["qc_status"] == "PASS"
    assert res["polarity_result"]["best_polarity"] == pol
    assert res["metadata"]["n_accepted"] == 5
    # waveform is baseline-relative, contractions upward, raw untouched
    wf = res["waveform"]
    np.testing.assert_array_equal(wf["raw"], raw)
    np.testing.assert_allclose(wf["amplitude"], pol * (raw - wf["baseline_raw"]))
    assert np.nanmedian(wf["amplitude"]) < 0.1


def test_pipeline_inversion_gives_identical_events():
    t, a = make_amplitude(CENTERS)
    up = _pipeline(to_raw(a, 1))
    down = _pipeline(to_raw(a, -1))
    np.testing.assert_array_equal(up["candidates"]["candidate_index"], down["candidates"]["candidate_index"])


def test_pipeline_baseline_shift_invariance():
    t, a = make_amplitude(CENTERS)
    r0 = _pipeline(to_raw(a, 1, offset=0.3))
    r1 = _pipeline(to_raw(a, 1, offset=2.7))
    np.testing.assert_array_equal(r0["candidates"]["candidate_index"], r1["candidates"]["candidate_index"])


@pytest.mark.parametrize("start", [0.0, 0.35, 0.9, 1.4])
def test_pipeline_independent_of_recording_start_phase(start):
    t, a = make_amplitude(CENTERS + [5.8], dur=7.0)
    n0 = int(start * FS)
    raw = to_raw(a, 1)[n0:]
    res = _pipeline(raw)
    expected = [c - start for c in CENTERS + [5.8] if c - start > 0.15]
    got = res["candidates"]["time_s"]
    assert res["polarity_result"]["best_polarity"] == 1
    assert len(got) == len(expected) or len(got) == len(expected) - 1   # a cut-off first contraction may be lost
    for g in got:
        assert np.min(np.abs(np.array(expected) - g)) < 0.04


def test_pipeline_absolute_timestamps_and_marker_consistency():
    t, a = make_amplitude(CENTERS)
    t0 = 44.2998
    res = _pipeline(to_raw(a, 1), t0=t0)
    ev = res["events"]
    acc = ev[ev["status"] == ACCEPTED]
    np.testing.assert_allclose(acc["time_abs_s"], t0 + acc["index"] / FS)
    np.testing.assert_allclose(res["candidates"]["time_s"], acc["time_abs_s"])
    np.testing.assert_allclose(acc["amplitude"], res["waveform"]["amplitude"][acc["index"].to_numpy()])
    np.testing.assert_allclose(acc["raw_value"], res["waveform"]["raw"][acc["index"].to_numpy()])
    np.testing.assert_allclose(res["waveform"]["time_s"][acc["index"].to_numpy()], acc["time_abs_s"])


def test_pipeline_reports_active_parameters():
    t, a = make_amplitude(CENTERS)
    res = analyze_v2_pipeline(to_raw(a, 1), FS, BeatCounterConfig(v2_min_rel_amplitude=0.4, v2_min_rel_recovery=0.6, v2_min_snr=9.0))
    assert res["metadata"]["detector_params"] == {"min_rel_amplitude": 0.4, "min_rel_recovery": 0.6, "min_snr": 9.0}


def test_manual_polarity_override():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, 1)
    res = analyze_v2_pipeline(raw, FS, BeatCounterConfig(v2_polarity_mode="negative"))
    assert res["polarity_result"]["best_polarity"] == -1
    assert res["polarity_result"]["method"] == "manual_override"


def test_pipeline_nan_gap_preserved():
    t, a = make_amplitude(CENTERS)
    raw = to_raw(a, 1)
    raw[int(2.2 * FS):int(2.5 * FS)] = np.nan
    res = _pipeline(raw)
    assert res["metadata"]["qc_status"] == "PASS"
    assert "nan_gaps" in res["metadata"]["qc_flags"]
    assert res["metadata"]["n_accepted"] == 5
    assert np.isnan(res["waveform"]["amplitude"][int(2.3 * FS)])


def test_amplitude_waveform_from_known_baseline():
    raw = np.concatenate([np.full(500, 0.4), np.full(10, 0.9), np.full(500, 0.4)])
    wf = compute_amplitude_waveform(raw, FS, 1)
    assert wf["amplitude"].max() == pytest.approx(0.5, abs=0.02)
    wf2 = compute_amplitude_waveform(1.0 - raw, FS, -1)
    np.testing.assert_allclose(wf2["amplitude"], wf["amplitude"], atol=1e-9)
    with pytest.raises(ValueError):
        compute_amplitude_waveform(raw, FS, 0)
