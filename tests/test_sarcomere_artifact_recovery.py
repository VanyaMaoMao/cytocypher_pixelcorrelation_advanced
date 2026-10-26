"""
tests/test_sarcomere_artifact_recovery.py
=========================================

Comprehensive unit and validation tests for conservative sarcomere-only
artifact reconstruction (pixel_counter.sarcomere_artifact_recovery).

Requirements Tested:
1. Non-destructive raw signal preservation: sig_raw is NEVER altered.
2. Narrow localized tracking artifacts (<= 40 ms) are bridged smoothly via shape-preserving PCHIP.
3. Long corrupted intervals (> 40 ms) are not reconstructed.
4. Zero amplitude fabrication: corrupted troughs retain abs_shortening_um = NaN and pct_shortening = NaN.
5. Shoulder excursion bounds: candidates are accepted as MAIN/SECONDARY only if intact shoulders bound the contraction depth.
6. Pure tracking spikes without surrounding mechanical evidence cannot become accepted contractions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pixel_counter.sarcomere_artifact_recovery import (
    reconcile_reconstructed_events,
    reconstruct_narrow_tracking_artifacts,
)
from pixel_counter.sarcomere_detector import detect_sarcomere_contractions
from pixel_counter.sarcomere_qc import detect_sarcomere_artifacts


def _make_synthetic_trace(
    fs: float = 250.0,
    duration_s: float = 6.0,
    l0_um: float = 1.80,
    beat_times_s: list[float] | None = None,
    shortening_um: float = 0.16,
    sigma_s: float = 0.045,
    noise_sd: float = 0.0005,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    sig = np.full(n, l0_um, dtype=float) + rng.normal(0.0, noise_sd, size=n)
    if beat_times_s is None:
        beat_times_s = [1.5, 3.0, 4.5]
    for bt in beat_times_s:
        sig -= shortening_um * np.exp(-0.5 * ((t - bt) / sigma_s) ** 2)
    return t, sig


def test_raw_signal_preservation_and_non_destructive():
    """Verify that reconstruct_narrow_tracking_artifacts leaves sig_raw 100% untouched."""
    fs = 250.0
    _, sig_clean = _make_synthetic_trace(fs=fs)
    sig_raw = sig_clean.copy()
    # Add narrow vertical tracking spike (2 samples) at sample 375 (t=1.5s trough)
    sig_raw[374:376] = 2.2335
    sig_raw_copy = sig_raw.copy()

    art_info = detect_sarcomere_artifacts(sig_raw, fs)
    sig_recon, recon_meta = reconstruct_narrow_tracking_artifacts(sig_raw, fs, art_info)

    # 1. Original sig_raw must be completely identical bit-for-bit
    assert np.array_equal(sig_raw, sig_raw_copy), "sig_raw was modified!"
    # 2. sig_recon must be a separate array
    assert sig_recon is not sig_raw
    # 3. Corrupted samples in sig_recon should be bridged below spike
    assert sig_recon[375] < 2.0


def test_narrow_vs_long_interval_reconstruction():
    """Verify that narrow spikes (<= 40 ms) are bridged while long tracking failures are NOT."""
    fs = 250.0
    _, sig_clean = _make_synthetic_trace(fs=fs, duration_s=8.0, beat_times_s=[2.0, 5.0])
    sig_raw = sig_clean.copy()

    # Spike 1: Narrow vertical spike at sample 500 (t=2.0s): 3 samples = 12 ms (<= 40 ms) -> RECONSTRUCT
    sig_raw[499:502] = 2.2335

    # Spike 2: Long tracking dropout at sample 1250 (t=5.0s): 25 samples = 100 ms (> 40 ms) -> DO NOT RECONSTRUCT
    sig_raw[1240:1265] = 2.2335

    art_info = detect_sarcomere_artifacts(sig_raw, fs)
    sig_recon, recon_meta = reconstruct_narrow_tracking_artifacts(sig_raw, fs, art_info)

    reconstructed_intervals = recon_meta.get("reconstructed_intervals", [])
    # Only the narrow spike should be in reconstructed intervals
    assert len(reconstructed_intervals) == 1
    iv = reconstructed_intervals[0]
    assert iv["start_idx"] <= 500 <= iv["end_idx"]
    assert iv["duration_ms"] <= 40.0


def test_zero_amplitude_fabrication():
    """Verify that events with artifact-interrupted troughs report abs_shortening_um = NaN."""
    fs = 250.0
    _, sig_clean = _make_synthetic_trace(fs=fs, duration_s=6.0, beat_times_s=[1.5, 3.5])
    sig_raw = sig_clean.copy()
    # Add narrow artifact right across the trough of beat 1 (sample 375)
    sig_raw[374:376] = 2.2335

    art_info = detect_sarcomere_artifacts(sig_raw, fs)
    sig_recon, recon_meta = reconstruct_narrow_tracking_artifacts(sig_raw, fs, art_info)

    ev_df_recon, l0_curve, _ = detect_sarcomere_contractions(sig_recon, fs, artifact_info=art_info)
    ev_df = reconcile_reconstructed_events(
        ev_df_recon, sig_raw, fs, art_info, recon_meta, t_main_pct=0.55, t_sce_pct=0.35, l0_curve=l0_curve
    )

    # Find the reconstructed beat near sample 375 (t=1.5s)
    beat1 = ev_df[np.abs(ev_df["time_s"] - 1.5) < 0.1]
    assert len(beat1) == 1
    row = beat1.iloc[0]

    assert bool(row["peak_corrupted"]) is True
    assert bool(row["has_valid_amplitude"]) is False
    assert row["recovery_status"] == "RECONSTRUCTED_ARTIFACT"
    assert np.isnan(row["abs_shortening_um"]), "Amplitudes must NOT be fabricated!"
    assert np.isnan(row["pct_shortening"]), "Percentages must NOT be fabricated!"
    # Since shoulders are deep, it passes as MAIN
    assert row["status"] == "ACCEPTED"
    assert row["event_class"] == "MAIN"

    # Beat 2 (uncorrupted) must retain its authentic measured amplitude
    beat2 = ev_df[np.abs(ev_df["time_s"] - 3.5) < 0.1]
    assert len(beat2) == 1
    row2 = beat2.iloc[0]
    assert bool(row2["peak_corrupted"]) is False
    assert bool(row2["has_valid_amplitude"]) is True
    assert pd.notna(row2["abs_shortening_um"])
    assert pd.notna(row2["pct_shortening"])
    assert 0.14 <= row2["abs_shortening_um"] <= 0.18


def test_pure_tracking_spikes_without_shoulders_rejected():
    """Verify that pure tracking spikes on flat baseline do not become accepted contractions."""
    fs = 250.0
    _, sig_flat = _make_synthetic_trace(fs=fs, duration_s=6.0, beat_times_s=[], shortening_um=0.0)
    sig_raw = sig_flat.copy()
    # Insert downward tracking spikes (not a contraction waveform)
    sig_raw[300] = 1.30
    sig_raw[700] = 1.30

    art_info = detect_sarcomere_artifacts(sig_raw, fs)
    sig_recon, recon_meta = reconstruct_narrow_tracking_artifacts(sig_raw, fs, art_info)

    ev_df_recon, l0_curve, _ = detect_sarcomere_contractions(sig_recon, fs, artifact_info=art_info)
    ev_df = reconcile_reconstructed_events(
        ev_df_recon, sig_raw, fs, art_info, recon_meta, t_main_pct=0.55, t_sce_pct=0.35, l0_curve=l0_curve
    )

    accepted = ev_df[ev_df["status"] == "ACCEPTED"] if (ev_df is not None and not ev_df.empty) else pd.DataFrame()
    assert len(accepted) == 0, "Pure tracking spikes must not become accepted contractions!"
