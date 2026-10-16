"""Experimental v2 signal representation: polarity, resting baseline and contraction amplitude.

Data-flow contract (single source of truth)
-------------------------------------------
``S_raw``  original PixelCorrelation waveform, stitched but NEVER inverted.
``polarity`` authoritative physiological orientation, decided exactly once from ``S_raw``:
             -1  contractions are downward excursions of ``S_raw``
             +1  contractions are upward excursions of ``S_raw``
              0  the waveform does not support a reliable decision (segment-level failure)
``B_raw``  resting-state baseline expressed in raw units.
``A(t) = polarity * (S_raw(t) - B_raw(t))``  baseline-relative contraction amplitude.
           Contractions are upward in ``A`` and the resting state is ~0.  Nothing is
           rectified, normalised or clipped; small negative residuals stay in the data.

Every later stage (detection, QC, plotting, export) consumes ``A`` plus the polarity
metadata; nothing downstream re-infers or re-applies orientation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, savgol_filter, correlate

# ---------------------------------------------------------------------------
# Internal numerical constants (documented, deliberately not user-facing)
# ---------------------------------------------------------------------------
DWELL_SMOOTH_S = 0.10        # smoothing used to judge "stationary" resting dwell
DWELL_SLOPE_QUANTILE = 30.0  # a sample is stationary if its |slope| is in the slowest 30 %
DWELL_BAND = 0.25            # an extreme state = outer 25 % of the 2-98 % amplitude range
DWELL_DECISIVE = 0.35        # |dwell asymmetry| needed for a decision on dwell alone
ASYM_MIN_LOG_RATIO = 0.35    # |log(fall/rise)| needed (both estimators) for the onset cue
CURV_DECISIVE = 0.85         # |curvature log ratio| needed (>= 2.34x sharpness ratio)
ENV_DECISIVE = 0.70          # |envelope stability log ratio| needed (>= 2.0x stability ratio)
MIN_VALID_S = 1.0            # minimum amount of valid signal for any polarity decision
MIN_EXCURSION_SNR = 10.0     # 2-98 % excursion range must exceed 10 x white-noise sigma (pure noise has ~4)

BASELINE_QUANTILE = 0.10     # resting level = lower decile of the oriented waveform ...
BASELINE_WINDOW_S = 2.0      # ... within a centred 2 s window (longer than one contraction episode)
BASELINE_SMOOTH_S = 0.30     # Gaussian sigma applied to the quantile trace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def valid_blocks(x: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Half-open index ranges of contiguous finite samples."""
    ok = np.isfinite(x)
    if not ok.any():
        return []
    d = np.diff(np.concatenate([[0], ok.view(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def estimate_noise_sigma(x: np.ndarray) -> float:
    """White-noise sigma from the MAD of first differences (insensitive to slow waveform)."""
    parts = []
    for s, e in valid_blocks(x, 3):
        parts.append(np.diff(x[s:e]))
    if not parts:
        return float("nan")
    d = np.concatenate(parts)
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------
def _dwell_asymmetry(x: np.ndarray, fs: float) -> float:
    """Resting-state dwell asymmetry in [-1, 1].

    The mechanical resting state is stationary: the waveform lingers there with a
    near-zero slope, whereas a contraction is a transient excursion.  We count the
    stationary samples (slowest 30 % of |slope| after 100 ms smoothing) that lie in the
    top band versus the bottom band of the amplitude range.

        D = (n_top - n_bottom) / (n_top + n_bottom)

    D > 0: rest sits at the top  -> contractions are downward (polarity -1).
    D < 0: rest sits at the bottom -> contractions are upward (polarity +1).
    Invariant to baseline shift, positive scaling, and antisymmetric under inversion.
    """
    k = max(3, int(DWELL_SMOOTH_S * fs) | 1)
    sm, slopes = [], []
    for s, e in valid_blocks(x, k * 2):
        b = x[s:e]
        xs = uniform_filter1d(median_filter(b, size=k, mode="nearest"), k, mode="nearest")
        sm.append(xs)
        slopes.append(np.abs(np.gradient(xs)) * fs)
    if not sm:
        return float("nan")
    xs = np.concatenate(sm)
    dx = np.concatenate(slopes)
    lo, hi = np.percentile(xs, [2, 98])
    rng = hi - lo
    if rng <= 0:
        return float("nan")
    stationary = dx <= np.percentile(dx, DWELL_SLOPE_QUANTILE)
    n_top = np.mean(stationary & (xs > hi - DWELL_BAND * rng))
    n_bot = np.mean(stationary & (xs < lo + DWELL_BAND * rng))
    if n_top + n_bot <= 0:
        return float("nan")
    return float((n_top - n_bot) / (n_top + n_bot))


def _onset_asymmetry(x: np.ndarray, fs: float) -> Tuple[float, float]:
    """Contraction-onset asymmetry cues (positive: the waveform rises faster than it falls).

    Physiological prior: a contraction starts abruptly and relaxation is slower.
    Two independent estimators are returned (both are log ratios, > 0 favours +1):
      * slope ratio:   log( p99 slope up / p99 slope down )
      * edge duration: log( median 10-90 % fall time / median 10-90 % rise time )
    """
    slope_ratios, rises, falls = [], [], []
    ups, downs = [], []
    for s, e in valid_blocks(x, 15):
        b = savgol_filter(x[s:e], 7 if e - s >= 7 else 5, 2)
        d = np.diff(b) * fs
        ups.append(d[d > 0]); downs.append(-d[d < 0])
        rng = np.percentile(b, 97) - np.percentile(b, 3)
        if rng <= 0:
            continue
        prom = 0.25 * rng
        pk, _ = find_peaks(b, prominence=prom)
        tr, _ = find_peaks(-b, prominence=prom)
        ext = sorted([(int(i), 1) for i in pk] + [(int(i), -1) for i in tr])
        clean: List[Tuple[int, int]] = []
        for i, t in ext:  # keep the more extreme of consecutive same-type extrema
            if clean and clean[-1][1] == t:
                if (t == 1 and b[i] > b[clean[-1][0]]) or (t == -1 and b[i] < b[clean[-1][0]]):
                    clean[-1] = (i, t)
            else:
                clean.append((i, t))
        for (a, ta), (c, tc) in zip(clean[:-1], clean[1:]):
            seg = b[a:c + 1]
            sw = seg[-1] - seg[0]
            if abs(sw) < 1e-12:
                continue
            frac = (seg - seg[0]) / sw
            i10, i90 = int(np.argmax(frac >= 0.1)), int(np.argmax(frac >= 0.9))
            dur = (i90 - i10) / fs
            if dur <= 0:
                continue
            (rises if ta == -1 else falls).append(dur)
    u = np.concatenate(ups) if ups else np.array([])
    dn = np.concatenate(downs) if downs else np.array([])
    slope_cue = float("nan")
    if u.size > 10 and dn.size > 10:
        slope_cue = float(np.log(np.percentile(u, 99) / max(np.percentile(dn, 99), 1e-12)))
    edge_cue = float("nan")
    if len(rises) >= 2 and len(falls) >= 2:
        edge_cue = float(np.log(np.median(falls) / np.median(rises)))
    return slope_cue, edge_cue


def _extremum_curvature_asymmetry(x: np.ndarray, fs: float) -> float:
    """Extremum sharpness/curvature asymmetry log(kappa_tr / kappa_pk).

    Contractions are transient excursions with sharper turnaround than the
    broader resting state.
      * curv_cue > 0: troughs are sharper (favours -1, downward contractions)
      * curv_cue < 0: peaks are sharper (favours +1, upward contractions)
    Invariant to positive amplitude scaling and baseline shift; antisymmetric
    under inversion.
    """
    sigma_n = max(1.0, 0.012 * fs)
    min_dist = max(5, int(0.15 * fs))
    pk_curvs, tr_curvs = [], []
    for s, e in valid_blocks(x, int(0.5 * fs)):
        b = x[s:e]
        lo, hi = np.percentile(b, [2, 98])
        rng = hi - lo
        if rng <= 0:
            continue
        bs = gaussian_filter1d(b, sigma_n, mode="nearest")
        pks, _ = find_peaks(bs, prominence=0.20 * rng, distance=min_dist)
        trs, _ = find_peaks(-bs, prominence=0.20 * rng, distance=min_dist)
        if pks.size == 0 or trs.size == 0:
            continue
        d2x = np.gradient(np.gradient(bs)) * (fs**2)
        p_c = -d2x[pks]
        t_c = d2x[trs]
        pk_curvs.extend(p_c[p_c > 0])
        tr_curvs.extend(t_c[t_c > 0])
    if len(pk_curvs) < 2 or len(tr_curvs) < 2:
        return float("nan")
    mp = float(np.mean(pk_curvs))
    mt = float(np.mean(tr_curvs))
    if mp <= 0 or mt <= 0:
        return float("nan")
    return float(np.log(mt / mp))


def _envelope_stability_asymmetry(x: np.ndarray, fs: float) -> float:
    """Envelope baseline stability log ratio: log(std(p90) / std(p10)).

    The resting state baseline level tracks more stably across beats than the
    contraction excursion peaks.
      * env_cue > 0: p10 is more stable (favours +1, baseline is bottom)
      * env_cue < 0: p90 is more stable (favours -1, baseline is top)
    Invariant to positive amplitude scaling and baseline shift; antisymmetric
    under inversion.
    """
    win = max(5, int(round(1.5 * fs)))
    p90_stds, p10_stds = [], []
    for s, e in valid_blocks(x, int(2.0 * fs)):
        b = x[s:e]
        lo, hi = np.percentile(b, [2, 98])
        rng = hi - lo
        if rng <= 0:
            continue
        bs = gaussian_filter1d(b, max(1.0, 0.012 * fs), mode="nearest")
        s90 = pd.Series(bs).rolling(win, center=True, min_periods=win // 2).quantile(0.90).to_numpy()
        s10 = pd.Series(bs).rolling(win, center=True, min_periods=win // 2).quantile(0.10).to_numpy()
        ok = np.isfinite(s90) & np.isfinite(s10)
        if np.sum(ok) < win:
            continue
        p90_stds.append(float(np.std(s90[ok])))
        p10_stds.append(float(np.std(s10[ok])))
    if not p90_stds or not p10_stds:
        return float("nan")
    m90 = float(np.mean(p90_stds))
    m10 = float(np.mean(p10_stds))
    if m90 <= 0 or m10 <= 0:
        return float("nan")
    return float(np.log(m90 / m10))


def _calcium_ecc_polarity(
    sig_raw: np.ndarray,
    fs: float,
    sig_cal: np.ndarray,
    fs_cal: float,
    t0_raw: float = 0.0,
    t0_cal: float = 0.0,
) -> Dict:
    """Evaluate physiological polarity from excitation-contraction coupling (ECC) timing.

    Physiological principle:
      In cardiac myocytes, calcium influx and sarcoplasmic reticulum release (calcium transient)
      triggers actomyosin crossbridge cycling and mechanical contraction.
      Genuine mechanical contraction peaks follow intracellular calcium peaks within a
      causal systolic latency window (typically 30-160 ms in human iPSC-CMs).
      The false inverted orientation corresponds to diastolic relaxation / troughs, which
      occur either non-causally (before or simultaneously with calcium) or late in the pacing cycle.
    """
    if sig_cal is None or len(sig_cal) == 0 or not np.any(np.isfinite(sig_cal)):
        return {"polarity": 0, "status": "NO_CALCIUM", "reason": "no_calcium_data"}

    # Align auxiliary calcium time axis to raw mechanical signal
    time_raw = t0_raw + np.arange(len(sig_raw)) / fs
    time_cal = t0_cal + np.arange(len(sig_cal)) / fs_cal
    if len(sig_cal) != len(sig_raw) or abs(t0_cal - t0_raw) > 1e-4 or abs(fs_cal - fs) > 1e-4:
        sig_cal_aligned = np.interp(time_raw, time_cal, sig_cal, left=np.nan, right=np.nan)
        time_cal = time_raw
        fs_cal = fs
    else:
        sig_cal_aligned = sig_cal

    # Calcium transient peak detection (RatioMetricCalcium transients are positive-going)
    ca_smooth = gaussian_filter1d(sig_cal_aligned, sigma=max(1.0, 0.012 * fs_cal), mode="nearest")
    ca_rng = np.nanpercentile(ca_smooth, 98) - np.nanpercentile(ca_smooth, 2)
    if not np.isfinite(ca_rng) or ca_rng <= 0:
        return {"polarity": 0, "status": "INVALID_CALCIUM", "reason": "flat_calcium"}

    pks_ca, _ = find_peaks(ca_smooth, distance=int(0.18 * fs_cal), prominence=max(0.12 * ca_rng, 1e-4))
    if len(pks_ca) < 2:
        return {"polarity": 0, "status": "INSUFFICIENT_CALCIUM", "reason": "insufficient_ca_peaks"}
    t_ca = time_cal[pks_ca]

    # Mechanical candidate peak detection on smoothed +S and -S
    sigma_m = max(1.0, 0.015 * fs)
    s_smooth = gaussian_filter1d(sig_raw, sigma_m, mode="nearest")
    s_rng = np.nanpercentile(s_smooth, 98) - np.nanpercentile(s_smooth, 2)
    if not np.isfinite(s_rng) or s_rng <= 0:
        return {"polarity": 0, "status": "INVALID_MECHANICAL", "reason": "flat_mechanical"}

    prom_m = max(0.10 * s_rng, 1e-4)
    dist_m = int(0.18 * fs)
    pks_pos, _ = find_peaks(s_smooth, distance=dist_m, prominence=prom_m)
    pks_neg, _ = find_peaks(-s_smooth, distance=dist_m, prominence=prom_m)
    t_pos = time_raw[pks_pos]
    t_neg = time_raw[pks_neg]

    # Cycle-adaptive systolic contraction latency window:
    # Contraction peak occurs 20ms up to 45% of pacing cycle length (capped between 140ms and 240ms)
    t_diffs = np.diff(t_ca)
    t_cycle = float(np.median(t_diffs)) if len(t_diffs) else 0.500
    max_sys_delay = max(0.140, min(0.240, 0.45 * t_cycle))

    # Match each calcium peak to earliest following mechanical peak within systolic window
    delays_pos = []
    for tc in t_ca:
        d = [tp - tc for tp in t_pos if 0.020 <= tp - tc <= max_sys_delay]
        if d:
            delays_pos.append(d[0])

    delays_neg = []
    for tc in t_ca:
        d = [tn - tc for tn in t_neg if 0.020 <= tn - tc <= max_sys_delay]
        if d:
            delays_neg.append(d[0])

    n_ca = len(t_ca)
    m_pos = len(delays_pos)
    m_neg = len(delays_neg)
    frac_pos = m_pos / n_ca
    frac_neg = m_neg / n_ca

    med_pos = float(np.median(delays_pos) * 1000) if m_pos else np.nan
    std_pos = float(np.std(delays_pos) * 1000) if m_pos >= 2 else np.nan

    med_neg = float(np.median(delays_neg) * 1000) if m_neg else np.nan
    std_neg = float(np.std(delays_neg) * 1000) if m_neg >= 2 else np.nan

    pol = 0
    reason = "ecc_insufficient_matches"
    if m_pos >= 2 and frac_pos >= 0.40 and (m_pos >= m_neg + 2 or (m_neg == 0 and m_pos >= 2)):
        pol = 1
        reason = f"ecc_systolic_match_pos_{m_pos}/{n_ca}_med_{med_pos:.0f}ms_vs_neg_{m_neg}/{n_ca}"
    elif m_neg >= 2 and frac_neg >= 0.40 and (m_neg >= m_pos + 2 or (m_pos == 0 and m_neg >= 2)):
        pol = -1
        reason = f"ecc_systolic_match_neg_{m_neg}/{n_ca}_med_{med_neg:.0f}ms_vs_pos_{m_pos}/{n_ca}"
    elif m_pos >= 2 and m_neg >= 2 and min(frac_pos, frac_neg) >= 0.40:
        pos_plausible = (35 <= med_pos <= 150)
        neg_plausible = (35 <= med_neg <= 150)
        if pos_plausible and not neg_plausible:
            pol = 1
            reason = f"ecc_latency_plausibility_pos_{med_pos:.0f}ms_vs_neg_{med_neg:.0f}ms"
        elif neg_plausible and not pos_plausible:
            pol = -1
            reason = f"ecc_latency_plausibility_neg_{med_neg:.0f}ms_vs_pos_{med_pos:.0f}ms"
        elif np.isfinite(std_pos) and np.isfinite(std_neg):
            if std_pos < std_neg * 0.6:
                pol = 1
                reason = f"ecc_lower_jitter_pos_{std_pos:.1f}ms_vs_neg_{std_neg:.1f}ms"
            elif std_neg < std_pos * 0.6:
                pol = -1
                reason = f"ecc_lower_jitter_neg_{std_neg:.1f}ms_vs_pos_{std_pos:.1f}ms"
            else:
                reason = f"ecc_ambiguous_jitter_pos_{std_pos:.1f}ms_neg_{std_neg:.1f}ms"
        else:
            reason = "ecc_ambiguous_both_systolic"

    return {
        "polarity": pol,
        "status": "OK" if pol != 0 else "UNDETERMINED",
        "method": "calcium_ecc_timing" if pol != 0 else "none",
        "reason": reason,
        "n_ca": n_ca,
        "m_pos": m_pos, "med_pos_ms": med_pos, "std_pos_ms": std_pos,
        "m_neg": m_neg, "med_neg_ms": med_neg, "std_neg_ms": std_neg,
        "t_cycle_s": t_cycle,
    }


def _estimate_pacing_cycle(sig: np.ndarray, fs: float) -> float:
    """Estimate pacing cycle length T_cycle (s) from raw PixelCorrelation autocorrelation."""
    finite = sig[np.isfinite(sig)]
    if len(finite) < int(fs):
        return 0.50
    s = gaussian_filter1d(finite, max(1.0, 0.015 * fs))
    d = s - gaussian_filter1d(s, max(2.0, 0.50 * fs))
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    ac /= max(ac[0], 1e-12)
    min_lag = int(0.18 * fs)
    max_lag = int(min(len(ac) - 1, 1.5 * fs))
    if min_lag >= max_lag:
        return 0.50
    pks, _ = find_peaks(ac[min_lag:max_lag], distance=max(3, int(0.15 * fs)), prominence=0.15)
    if len(pks):
        return float((pks[0] + min_lag) / fs)
    return 0.50


def _cycle_morphology_cues(finite: np.ndarray, fs: float, tc: float) -> dict:
    """Extract robust cycle-aligned representative waveform and inter-cycle consistency metrics."""
    p_len = int(round(tc * fs))
    if p_len < 15 or len(finite) // p_len < 3:
        return {}

    trend = gaussian_filter1d(finite, max(5.0, 1.5 * fs))
    detr = finite - trend
    n_cycles = len(detr) // p_len
    cycles = [detr[i * p_len : (i + 1) * p_len] for i in range(n_cycles)]
    template = np.median(cycles, axis=0)
    max_shift = max(2, p_len // 4)
    aligned = []
    for c in cycles:
        ccorr = correlate(c - np.mean(c), template - np.mean(template), mode="full")
        lags = np.arange(-len(template) + 1, len(c))
        valid = (lags >= -max_shift) & (lags <= max_shift)
        best_lag = lags[valid][np.argmax(ccorr[valid])]
        aligned.append(np.roll(c, -best_lag))

    med_c = np.median(aligned, axis=0)
    i_min, i_max = int(np.argmin(med_c)), int(np.argmax(med_c))
    n = len(med_c)

    # Circular rise and fall sample counts
    t_rise = (i_max - i_min) % n
    t_fall = (i_min - i_max) % n
    dur_ratio = float(np.log(max(t_fall, 1) / max(t_rise, 1)))

    # Inter-cycle peak vs trough variability
    pk_vals = [float(np.max(c)) for c in aligned]
    tr_vals = [float(np.min(c)) for c in aligned]
    std_pk = float(np.std(pk_vals))
    std_tr = float(np.std(tr_vals))
    var_ratio = float(np.log(max(std_pk, 1e-6) / max(std_tr, 1e-6)))

    return {
        "cycle_dur_ratio": dur_ratio,
        "cycle_var_ratio": var_ratio,
        "cycle_std_pk": std_pk,
        "cycle_std_tr": std_tr,
    }


def _morphology_polarity(sig_raw: np.ndarray, fs: float) -> Dict:
    """Evaluate physiological polarity from PixelCorrelation morphological cues."""
    x = np.asarray(sig_raw, dtype=float)
    out = {"polarity": 0, "status": "UNDETERMINED", "method": "none", "polarity_source": "automatic",
           "dwell": float("nan"), "slope_cue": float("nan"), "edge_cue": float("nan"),
           "curv_cue": float("nan"), "env_cue": float("nan"), "vel_cue": float("nan"),
           "cycle_dur_ratio": float("nan"), "cycle_var_ratio": float("nan"), "reason": ""}
    if not np.isfinite(fs) or fs <= 0 or np.isfinite(x).sum() / fs < MIN_VALID_S:
        out["reason"] = "insufficient_valid_data"
        return out
    d = _dwell_asymmetry(x, fs)
    out["dwell"] = d
    sn = estimate_noise_sigma(x)
    finite = x[np.isfinite(x)]
    excursion = float(np.percentile(finite, 98) - np.percentile(finite, 2))
    if not np.isfinite(sn) or excursion < MIN_EXCURSION_SNR * max(sn, 1e-12):
        out["reason"] = "no_resolvable_excursion"
        return out

    sc, ec = _onset_asymmetry(x, fs)
    out["slope_cue"], out["edge_cue"] = sc, ec
    cc = _extremum_curvature_asymmetry(x, fs)
    out["curv_cue"] = cc
    ecue = _envelope_stability_asymmetry(x, fs)
    out["env_cue"] = ecue

    tc = _estimate_pacing_cycle(finite, fs)
    is_high_freq = (tc <= 0.40)  # pacing >= 2.5 Hz (e.g. 3 Hz)
    freq = 1.0 / tc if tc > 0 else 0.0

    # Calculate 95th percentile derivative velocity ratio
    dx = np.diff(finite) * fs
    pos_dx = dx[dx > 0]
    neg_dx = -dx[dx < 0]
    v_up = np.percentile(pos_dx, 95) if len(pos_dx) > 10 else 1.0
    v_dn = np.percentile(neg_dx, 95) if len(neg_dx) > 10 else 1.0
    vel_log_ratio = float(np.log(v_up / max(v_dn, 1e-12)))
    out["vel_cue"] = vel_log_ratio

    # Standard directional votes
    dwell_thresh = 0.75 if is_high_freq else DWELL_DECISIVE
    v_dwell = (-1 if d > 0 else 1) if (np.isfinite(d) and abs(d) >= dwell_thresh) else 0

    min_onset = 0.55 if is_high_freq else ASYM_MIN_LOG_RATIO
    v_onset = (1 if sc > 0 else -1) if (np.isfinite(sc) and np.isfinite(ec) and np.sign(sc) == np.sign(ec)
                                        and min(abs(sc), abs(ec)) >= min_onset) else 0
    v_curv = (-1 if cc > 0 else 1) if (np.isfinite(cc) and abs(cc) >= CURV_DECISIVE) else 0
    v_env = (1 if ecue > 0 else -1) if (np.isfinite(ecue) and abs(ecue) >= ENV_DECISIVE) else 0

    # Physiological Refinement Rule 1: Dwell + Curvature consensus at low/mid frequency (< 2.5 Hz)
    # Cell spends >= 70% of time at top baseline (d >= 0.40) and troughs are much sharper than peaks (cc >= 0.80).
    # Overrules apparent onset recoil velocity bounce.
    if not is_high_freq and np.isfinite(d) and d >= 0.40 and np.isfinite(cc) and cc >= 0.80:
        out.update(polarity=-1, status="OK", method="dwell_and_curvature_consensus",
                   reason=f"dwell_{d:+.2f}_and_curv_{cc:+.2f}_consensus")
        return out

    # Physiological Refinement Rule 2: Overwhelming onset kinetics at 2 Hz without strong opposing curvature
    # In prolonged contracture, median dwell is depressed, but active contraction downstroke velocity
    # is overwhelmingly faster (|sc| >= 0.85) than relaxation, and curvature does not contradict.
    # Exclude cases with extreme dwell (|d| >= 0.85) which represent contradictory resting state evidence (e.g. Seg 25).
    if 1.5 <= freq <= 2.4 and v_onset != 0 and abs(d) < 0.85:
        if min(abs(sc), abs(ec)) >= 0.40 and max(abs(sc), abs(ec)) >= 0.85:
            if not (v_curv != 0 and v_curv != v_onset):
                out.update(polarity=v_onset, status="OK", method="overwhelming_onset_kinetics",
                           reason=f"overwhelming_onset_sc_{sc:+.2f}_ec_{ec:+.2f}")
                return out

    # Case 1: Dwell is decisive
    if v_dwell != 0:
        # Check if Onset strongly contradicts Dwell
        if v_onset != 0 and v_onset != v_dwell:
            if v_curv == v_onset:
                out.update(polarity=v_onset, status="OK", method="onset_and_curvature_consensus",
                           reason=f"onset_{v_onset:+d}_curv_{v_curv:+d}_overrule_dwell_{v_dwell:+d}")
                return out
            if abs(d) >= 0.85 and abs(ecue) >= 2.0 and v_env == v_dwell and v_curv == v_dwell:
                out.update(polarity=v_dwell, status="OK", method="extreme_dwell_and_envelope",
                           reason=f"extreme_dwell_{d:+.2f}_env_{ecue:+.2f}_consensus_{v_dwell:+d}")
                return out
            out["reason"] = f"conflict_dwell_{v_dwell:+d}_vs_onset_{v_onset:+d}"
            return out

        # Check if Curvature strongly contradicts Dwell (low/mid frequency only)
        if v_curv != 0 and v_curv != v_dwell and not is_high_freq:
            mild_onset_agrees = (np.isfinite(sc) and np.sign(sc) == v_dwell and abs(sc) >= 0.20) or \
                                (np.isfinite(ec) and np.sign(ec) == v_dwell and abs(ec) >= 0.20)
            if mild_onset_agrees:
                out.update(polarity=v_dwell, status="OK", method="resting_dwell",
                           reason=f"dwell_{d:+.2f}_supported_by_mild_onset")
                return out
            if abs(d) >= 0.80:
                out.update(polarity=v_dwell, status="OK", method="resting_dwell",
                           reason=f"overwhelming_dwell_{d:+.2f}")
                return out
            out["reason"] = f"conflict_dwell_{v_dwell:+d}_vs_curv_{v_curv:+d}"
            return out

        out.update(polarity=v_dwell, status="OK", method="resting_dwell", reason=f"dwell_asymmetry_{d:+.2f}")
        return out

    # Physiological Refinement Rule 3: Mild Dwell + Onset consensus at low/mid frequency (< 2.5 Hz)
    # Mild resting dwell (|d| >= 0.08) and onset kinetics both point in the same direction,
    # with no opposing extremum curvature.
    if not is_high_freq and np.isfinite(d) and abs(d) >= 0.08:
        mild_v_dwell = -1 if d > 0 else 1
        if np.isfinite(sc) and np.sign(sc) == mild_v_dwell and abs(sc) >= 0.20 and \
           np.isfinite(ec) and np.sign(ec) == mild_v_dwell and abs(ec) >= 0.35:
            if not (np.isfinite(cc) and np.sign(-cc) != mild_v_dwell and abs(cc) >= 0.50):
                out.update(polarity=mild_v_dwell, status="OK", method="mild_dwell_and_onset_consensus",
                           reason=f"mild_dwell_{d:+.2f}_onset_agree_{sc:+.2f}")
                return out

    # Physiological Refinement Rule 4: High-frequency kinetic & cycle morphology consensus at 3 Hz
    if is_high_freq:
        cm = _cycle_morphology_cues(finite, fs, tc)
        cycle_dur = cm.get("cycle_dur_ratio", float("nan"))
        cycle_var = cm.get("cycle_var_ratio", float("nan"))
        out["cycle_dur_ratio"] = cycle_dur
        out["cycle_var_ratio"] = cycle_var

        # Rule 4A: Fast onset slope and edge duration supported by derivative velocity asymmetry
        if np.isfinite(sc) and np.isfinite(ec) and np.sign(sc) == np.sign(ec) and \
           abs(sc) >= 0.30 and abs(ec) >= 0.60:
            v_on_hf = 1 if sc > 0 else -1
            if np.sign(vel_log_ratio) == v_on_hf and abs(vel_log_ratio) >= 0.10:
                out.update(polarity=v_on_hf, status="OK", method="high_freq_kinetic_consensus",
                           reason=f"hf_onset_sc_{sc:+.2f}_ec_{ec:+.2f}_vel_{vel_log_ratio:+.2f}")
                return out

        # Rule 4B: Multi-cue kinetic & curvature consensus (+1)
        # Strong onset edge duration (ec >= 0.70), positive velocity asymmetry (vel >= 0.20),
        # and sharper peak curvature than trough (cc <= -0.50).
        if np.isfinite(ec) and ec >= 0.70 and vel_log_ratio >= 0.20 and np.isfinite(cc) and cc <= -0.50:
            out.update(polarity=1, status="OK", method="hf_multicue_consensus",
                       reason=f"hf_multicue_ec_{ec:+.2f}_vel_{vel_log_ratio:+.2f}_curv_{cc:+.2f}")
            return out

        # Rule 4C: Extrema variance asymmetry & upper resting dwell (-1)
        # Troughs vary significantly more than peaks (cycle_var_ratio <= -1.50), resting dwell is at top (d >= 0.45),
        # and troughs are sharper than top dome (cc >= 0.70).
        if np.isfinite(cycle_var) and cycle_var <= -1.50 and np.isfinite(d) and d >= 0.45 and np.isfinite(cc) and cc >= 0.70:
            out.update(polarity=-1, status="OK", method="hf_extrema_variance_asymmetry",
                       reason=f"hf_var_{cycle_var:+.2f}_dwell_{d:+.2f}_curv_{cc:+.2f}")
            return out

        # Rule 4D: Aligned cycle rise/fall duration asymmetry (+1)
        # Cycle rise time is >= 1.6x faster than relaxation decay (cycle_dur_ratio >= 0.50),
        # supported by positive onset slope (sc >= 0.15) and positive velocity ratio (vel >= 0.15).
        if np.isfinite(cycle_dur) and cycle_dur >= 0.50 and np.isfinite(sc) and sc >= 0.15 and vel_log_ratio >= 0.15:
            out.update(polarity=1, status="OK", method="hf_cycle_duration_asymmetry",
                       reason=f"hf_dur_{cycle_dur:+.2f}_sc_{sc:+.2f}_vel_{vel_log_ratio:+.2f}")
            return out

        # Rule 4E: Lower resting dwell with cycle variance corroboration (+1)
        # Cell spends >80% time near bottom rail (d <= -0.60), and peaks vary significantly more than troughs (cycle_var >= 0.80).
        if np.isfinite(d) and d <= -0.60 and np.isfinite(cycle_var) and cycle_var >= 0.80:
            out.update(polarity=1, status="OK", method="hf_dwell_and_variance_consensus",
                       reason=f"hf_dwell_{d:+.2f}_var_{cycle_var:+.2f}")
            return out

    # If high frequency and moderate dwell (0.35 <= |d| < 0.75), accept if corroborated by onset
    if is_high_freq and abs(d) >= DWELL_DECISIVE:
        mild_v_dwell = -1 if d > 0 else 1
        if np.isfinite(sc) and np.sign(sc) == mild_v_dwell and abs(sc) >= 0.20:
            out.update(polarity=mild_v_dwell, status="OK", method="resting_dwell",
                       reason=f"high_freq_dwell_{d:+.2f}_supported_by_onset_{sc:+.2f}")
            return out

    # Case 2: Onset is decisive
    if v_onset != 0:
        out.update(polarity=v_onset, status="OK", method="onset_asymmetry",
                   reason=f"onset_agree_sc_{sc:+.2f}_ec_{ec:+.2f}")
        return out

    # Case 3: Curvature is decisive (low/mid frequency only)
    if v_curv != 0 and not is_high_freq:
        if np.isfinite(d) and abs(d) < 0.05:
            out["reason"] = f"symmetric_dwell_unsupported_curv_{cc:+.2f}"
            return out
        if np.isfinite(d) and np.sign(-d) != v_curv and abs(d) >= 0.20:
            out["reason"] = f"curv_dwell_mild_conflict"
            return out
        out.update(polarity=v_curv, status="OK", method="extremum_curvature",
                   reason=f"curvature_sharpness_{cc:+.2f}")
        return out

    # Case 4: Envelope is decisive ONLY if dwell does not contradict, onset does not oppose, AND not high frequency
    if v_env != 0 and not is_high_freq:
        if np.isfinite(d) and abs(d) >= 0.10 and np.sign(d) == v_env:
            out["reason"] = f"env_contradicts_mild_dwell_{d:+.2f}"
            return out
        if np.isfinite(sc) and np.isfinite(ec) and np.sign(sc) == np.sign(ec) and np.sign(sc) != v_env:
            out["reason"] = f"env_{v_env:+d}_contradicts_unanimous_onset"
            return out
        out.update(polarity=v_env, status="OK", method="envelope_stability",
                   reason=f"envelope_stability_{ecue:+.2f}")
        return out

    out["reason"] = f"indecisive_dwell_{d:+.2f}_onset_{sc:+.2f}_{ec:+.2f}_curv_{cc:+.2f}"
    return out


def determine_raw_polarity(
    sig_raw: np.ndarray,
    fs: float,
    sig_cal: Optional[np.ndarray] = None,
    fs_cal: Optional[float] = None,
    t0_raw: float = 0.0,
    t0_cal: float = 0.0,
) -> Dict:
    """Authoritative polarity decision from the ORIGINAL raw waveform.

    Supports optional auxiliary RatioMetricCalcium for excitation-contraction coupling (ECC)
    timing verification, with seamless fallback to PixelCorrelation morphology.

    Returns dict(polarity, status, method, dwell, slope_cue, edge_cue, curv_cue, env_cue, reason).
    ``polarity`` is -1 / +1, or 0 when the waveform cannot support a decision
    (status == "UNDETERMINED"); in that case no contractions should be reported.

    Decision architecture (two-stage physiological verification):
      Stage 1 (Calcium ECC Timing):
        If synchronized RatioMetricCalcium is available, evaluates causal systolic contraction
        delays across cycles. If one orientation clearly dominates (difference >= 2 matches,
        or sole orientation with causal matches), decides polarity by ECC timing.
      Stage 2 (Morphological Cues):
        If calcium is absent, flat, or ambiguous:
        - Resting dwell asymmetry D (stationary sample occupancy).
        - Contraction-onset asymmetry (slope log-ratio sc, edge duration log-ratio ec).
        - Extremum curvature / sharpness asymmetry curv_cue.
        - Envelope baseline stability asymmetry env_cue.
        If morphology is UNDETERMINED and ECC timing resolves it, ECC timing is adopted.
        Fails safe to UNDETERMINED when evidence is insufficient or contradictory.
    """
    morph = _morphology_polarity(sig_raw, fs)

    # Check if auxiliary calcium signal is provided
    if sig_cal is not None and len(sig_cal) > 0 and np.any(np.isfinite(sig_cal)):
        fs_ca = fs_cal if (fs_cal is not None and np.isfinite(fs_cal) and fs_cal > 0) else fs
        ecc = _calcium_ecc_polarity(sig_raw, fs, sig_cal, fs_ca, t0_raw=t0_raw, t0_cal=t0_cal)

        # Attach calcium diagnostic evidence to output for auditability
        morph.update({
            "ecc_n_ca": ecc.get("n_ca", 0),
            "ecc_m_pos": ecc.get("m_pos", 0),
            "ecc_m_neg": ecc.get("m_neg", 0),
            "ecc_med_pos_ms": ecc.get("med_pos_ms", float("nan")),
            "ecc_med_neg_ms": ecc.get("med_neg_ms", float("nan")),
            "ecc_std_pos_ms": ecc.get("std_pos_ms", float("nan")),
            "ecc_std_neg_ms": ecc.get("std_neg_ms", float("nan")),
        })

        m_pos = ecc.get("m_pos", 0)
        m_neg = ecc.get("m_neg", 0)
        e_pol = ecc.get("polarity", 0)

        # Clear ECC winner criteria
        clear_ecc = False
        if e_pol == 1 and (m_pos >= m_neg + 2 or (m_neg == 0 and m_pos >= 2)):
            clear_ecc = True
        elif e_pol == -1 and (m_neg >= m_pos + 2 or (m_pos == 0 and m_neg >= 2)):
            clear_ecc = True

        if clear_ecc:
            morph.update(polarity=e_pol, status="OK", method="calcium_ecc_timing", reason=ecc["reason"])
            return morph

        if morph["status"] != "OK" and e_pol != 0:
            morph.update(polarity=e_pol, status="OK", method="calcium_ecc_timing",
                         reason=f"ca_resolved_{ecc['reason']}")
            return morph

    return morph



# ---------------------------------------------------------------------------
# Baseline and amplitude waveform
# ---------------------------------------------------------------------------
def estimate_resting_baseline(oriented: np.ndarray, fs: float) -> np.ndarray:
    """Resting baseline of an *oriented* waveform (contractions upward).

    Rolling lower-decile in a centred window, lightly smoothed.  Because contraction
    episodes occupy well under 90 % of any 2 s window, the decile lies on the resting
    state, is insensitive to the contractions themselves and tracks slow drift.
    Returns an array the same length as the input (NaN gaps are bridged by interpolation
    so that the baseline can be drawn continuously; the amplitude stays NaN there).
    """
    u = np.asarray(oriented, dtype=float)
    n = u.size
    base = np.full(n, np.nan)
    if np.isfinite(u).sum() < 5:
        return base
    win = max(5, int(round(BASELINE_WINDOW_S * fs)))
    win = min(win, n)
    q = pd.Series(u).rolling(win, center=True, min_periods=max(5, win // 4)).quantile(BASELINE_QUANTILE)
    q = q.to_numpy()
    idx = np.arange(n)
    ok = np.isfinite(q)
    if ok.sum() < 2:
        return base
    q = np.interp(idx, idx[ok], q[ok])
    sigma = max(1.0, BASELINE_SMOOTH_S * fs)
    base = gaussian_filter1d(q, sigma, mode="nearest")
    return base


def compute_amplitude_waveform(sig_raw: np.ndarray, fs: float, polarity: int) -> Dict[str, np.ndarray]:
    """Return A(t) = polarity * (S_raw - B_raw) and its components (raw data untouched)."""
    s = np.asarray(sig_raw, dtype=float)
    if polarity not in (-1, 1):
        raise ValueError("polarity must be -1 or +1")
    oriented = polarity * s
    b_o = estimate_resting_baseline(oriented, fs)
    b_raw = polarity * b_o                      # baseline in raw units
    amp = polarity * (s - b_raw)                # == oriented - b_o
    return {"amplitude": amp, "baseline_raw": b_raw, "baseline_oriented": b_o}
