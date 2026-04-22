from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

from .config import BeatCounterConfig
from .preprocessing import (
    baseline_shift_percentile,
    compute_global_snr,
    compute_spike_fraction,
    estimate_dominant_period_autocorr,
    estimate_noise_mad,
    moving_average_smooth,
)

def detect_vertical_line_artifacts(sig: np.ndarray, config: BeatCounterConfig) -> Dict:
    s = np.asarray(sig, dtype=float)
    if s.size < 5:
        return {"centers": np.array([], dtype=int), "n_artifacts": 0, "jump_thr": 0.0}
    abs_d1 = np.abs(np.diff(s))
    if abs_d1.size < 3:
        return {"centers": np.array([], dtype=int), "n_artifacts": 0, "jump_thr": 0.0}
    q99 = float(np.quantile(abs_d1, 0.99))
    q999 = float(np.quantile(abs_d1, 0.999))
    thr = float(max(config.vertical_artifact_q99_mult * q99, config.vertical_artifact_q999_mult * q999))
    if thr <= 1e-12:
        return {"centers": np.array([], dtype=int), "n_artifacts": 0, "jump_thr": 0.0}

    c = []
    for i in range(1, s.size - 1):
        left = float(s[i] - s[i - 1])
        right = float(s[i + 1] - s[i])
        if left * right >= 0:
            continue
        if min(abs(left), abs(right)) < thr:
            continue
        if max(abs(left), abs(right)) < config.vertical_artifact_asym_mult * thr:
            continue
        c.append(i)

    dedup = []
    gap = max(1, int(config.vertical_artifact_dedup_gap_samples))
    for x in sorted(c):
        if not dedup or x - dedup[-1] > gap:
            dedup.append(x)

    return {"centers": np.asarray(dedup, dtype=int), "n_artifacts": int(len(dedup)), "jump_thr": thr}

def suppress_vertical_line_artifacts(sig: np.ndarray, centers: np.ndarray, halfwin: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(sig, dtype=float)
    centers = np.asarray(centers, dtype=int)
    if s.size == 0 or centers.size == 0:
        return s.copy(), np.zeros(s.size, dtype=bool)

    out = s.copy()
    mask = np.zeros(s.size, dtype=bool)
    hw = max(0, int(halfwin))
    for c in centers:
        l = max(0, int(c) - hw)
        r = min(s.size - 1, int(c) + hw)
        la = max(0, l - 1)
        ra = min(s.size - 1, r + 1)
        if ra <= la:
            continue
        y0 = float(out[la])
        y1 = float(out[ra])
        span = float(ra - la)
        for j in range(l, r + 1):
            a = float(j - la) / span
            out[j] = (1 - a) * y0 + a * y1
            mask[j] = True
    return out, mask

def detect_discontinuity_artifact_centers(sig: np.ndarray, config: BeatCounterConfig) -> Dict:
    s = np.asarray(sig, dtype=float)
    if s.size < 5:
        return {"centers": np.array([], dtype=int), "jump_thr": 0.0}
    d = np.diff(s)
    abs_d = np.abs(d)
    if abs_d.size < 3:
        return {"centers": np.array([], dtype=int), "jump_thr": 0.0}

    mad_d = max(float(median_abs_deviation(d, scale="normal")), 1e-12)
    q999 = float(np.quantile(abs_d, 0.999))
    jump_thr = float(max(float(config.discontinuity_jump_mad_mult) * mad_d, float(config.discontinuity_jump_q999_mult) * q999))

    centers: List[int] = []

    for i in range(1, len(s) - 1):
        left = float(d[i - 1])
        right = float(d[i])
        bipolar = (left * right < 0) and (max(abs(left), abs(right)) >= jump_thr) and (min(abs(left), abs(right)) >= float(config.discontinuity_bipolar_min_frac) * jump_thr)
        if bipolar:
            centers.append(int(i))

    dedup: List[int] = []
    for c in sorted(centers):
        if not dedup or c - dedup[-1] > 1:
            dedup.append(c)
    return {"centers": np.asarray(dedup, dtype=int), "jump_thr": float(jump_thr)}

def _row_corr_median(rows: List[np.ndarray]) -> float:
    if len(rows) < 2:
        return np.nan
    min_len = min(len(r) for r in rows)
    norm = []
    for r in rows:
        x = np.asarray(r[:min_len], dtype=float)
        x = x - np.median(x)
        sd = float(np.std(x))
        if sd > 1e-12:
            x = x / sd
        norm.append(x)
    corrs = []
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            c = float(np.corrcoef(norm[i], norm[j])[0, 1])
            if np.isfinite(c):
                corrs.append(c)
    return float(np.median(corrs)) if corrs else np.nan

def detect_mixed_direction_flip(sig: np.ndarray, fs: float, config: BeatCounterConfig) -> Dict:
    y = np.asarray(sig, dtype=float)
    if y.size < max(40, int(0.8 * fs)):
        return {"detected": False, "transition": 0.0, "spread": 0.0, "rel_strength": 0.0, "n_pos": 0, "n_neg": 0}
    y = y - float(np.median(y))
    noise = estimate_noise_mad(y)
    prom_thr = max(0.010, 2.8 * float(noise))
    dist = max(1, int(round(0.10 * fs)))

    def _find(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p, props = find_peaks(arr, prominence=prom_thr, distance=dist, width=2)
        if p.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        w = np.asarray(props.get("widths", np.array([], dtype=float)), dtype=float) / max(fs, 1e-9)
        pr = np.asarray(props.get("prominences", np.array([], dtype=float)), dtype=float)
        if w.size != p.size or pr.size != p.size:
            return np.array([], dtype=int), np.array([], dtype=float)
        keep = (w >= 0.014) & (w <= 0.80)
        return np.asarray(p[keep], dtype=int), np.asarray(pr[keep], dtype=float)

    p_pos, prom_pos = _find(y)
    p_neg, prom_neg = _find(-y)
    if p_pos.size < 2 or p_neg.size < 2:
        return {"detected": False, "transition": 0.0, "spread": 0.0, "rel_strength": 0.0, "n_pos": int(p_pos.size), "n_neg": int(p_neg.size)}

    rel_strength = float(min(np.median(prom_pos), np.median(prom_neg)) / max(max(np.median(prom_pos), np.median(prom_neg)), 1e-12))
    mid = int(y.size // 2)
    pos_early = float(np.mean(p_pos < mid))
    pos_late = float(np.mean(p_pos >= mid))
    neg_early = float(np.mean(p_neg < mid))
    neg_late = float(np.mean(p_neg >= mid))
    transition = float(max(pos_early * neg_late, neg_early * pos_late))
    spread = float(abs(np.median(p_pos) - np.median(p_neg)) / max(float(y.size), 1.0))
    detected = bool(
        transition >= float(config.qc_mixed_flip_transition_min)
        and spread >= float(config.qc_mixed_flip_spread_min)
        and rel_strength >= float(config.qc_mixed_flip_rel_strength_min)
    )
    return {
        "detected": detected,
        "transition": transition,
        "spread": spread,
        "rel_strength": rel_strength,
        "n_pos": int(p_pos.size),
        "n_neg": int(p_neg.size),
    }

def compute_sheet_structure_features(df: pd.DataFrame, y_cols: Sequence[str], fs: float, invert_all: bool, orient_meta: Dict, sig: np.ndarray, config: BeatCounterConfig) -> Dict:
    rows = []
    for _, row in df.iterrows():
        tr = row[list(y_cols)].values.astype(float)
        tr = tr[~np.isnan(tr)]
        if tr.size > 0:
            rows.append(-tr if invert_all else tr)
    if not rows:
        return {"n_rows": 0}

    rows_bs = [baseline_shift_percentile(r, p=5.0) for r in rows]
    corr_med = _row_corr_median(rows_bs)
    snr, noise_mad, _ = compute_global_snr(sig)
    _, periodicity = estimate_dominant_period_autocorr(sig, fs)
    spike_fraction = compute_spike_fraction(sig)
    mixed_flip = detect_mixed_direction_flip(sig, fs, config=config)

    strengths = []
    row_per = []
    for r in rows_bs:
        nm = estimate_noise_mad(r)
        sm = moving_average_smooth(r, fs, smooth_ms=14.0)
        p, props = find_peaks(sm, prominence=max(0.75 * noise_mad, 2.2 * nm, 0.006), distance=max(1, int(0.08 * fs)), width=max(2, int(0.01 * fs)))
        if p.size:
            strength = float(np.max(props["prominences"]) / max(nm, 1e-12))
            strengths.append(float(np.clip(strength, 0.0, 250.0)))
        else:
            strengths.append(0.0)
        _, ps = estimate_dominant_period_autocorr(sm, fs)
        row_per.append(float(ps))

    strengths = np.asarray(strengths, dtype=float)
    row_per = np.asarray(row_per, dtype=float)
    split = strengths.size // 2 if strengths.size >= 4 else strengths.size
    if strengths.size >= 4:
        early = float(np.median(strengths[:split]))
        late = float(np.median(strengths[split:]))
        late_per = float(np.median(row_per[split:]))
    else:
        early = float(np.median(strengths))
        late = early
        late_per = float(np.median(row_per))

    return {
        "n_rows": int(len(rows)),
        "corr_med": corr_med,
        "snr": float(snr),
        "periodicity_score": float(periodicity),
        "spike_fraction": float(spike_fraction),
        "early_strength": float(early),
        "late_strength": float(late),
        "late_periodicity": float(late_per),
        "orientation_down_fraction": float(orient_meta.get("down_fraction", np.nan)),
        "orientation_up_fraction": float(orient_meta.get("up_fraction", np.nan)),
        "orientation_confidence": float(orient_meta.get("confidence", np.nan)),
        "orientation_transition_score": float(orient_meta.get("row_direction_transition_score", 0.0)),
        "mixed_direction_flip_detected": bool(mixed_flip.get("detected", False)),
        "mixed_direction_flip_transition": float(mixed_flip.get("transition", 0.0)),
        "mixed_direction_flip_spread": float(mixed_flip.get("spread", 0.0)),
        "mixed_direction_flip_rel_strength": float(mixed_flip.get("rel_strength", 0.0)),
    }

def _evaluate_mixed_orientation_rule(features: Dict, config: BeatCounterConfig) -> Tuple[bool, Dict]:
    down = float(features.get("orientation_down_fraction", np.nan))
    up = float(features.get("orientation_up_fraction", np.nan))
    minor = min(down, up) if np.isfinite(down) and np.isfinite(up) else 0.0
    transition = float(features.get("orientation_transition_score", 0.0))
    periodicity = float(features.get("periodicity_score", 0.0))
    corr = float(features.get("corr_med", np.nan))
    mixed_flip_detected = bool(features.get("mixed_direction_flip_detected", False))

    minor_term = float(np.clip(
        (minor - float(config.qc_mixed_orientation_minor_min))
        / max(1.0 - float(config.qc_mixed_orientation_minor_min), 1e-12),
        0.0,
        1.0,
    ))
    transition_term = float(np.clip(
        (transition - float(config.qc_mixed_orientation_transition_min))
        / max(1.0 - float(config.qc_mixed_orientation_transition_min), 1e-12),
        0.0,
        1.0,
    ))
    periodicity_term = float(np.clip(
        (float(config.qc_mixed_orientation_periodicity_max) - periodicity)
        / max(float(config.qc_mixed_orientation_periodicity_max), 1e-12),
        0.0,
        1.0,
    ))
    corr_term = float(np.clip(
        (float(config.qc_mixed_orientation_corr_max) - corr)
        / max(float(config.qc_mixed_orientation_corr_max), 1e-12),
        0.0,
        1.0,
    )) if np.isfinite(corr) else 1.0

    mixed_score = float(
        float(config.qc_mixed_orientation_weight_minor) * minor_term
        + float(config.qc_mixed_orientation_weight_transition) * transition_term
        + float(config.qc_mixed_orientation_weight_periodicity) * periodicity_term
        + float(config.qc_mixed_orientation_weight_corr) * corr_term
        + (float(config.qc_mixed_orientation_flip_bonus) if mixed_flip_detected else 0.0)
    )

    gate_pass = bool(
        minor >= float(config.qc_mixed_orientation_gate_minor)
        or transition >= float(config.qc_mixed_orientation_gate_transition)
        or mixed_flip_detected
    )
    reject = bool(gate_pass and mixed_score >= float(config.qc_mixed_orientation_reject_score))
    details = {
        "qc_mixed_minor_fraction": float(minor),
        "qc_mixed_transition_score": float(transition),
        "qc_mixed_periodicity_score": float(periodicity),
        "qc_mixed_corr_med": float(corr) if np.isfinite(corr) else np.nan,
        "qc_mixed_minor_term": float(minor_term),
        "qc_mixed_transition_term": float(transition_term),
        "qc_mixed_periodicity_term": float(periodicity_term),
        "qc_mixed_corr_term": float(corr_term),
        "qc_mixed_flip_detected": bool(mixed_flip_detected),
        "qc_mixed_orientation_score": float(mixed_score),
        "qc_mixed_orientation_gate_pass": bool(gate_pass),
        "qc_mixed_orientation_reject": bool(reject),
    }
    return reject, details


def evaluate_segment_qc_with_features(features: Dict, n_vertical_artifacts: int, config: BeatCounterConfig) -> Tuple[bool, str, Dict]:
    f = dict(features or {})
    n_rows = int(features.get("n_rows", 0))
    corr = float(features.get("corr_med", np.nan))
    periodicity = float(features.get("periodicity_score", 0.0))
    snr = float(features.get("snr", 0.0))
    spike = float(features.get("spike_fraction", 0.0))
    if n_rows < int(config.qc_min_rows):
        if (
            n_rows >= 2
            and snr >= 2.0
            and np.isfinite(corr)
            and corr >= 0.95
            and spike <= 0.08
            and n_vertical_artifacts < int(config.vertical_artifact_reject_count)
        ):
            f["qc_reason"] = "pass_short_rows"
            return True, "pass_short_rows", f
        f["qc_reason"] = "too_few_rows"
        return False, "too_few_rows", f

    mixed_reject, mixed_details = _evaluate_mixed_orientation_rule(f, config)
    f.update(mixed_details)
    if mixed_reject:
        f["qc_reason"] = "mixed_orientation_inconsistent_morphology"
        return False, "mixed_orientation_inconsistent_morphology", f

    if n_rows <= 3:
        early = float(features.get("early_strength", 0.0))
        if (
            snr >= 2.0
            and spike <= 0.03
            and early >= 20.0
            and n_vertical_artifacts < int(config.vertical_artifact_reject_count)
        ):
            f["qc_reason"] = "pass_sparse_short_recording"
            return True, "pass_sparse_short_recording", f

    if n_vertical_artifacts >= config.vertical_artifact_reject_count and periodicity < config.vertical_artifact_reject_min_periodicity:
        f["qc_reason"] = "dominant_vertical_line_artifacts"
        return False, "dominant_vertical_line_artifacts", f

    if snr < config.qc_low_snr_threshold and periodicity < config.qc_low_snr_periodicity_min:
        down = float(features.get("orientation_down_fraction", np.nan))
        up = float(features.get("orientation_up_fraction", np.nan))
        allow_low_snr_clean = (
            n_rows >= 12
            and snr >= 2.0
            and spike <= 0.002
            and max(down, up) >= 0.60
        )
        if not allow_low_snr_clean:
            f["qc_reason"] = "low_conf_orientation_low_snr_noise"
            return False, "low_conf_orientation_low_snr_noise", f

    if spike >= config.qc_nonperiodic_spiky_min_spike_fraction and periodicity <= config.qc_nonperiodic_spiky_max_periodicity:
        f["qc_reason"] = "nonperiodic_spiky_noise"
        return False, "nonperiodic_spiky_noise", f

    if n_rows >= config.qc_nonstationary_min_rows:
        early = float(features.get("early_strength", 0.0))
        late = float(features.get("late_strength", 0.0))
        late_per = float(features.get("late_periodicity", 0.0))
        if early >= config.qc_nonstationary_min_early_strength and late <= max(config.qc_nonstationary_min_late_strength, config.qc_nonstationary_drop_ratio * early) and late_per <= config.qc_nonstationary_max_late_periodicity:
            likely_physiologic_pause = (
                periodicity >= float(config.qc_nonstationary_coherent_periodicity_min) * 0.70
                and snr >= float(config.qc_nonstationary_noise_snr_max)
                and spike < float(config.qc_nonstationary_noise_spike_min)
            )
            severe_late_noise = (
                periodicity <= float(config.qc_nonstationary_noise_periodicity_max)
                and snr <= float(config.qc_nonstationary_noise_snr_max)
            ) or (
                spike >= float(config.qc_nonstationary_noise_spike_min)
                and periodicity <= float(config.qc_nonstationary_max_late_periodicity)
            )
            if severe_late_noise and not likely_physiologic_pause:
                f["qc_reason"] = "nonstationary_late_noise"
                return False, "nonstationary_late_noise", f

    f["qc_reason"] = "pass"
    return True, "pass", f


def evaluate_segment_qc(features: Dict, n_vertical_artifacts: int, config: BeatCounterConfig) -> Tuple[bool, str]:
    qc_pass, qc_reason, enriched_features = evaluate_segment_qc_with_features(features, n_vertical_artifacts, config)
    features.clear()
    features.update(enriched_features)
    return qc_pass, qc_reason
