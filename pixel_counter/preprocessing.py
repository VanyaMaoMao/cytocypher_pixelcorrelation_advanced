from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter
from scipy.signal import correlate, find_peaks, peak_prominences
from scipy.stats import median_abs_deviation

from .config import BeatCounterConfig

def baseline_shift_percentile(sig: np.ndarray, p: float = 5.0) -> np.ndarray:
    return np.asarray(sig, dtype=float) - float(np.percentile(sig, p))

def _rolling_quantile_trend(sig: np.ndarray, fs: float, window_s: float, quantile: float) -> np.ndarray:
    s = np.asarray(sig, dtype=float)
    if s.size < 30:
        return np.full(s.size, np.nan, dtype=float)

    win = max(31, int(round(float(window_s) * fs)))
    if win % 2 == 0:
        win += 1
    if win >= s.size:
        win = max(31, int(round(0.75 * s.size)))
        if win % 2 == 0:
            win -= 1

    # Percentile filter is significantly faster than pandas rolling quantile
    # for dense numeric arrays while preserving percentile-based trend behavior.
    q = float(np.clip(quantile, 0.01, 0.99))
    pct = 100.0 * q

    s_work = s.copy()
    if not np.isfinite(s_work).all():
        idx = np.arange(s_work.size, dtype=float)
        ok = np.isfinite(s_work)
        n_ok = int(np.sum(ok))
        if n_ok >= 2:
            s_work = np.interp(idx, idx[ok], s_work[ok])
        elif n_ok == 1:
            s_work = np.full(s_work.size, float(s_work[ok][0]), dtype=float)
        else:
            return np.full(s.size, np.nan, dtype=float)

    trend = percentile_filter(s_work, percentile=pct, size=win, mode="reflect")
    if not np.isfinite(trend).any():
        return trend

    idx = np.arange(s.size, dtype=float)
    ok = np.isfinite(trend)
    if int(np.sum(ok)) >= 2:
        trend = np.interp(idx, idx[ok], trend[ok])
    else:
        fill = float(trend[ok][0]) if int(np.sum(ok)) == 1 else float(np.median(s))
        trend = np.full(s.size, fill, dtype=float)
    return trend

def normalize_slow_trend(
    sig: np.ndarray,
    fs: float,
    config: BeatCounterConfig,
    *,
    strength: float,
    min_ratio: float,
    row_corr: float = np.nan,
    force: bool = False,
) -> Tuple[np.ndarray, Dict]:
    s = np.asarray(sig, dtype=float)
    meta = {
        "applied": False,
        "strength": float(strength),
        "window_s": float(config.drift_window_s),
        "quantile": float(config.drift_quantile),
        "drift_detected": False,
        "drift_ratio": 0.0,
        "drift_span": 0.0,
        "signal_span": 0.0,
        "trend_time_corr": np.nan,
        "row_corr": float(row_corr) if np.isfinite(row_corr) else np.nan,
    }
    if s.size < 30 or float(strength) <= 0:
        return s.copy(), meta

    trend = _rolling_quantile_trend(s, fs, config.drift_window_s, config.drift_quantile)
    if not np.isfinite(trend).any():
        return s.copy(), meta

    drift_span = float(np.percentile(trend, 95) - np.percentile(trend, 5))
    signal_span = float(np.percentile(s, 95) - np.percentile(s, 5))
    drift_ratio = float(drift_span / max(signal_span, 1e-12))
    trend_ok = np.isfinite(trend)
    if int(np.sum(trend_ok)) >= 4:
        ti = np.arange(s.size, dtype=float)[trend_ok]
        trend_corr = float(np.corrcoef(ti, trend[trend_ok])[0, 1])
    else:
        trend_corr = np.nan
    meta.update({
        "drift_ratio": drift_ratio,
        "drift_span": drift_span,
        "signal_span": signal_span,
        "trend_time_corr": trend_corr,
    })

    monotonic_ok = np.isfinite(trend_corr) and abs(float(trend_corr)) >= 0.30
    strong_drift = drift_ratio >= (1.35 * float(min_ratio))
    drift_detected = bool(drift_ratio >= float(min_ratio) and (monotonic_ok or strong_drift))
    meta["drift_detected"] = drift_detected

    if not force and not drift_detected:
        return s.copy(), meta
    if (
        not force
        and np.isfinite(row_corr)
        and float(row_corr) > float(config.drift_max_row_corr_for_apply)
        and not strong_drift
    ):
        return s.copy(), meta

    trend_centered = trend - float(np.median(trend))
    out = s - float(strength) * trend_centered
    out = baseline_shift_percentile(out, p=5.0)
    meta["applied"] = True
    return out, meta

def estimate_noise_mad(sig: np.ndarray) -> float:
    s = np.asarray(sig, dtype=float)
    if s.size == 0:
        return 1e-12
    quiet = s[s <= np.percentile(s, 30)]
    if quiet.size < 5:
        quiet = s
    return max(float(median_abs_deviation(quiet, scale="normal")), 1e-12)

def moving_average_smooth(sig: np.ndarray, fs: float, smooth_ms: float = 15.0) -> np.ndarray:
    s = np.asarray(sig, dtype=float)
    k = max(3, int(round((smooth_ms / 1000.0) * fs)))
    if k % 2 == 0:
        k += 1
    if s.size < k:
        return s.copy()
    return np.convolve(s, np.ones(k) / k, mode="same")

def compute_global_snr(sig: np.ndarray) -> Tuple[float, float, float]:
    signal_mad = float(median_abs_deviation(sig, scale="normal"))
    noise_mad = estimate_noise_mad(sig)
    return float(signal_mad / max(noise_mad, 1e-12)), float(noise_mad), float(signal_mad)

def estimate_dominant_period_autocorr(sig: np.ndarray, fs: float) -> Tuple[float, float]:
    s = np.asarray(sig, dtype=float)
    if s.size < int(0.5 * fs):
        return np.nan, 0.0
    x = s - np.median(s)
    sd = float(np.std(x))
    if sd < 1e-12:
        return np.nan, 0.0
    x = x / sd
    acf = correlate(x, x, mode="full", method="fft")
    acf = acf[len(acf) // 2 :]
    acf = acf / max(float(acf[0]), 1e-12)
    lo = max(2, int(0.15 * fs))
    hi = min(len(acf) - 1, int(2.0 * fs))
    if hi <= lo:
        return np.nan, 0.0
    pk, props = find_peaks(acf[lo : hi + 1], prominence=0.02)
    if pk.size == 0:
        return np.nan, 0.0
    best = int(pk[np.argmax(props["prominences"])]) + lo
    return float(best / fs), float(acf[best])

def compute_spike_fraction(sig: np.ndarray, z_thr: float = 8.0) -> float:
    d = np.diff(np.asarray(sig, dtype=float))
    if d.size == 0:
        return 0.0
    med = float(np.median(d))
    mad = max(float(median_abs_deviation(d, scale="normal")), 1e-12)
    z = np.abs((d - med) / mad)
    return float(np.mean(z > z_thr))

def _row_dominant_direction(trace: np.ndarray) -> str:
    x = np.asarray(trace, dtype=float) - np.median(trace)
    if x.size == 0 or np.max(np.abs(x)) < 1e-12:
        return "flat"
    return "down" if abs(np.min(x)) > abs(np.max(x)) else "up"

def choose_orientation_make_peaks_positive(
    sig: np.ndarray,
    fs: float,
    edge_s: float,
    min_sep_s: float,
    n_events: int,
    smooth_ms: float,
    conf_min: float,
) -> Tuple[np.ndarray, Dict]:
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return sig.copy(), {"invert": False, "method": "empty", "vote_conf": np.nan}

    edge_n = int(edge_s * fs)
    if sig.size < 2 * edge_n + 50:
        edge_n = max(0, min(edge_n, sig.size // 10))
    core = sig[edge_n : sig.size - edge_n] if (sig.size - 2 * edge_n) > 50 else sig.copy()
    x = core - np.median(core)

    win_s = float(np.clip(sig.size / max(fs, 1e-9) * 0.35, 1.5, 8.0))
    trend = _rolling_quantile_trend(x, fs, window_s=win_s, quantile=0.20)
    if np.isfinite(trend).any():
        x = x - (trend - float(np.median(trend[np.isfinite(trend)])))

    k = max(5, int(round((smooth_ms / 1000.0) * fs)))
    if k % 2 == 0:
        k += 1
    xs = np.convolve(x, np.ones(k) / k, mode="same") if k > 3 and x.size >= k else x

    def polarity_score(y: np.ndarray) -> Tuple[float, Dict]:
        y = np.asarray(y, dtype=float)
        y = y - np.median(y)
        noise = estimate_noise_mad(y)
        min_dist = max(1, int(min_sep_s * fs))
        pks, props = find_peaks(
            y,
            prominence=max(0.002, 2.7 * noise),
            distance=min_dist,
            width=2,
        )
        if pks.size == 0:
            return -np.inf, {"n_peaks": 0}

        p_prom = np.asarray(props.get("prominences", np.array([], dtype=float)), dtype=float)
        p_w = np.asarray(props.get("widths", np.array([], dtype=float)), dtype=float) / max(fs, 1e-9)
        if p_prom.size == 0:
            return -np.inf, {"n_peaks": 0}

        width_ok = (p_w >= 0.008) & (p_w <= 0.90)
        if int(np.sum(width_ok)) >= 3:
            pks = pks[width_ok]
            p_prom = p_prom[width_ok]
            p_w = p_w[width_ok]

        if pks.size < 3:
            return -np.inf, {"n_peaks": int(pks.size)}

        duration_s = max(float(y.size / max(fs, 1e-9)), 1e-9)
        peak_rate_hz = float(pks.size / duration_s)
        top = np.sort(p_prom)[-min(int(n_events), int(p_prom.size)) :]
        prom_snr = float(np.median(top) / max(noise, 1e-12))
        med_w = float(np.median(p_w)) if p_w.size else 0.0
        ibi = np.diff(pks) / max(fs, 1e-9)
        ibi_cv = float(np.std(ibi) / max(np.mean(ibi), 1e-12)) if ibi.size >= 2 else 1.0
        _, periodicity = estimate_dominant_period_autocorr(y, fs)
        periodicity = float(periodicity) if np.isfinite(periodicity) else 0.0

        score = float(
            1.4 * np.log1p(max(prom_snr, 0.0))
            + 0.9 * periodicity
            + 0.5 * min(max(med_w / 0.22, 0.0), 2.0)
            - 1.6 * max(ibi_cv, 0.0)
            - 0.25 * max(0.0, peak_rate_hz - 3.8)
        )
        return score, {
            "n_peaks": int(pks.size),
            "peak_rate_hz": float(peak_rate_hz),
            "prom_snr": float(prom_snr),
            "med_width_s": float(med_w),
            "ibi_cv": float(ibi_cv),
            "periodicity": float(periodicity),
        }

    score_keep, feat_keep = polarity_score(xs)
    score_inv, feat_inv = polarity_score(-xs)
    score_delta = float(score_inv - score_keep)
    score_gap = float(abs(score_delta))
    meta = {
        "invert": None,
        "method": None,
        "vote_conf": np.nan,
        "orientation_conf": np.nan,
        "score_keep": float(score_keep) if np.isfinite(score_keep) else -np.inf,
        "score_invert": float(score_inv) if np.isfinite(score_inv) else -np.inf,
        "score_delta": float(score_delta),
        "score_gap": float(score_gap),
        "n_pos": int(feat_keep.get("n_peaks", 0)),
        "n_neg": int(feat_inv.get("n_peaks", 0)),
        "features_keep": feat_keep,
        "features_invert": feat_inv,
    }

    abs_x = np.abs(xs)
    vote_inv = None
    vote_conf = np.nan
    pks, _ = find_peaks(abs_x, distance=max(1, int(min_sep_s * fs)))
    if pks.size >= 3:
        top = pks[np.argsort(abs_x[pks])[::-1][: min(n_events, pks.size)]]
        vals = xs[top]
        votes = np.sign(vals)
        votes = votes[votes != 0]
        if votes.size >= 3:
            med = np.sign(np.median(votes))
            vote_conf = float(np.mean(votes == med))
            vote_inv = bool(med < 0)

    margin = 0.12
    conf = float(np.clip(score_gap / 0.8, 0.0, 1.0))
    meta["orientation_conf"] = conf
    if np.isfinite(vote_conf):
        meta["vote_conf"] = float(vote_conf)

    if (np.isfinite(score_keep) or np.isfinite(score_inv)) and score_gap >= margin:
        invert = bool(score_inv > score_keep)
        meta.update({"invert": invert, "method": "morphology_score"})
        return (-sig if invert else sig), meta

    if vote_inv is not None and np.isfinite(vote_conf) and vote_conf >= conf_min:
        meta.update({"invert": bool(vote_inv), "method": "vote_tiebreak"})
        return (-sig if vote_inv else sig), meta

    if np.isfinite(score_keep) or np.isfinite(score_inv):
        invert = bool(score_inv > score_keep)
        meta.update({"invert": invert, "method": "morphology_fallback"})
        return (-sig if invert else sig), meta

    invert = bool(vote_inv) if vote_inv is not None else False
    meta.update({"invert": invert, "method": "vote_fallback"})
    return (-sig if invert else sig), meta

def build_concatenated_signal(
    df: pd.DataFrame,
    y_cols: List[str],
    fs: float,
    config: BeatCounterConfig,
    force_invert: Optional[bool] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, float, int]], Dict]:
    row_entries: List[Dict] = []
    for ridx, row in df.iterrows():
        tr = row[y_cols].values.astype(float)
        tr = tr[~np.isnan(tr)]
        if tr.size == 0:
            continue
        begin = pd.to_numeric(row.get("Begin", np.nan), errors="coerce")
        end = pd.to_numeric(row.get("End", np.nan), errors="coerce")
        row_entries.append(
            {
                "row_idx": int(ridx),
                "tid": int(row["_tn"]) if "_tn" in row else len(row_entries),
                "trace_raw": tr,
                "begin": float(begin) if np.isfinite(begin) else np.nan,
                "end": float(end) if np.isfinite(end) else np.nan,
            }
        )

    if not row_entries:
        meta = {"invert": True if force_invert is None else bool(force_invert), "method": "empty", "confidence": np.nan, "down_fraction": np.nan, "up_fraction": np.nan}
        return np.array([], dtype=float), [], meta

    dirs = [_row_dominant_direction(x["trace_raw"]) for x in row_entries]
    n_down = int(sum(d == "down" for d in dirs))
    n_up = int(sum(d == "up" for d in dirs))
    n_dir = max(1, n_down + n_up)
    down_frac = float(n_down / n_dir)
    up_frac = float(n_up / n_dir)
    dir_seq = [d for d in dirs if d in {"down", "up"}]
    switch_count = int(sum(1 for i in range(1, len(dir_seq)) if dir_seq[i] != dir_seq[i - 1])) if len(dir_seq) >= 2 else 0
    if len(dir_seq) >= 4:
        half = len(dir_seq) // 2
        first = dir_seq[:half]
        second = dir_seq[half:]
        f_down = float(np.mean([1.0 if d == "down" else 0.0 for d in first])) if first else 0.0
        s_down = float(np.mean([1.0 if d == "down" else 0.0 for d in second])) if second else 0.0
        transition_score = float(max(f_down * (1.0 - s_down), (1.0 - f_down) * s_down))
    else:
        transition_score = 0.0

    if force_invert is None:
        sample = np.concatenate([x["trace_raw"] for x in row_entries])
        _, meta = choose_orientation_make_peaks_positive(
            sample,
            fs,
            edge_s=config.orient_edge_s,
            min_sep_s=config.orient_min_sep_s,
            n_events=config.orient_n_events,
            smooth_ms=config.orient_smooth_ms,
            conf_min=config.orient_conf_min,
        )
        invert_all = bool(meta.get("invert", True))
        score_gap = float(meta.get("score_gap", np.nan))
        weak_decision = bool(meta.get("method") in {"vote_fallback", "morphology_fallback", "vote_tiebreak"}) or (
            np.isfinite(score_gap) and score_gap < 0.12
        )
        row_prior_invert = bool(down_frac >= up_frac)
        if weak_decision and max(down_frac, up_frac) >= 0.80:
            invert_all = row_prior_invert
            meta["method"] = "row_direction_prior"
            meta["vote_conf"] = np.nan
    else:
        invert_all = bool(force_invert)
        meta = {"invert": invert_all, "method": "forced", "vote_conf": np.nan}

    confidence = max(down_frac, up_frac)
    if np.isfinite(meta.get("vote_conf", np.nan)):
        confidence = max(confidence, float(meta["vote_conf"]))
    meta.update(
        {
            "invert": invert_all,
            "confidence": float(confidence),
            "down_fraction": down_frac,
            "up_fraction": up_frac,
            "row_direction_switches": int(switch_count),
            "row_direction_transition_score": float(transition_score),
            "row_direction_count": int(len(dir_seq)),
        }
    )

    for entry in row_entries:
        entry["trace"] = (-entry["trace_raw"] if invert_all else entry["trace_raw"]).astype(float)

    begin_vals = np.array([entry["begin"] for entry in row_entries], dtype=float)
    use_begin_end = bool(np.isfinite(begin_vals).all())
    if use_begin_end:
        global_begin = float(np.min(begin_vals))
        for entry in row_entries:
            start_idx = int(round((float(entry["begin"]) - global_begin) * fs))
            entry["start_idx"] = start_idx
            entry["end_idx"] = int(start_idx + len(entry["trace"]))
    else:
        y_offsets = []
        for c in y_cols:
            m = re.match(r"^y\s+(-?\d+)$", str(c))
            if m:
                y_offsets.append(int(m.group(1)))
        overlap = int(max(0, -min(y_offsets))) if y_offsets else 0
        start_idx = 0
        for entry in row_entries:
            entry["start_idx"] = int(start_idx)
            entry["end_idx"] = int(start_idx + len(entry["trace"]))
            start_idx += max(1, int(len(entry["trace"]) - overlap))
        global_begin = 0.0

    row_entries.sort(key=lambda x: (int(x["start_idx"]), int(x["tid"])))

    overlap_samples: List[int] = []
    partition_bounds: List[Tuple[int, int, int]] = []
    for i, entry in enumerate(row_entries):
        start_idx = int(entry["start_idx"])
        end_idx = int(entry["end_idx"])
        if i == 0:
            left_global = start_idx
        else:
            prev_end = int(row_entries[i - 1]["end_idx"])
            left_global = int(round((prev_end + start_idx) / 2.0))
            overlap_samples.append(max(0, prev_end - start_idx))

        if i == len(row_entries) - 1:
            right_global = end_idx
        else:
            next_start = int(row_entries[i + 1]["start_idx"])
            right_global = int(round((end_idx + next_start) / 2.0))

        left_global = max(left_global, start_idx)
        right_global = min(right_global, end_idx)
        if right_global <= left_global:
            left_global = start_idx
            right_global = end_idx

        partition_bounds.append((int(entry["tid"]), int(left_global), int(right_global)))

    global_start_idx = int(min(int(entry["start_idx"]) for entry in row_entries))
    global_end_idx = int(max(int(entry["end_idx"]) for entry in row_entries))
    n_global = max(0, global_end_idx - global_start_idx)
    if n_global <= 0:
        meta.update({"stitch_mode": "empty_global_window"})
        return np.array([], dtype=float), [], meta

    accum = np.zeros(n_global, dtype=float)
    weights = np.zeros(n_global, dtype=float)
    row_offsets: List[float] = []

    for entry in row_entries:
        s_abs = int(entry["start_idx"])
        e_abs = int(entry["end_idx"])
        s = int(s_abs - global_start_idx)
        e = int(e_abs - global_start_idx)
        tr = np.asarray(entry["trace"], dtype=float).copy()
        if tr.size == 0 or e <= s:
            row_offsets.append(0.0)
            continue

        overlap_mask = weights[s:e] > 0
        if int(np.sum(overlap_mask)) >= 5:
            base_overlap = accum[s:e][overlap_mask] / np.maximum(weights[s:e][overlap_mask], 1e-12)
            delta = float(np.median(base_overlap - tr[overlap_mask]))
            tr = tr + delta
            row_offsets.append(delta)
        else:
            row_offsets.append(0.0)

        accum[s:e] += tr
        weights[s:e] += 1.0

    valid = weights > 0
    if not np.any(valid):
        meta.update({"stitch_mode": "empty_post_blending"})
        return np.array([], dtype=float), [], meta

    first_valid = int(np.argmax(valid))
    last_valid = int(len(valid) - np.argmax(valid[::-1]))
    blended = np.full(last_valid - first_valid, np.nan, dtype=float)
    w_block = weights[first_valid:last_valid]
    a_block = accum[first_valid:last_valid]
    ok = w_block > 0
    blended[ok] = a_block[ok] / np.maximum(w_block[ok], 1e-12)

    gap_mask = ~np.isfinite(blended)
    internal_gap_count = int(np.sum(gap_mask))
    stitched_gap_ranges_samples: List[List[int]] = []
    if internal_gap_count > 0:
        gap_idx = np.where(gap_mask)[0]
        run_start = int(gap_idx[0])
        run_prev = int(gap_idx[0])
        for gi in gap_idx[1:]:
            gi_i = int(gi)
            if gi_i != (run_prev + 1):
                stitched_gap_ranges_samples.append([int(run_start), int(run_prev + 1)])
                run_start = gi_i
            run_prev = gi_i
        stitched_gap_ranges_samples.append([int(run_start), int(run_prev + 1)])
    if internal_gap_count > 0:
        idx = np.arange(blended.size, dtype=float)
        valid_idx = np.where(np.isfinite(blended))[0]
        if valid_idx.size >= 2:
            blended = np.interp(idx, valid_idx.astype(float), blended[valid_idx].astype(float))
        else:
            blended = np.nan_to_num(blended, nan=0.0)

    stitched = baseline_shift_percentile(blended, p=5.0)

    seg_meta: List[Tuple[int, int, int, float, int]] = []
    for tid, l_abs, r_abs in partition_bounds:
        l = int(l_abs - global_start_idx - first_valid)
        r = int(r_abs - global_start_idx - first_valid)
        l = max(0, l)
        r = min(stitched.size, r)
        if r <= l:
            continue
        seg = stitched[l:r]
        nm = estimate_noise_mad(seg)
        seg_meta.append((int(tid), int(l), int(r), float(nm), int(r - l)))

    median_overlap = float(np.median(overlap_samples)) if overlap_samples else 0.0
    offset_abs = np.abs(np.asarray(row_offsets, dtype=float))
    median_offset = float(np.median(offset_abs)) if offset_abs.size else 0.0
    max_offset = float(np.max(offset_abs)) if offset_abs.size else 0.0
    time_origin_s = float(global_begin + (first_valid / fs))
    meta.update(
        {
            "stitch_mode": "overlap_weighted_blend",
            "stitch_used_begin_end": bool(use_begin_end),
            "stitch_global_begin": float(global_begin),
            "stitch_median_overlap_samples": float(median_overlap),
            "stitch_rows": int(len(row_entries)),
            "stitch_internal_gap_samples": int(internal_gap_count),
            "stitched_gap_ranges_samples": stitched_gap_ranges_samples,
            "stitch_median_row_offset": float(median_offset),
            "stitch_max_row_offset": float(max_offset),
            "time_origin_s": float(time_origin_s),
        }
    )
    return stitched, seg_meta, meta

def build_transient_id_vector(sig_len: int, seg_meta: List[Tuple[int, int, int, float, int]]) -> np.ndarray:
    out = np.empty(sig_len, dtype=int)
    for tid, s, e, _, _ in seg_meta:
        out[s:e] = int(tid)
    return out

def build_noise_vector(sig_len: int, seg_meta: List[Tuple[int, int, int, float, int]]) -> np.ndarray:
    out = np.zeros(sig_len, dtype=float)
    for _, s, e, nm, _ in seg_meta:
        out[s:e] = float(nm)
    return out
