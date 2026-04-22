from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, replace
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import median_abs_deviation

from .config import AFCReviewConfig, BeatCounterConfig
from .io_utils import load_cytocypher_excel
from .plotting import plot_events
from .preprocessing import (
    build_concatenated_signal,
    build_transient_id_vector,
    compute_global_snr,
    normalize_slow_trend,
)
from .qc import (
    compute_sheet_structure_features,
    detect_discontinuity_artifact_centers,
    detect_vertical_line_artifacts,
    evaluate_segment_qc,
    suppress_vertical_line_artifacts,
)
from .results import AFCEvent, AFCSegmentReviewDecision, AFCSegmentReviewItem, AFCReviewDecision, AFCReviewItem

def detect_raw_peaks(sig: np.ndarray, fs: float, config: BeatCounterConfig) -> Tuple[np.ndarray, Dict]:
    min_dist = max(1, int(config.min_peak_distance_s * fs))
    pks, props = find_peaks(sig, prominence=config.prom0, distance=min_dist, width=2)
    if pks.size == 0:
        return pks, {}
    w = props["widths"] / fs
    pr = props["prominences"]
    m = (w >= config.min_width_s) & (w <= config.max_width_s)
    return pks[m], {"proms": pr[m], "widths_s": w[m]}

def detect_raw_peaks_transientwise(sig: np.ndarray, seg_meta: List[Tuple[int, int, int, float, int]], fs: float, config: BeatCounterConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_peaks: List[int] = []
    all_proms: List[float] = []
    all_widths: List[float] = []
    all_tids: List[int] = []

    for tid, s, e, _, _ in seg_meta:
        s = int(s)
        e = int(e)
        if e - s < 5:
            continue
        pks_local, props = detect_raw_peaks(np.asarray(sig[s:e], dtype=float), fs, config)
        if pks_local.size == 0:
            continue
        all_peaks.extend((pks_local + s).astype(int).tolist())
        all_proms.extend(np.asarray(props["proms"], dtype=float).tolist())
        all_widths.extend(np.asarray(props["widths_s"], dtype=float).tolist())
        all_tids.extend([int(tid)] * int(pks_local.size))

    if not all_peaks:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)

    order = np.argsort(np.asarray(all_peaks, dtype=int))
    peaks = np.asarray(all_peaks, dtype=int)[order]
    proms = np.asarray(all_proms, dtype=float)[order]
    widths = np.asarray(all_widths, dtype=float)[order]
    tids = np.asarray(all_tids, dtype=int)[order]
    return peaks, proms, widths, tids

def filter_needle_spike_peaks(sig: np.ndarray, peaks: np.ndarray, proms: np.ndarray, widths_s: np.ndarray, config: BeatCounterConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if peaks.size == 0:
        return peaks, proms, widths_s, 0
    s = np.asarray(sig, dtype=float)
    d = np.diff(s)
    jump_mad = max(float(median_abs_deviation(d, scale="normal")), 1e-12) if d.size else 1e-12
    keep = np.ones(peaks.size, dtype=bool)
    removed = 0
    for i, pk in enumerate(peaks):
        pk = int(pk)
        if pk <= 0 or pk >= len(s) - 1:
            continue
        jump_l = abs(float(s[pk] - s[pk - 1]))
        jump_r = abs(float(s[pk + 1] - s[pk]))
        jump = max(jump_l, jump_r)
        ratio = jump / (float(proms[i]) + 1e-12)
        local_curv = abs(float(s[pk + 1] - 2.0 * s[pk] + s[pk - 1]))
        narrow = float(widths_s[i]) <= float(config.main_spike_max_width_s)
        jump_abs_thr = max(float(config.main_spike_min_abs_jump), 5.0 * jump_mad)
        needle_like = (
            ratio >= float(config.main_spike_max_jump_prom_ratio)
            and jump >= jump_abs_thr
            and local_curv >= float(config.main_spike_curvature_mad_mult) * jump_mad
        )
        if narrow and needle_like:
            keep[i] = False
            removed += 1
    return peaks[keep], proms[keep], widths_s[keep], removed

def compute_prominence_thresholds(proms: np.ndarray, config: BeatCounterConfig) -> Tuple[float, float, Dict]:
    p95 = float(np.quantile(proms, 0.95))
    strong = float(max(config.min_main_prom, 0.5 * p95))
    p50 = float(np.quantile(proms, 0.50))
    small = proms[proms <= p50]
    if small.size < 5:
        small = proms
    med = float(np.median(small))
    mad = float(median_abs_deviation(small, scale="normal")) if small.size > 2 else float(np.median(np.abs(small - med)))
    weak = float(max(config.min_secondary_prom, med + config.sensitivity * mad))
    if weak >= strong:
        weak = float(0.6 * strong)
    return strong, weak, {"p95": p95, "p50": p50, "small_med": med, "small_mad": mad}


def _infer_segment_index_from_name(name: Optional[str]) -> int:
    if name is None:
        return -1
    m = re.search(r"Segment\s+(\d+)", str(name))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def _build_peak_debug_rows(
    *,
    segment_name: str,
    segment_index: int,
    time: np.ndarray,
    sig: np.ndarray,
    raw_peaks_all: np.ndarray,
    raw_proms_all: np.ndarray,
    raw_widths_all: np.ndarray,
    raw_tids_all: np.ndarray,
    raw_survivors: Set[int],
    main_candidates: Set[int],
    main_after_dedup: Set[int],
    main_after_short_gap: Set[int],
    main_after_local: Set[int],
    main_after_interbeat: Set[int],
    final_main: Set[int],
    rescue: Set[int],
    removed_by_rescue: Set[int],
    promoted_gap: Set[int],
    promoted_transient: Set[int],
    rejected_in_stitched_gap: Set[int],
    strong_thr: float,
) -> List[Dict]:
    rows: List[Dict] = []
    raw_peaks_all = np.asarray(raw_peaks_all, dtype=int)
    raw_proms_all = np.asarray(raw_proms_all, dtype=float)
    raw_widths_all = np.asarray(raw_widths_all, dtype=float)
    raw_tids_all = np.asarray(raw_tids_all, dtype=int)
    time = np.asarray(time, dtype=float)
    sig = np.asarray(sig, dtype=float)

    for i, (pk, pr, wd, tid) in enumerate(zip(raw_peaks_all, raw_proms_all, raw_widths_all, raw_tids_all), start=1):
        pk_i = int(pk)
        pr_f = float(pr)
        wd_f = float(wd)
        tid_i = int(tid)
        t_s = float(time[pk_i]) if 0 <= pk_i < time.size else np.nan
        amp = float(sig[pk_i]) if 0 <= pk_i < sig.size else np.nan

        stage_first_seen = "raw_detected"
        if pk_i in main_candidates:
            if np.isfinite(strong_thr) and pr_f >= float(strong_thr):
                stage_first_seen = "above_strong_threshold"
            else:
                stage_first_seen = "above_weak_threshold"
        elif pk_i in promoted_gap:
            stage_first_seen = "gap_fill_promotion"
        elif pk_i in promoted_transient:
            stage_first_seen = "transient_coherence_promotion"

        final_label = "dropped"

        if pk_i in final_main:
            final_label = "main"
        elif pk_i in rescue:
            final_label = "rescue"
        elif pk_i not in raw_survivors:
            final_label = "dropped"
        elif pk_i not in main_candidates and pk_i not in promoted_gap and pk_i not in promoted_transient:
            final_label = "raw_only"
        else:
            final_label = "dropped"

        rejection_reason = ""
        if final_label not in {"main", "rescue"}:
            if pk_i in rejected_in_stitched_gap:
                rejection_reason = "ineligible_interpolated_stitch_gap"
            elif pk_i not in raw_survivors:
                rejection_reason = "filtered_as_artifact_or_spike"
            elif pk_i not in main_candidates and pk_i not in promoted_gap and pk_i not in promoted_transient:
                rejection_reason = "below_strong_threshold"
            elif pk_i not in main_after_dedup:
                rejection_reason = "pruned_as_close_duplicate"
            elif pk_i not in main_after_short_gap:
                rejection_reason = "failed_short_gap_filter"
            elif pk_i not in main_after_local:
                rejection_reason = "failed_local_weak_filter"
            elif pk_i not in main_after_interbeat:
                rejection_reason = "filtered_as_tiny_between_beats"
            elif pk_i in removed_by_rescue:
                rejection_reason = "replaced_by_rescue"
            else:
                rejection_reason = "dropped_after_rescue_stage"

        rows.append(
            {
                "segment_name": str(segment_name),
                "segment_index": int(segment_index),
                "peak_index_raw": int(i),
                "time_s": float(t_s),
                "amplitude": float(amp),
                "prominence": float(pr_f),
                "width_s": float(wd_f),
                "transient_index": int(tid_i),
                "stage_first_seen": str(stage_first_seen),
                "survived_raw_filter": bool(pk_i in raw_survivors),
                "survived_main_candidate_stage": bool(pk_i in main_candidates),
                "survived_dedup_stage": bool(pk_i in main_after_dedup),
                "survived_short_gap_prune": bool(pk_i in main_after_short_gap),
                "survived_local_weak_prune": bool(pk_i in main_after_local),
                "survived_interbeat_tiny_filter": bool(pk_i in main_after_interbeat),
                "survived_rescue_stage": bool(pk_i in final_main or pk_i in rescue),
                "final_label": str(final_label),
                "rejection_reason": str(rejection_reason),
                "in_stitched_gap_region": bool(pk_i in rejected_in_stitched_gap),
                "notes": "",
            }
        )
    return rows



def dedup_peaks_by_prom(peaks: Sequence[int], prom_map: Dict[int, float], fs: float, min_sep_s: float) -> List[int]:
    if len(peaks) <= 1:
        return sorted(int(x) for x in peaks)
    p = np.array(sorted(int(x) for x in peaks), dtype=int)
    sep = max(1, int(round(min_sep_s * fs)))
    keep: List[int] = []
    i = 0
    while i < len(p):
        g = [int(p[i])]
        j = i + 1
        while j < len(p) and (p[j] - p[j - 1]) <= sep:
            g.append(int(p[j]))
            j += 1
        keep.append(max(g, key=lambda z: prom_map.get(z, 0.0)))
        i = j
    return sorted(set(keep))

def deduplicate_main_candidates(
    *,
    sig: np.ndarray,
    peaks: np.ndarray,
    proms: np.ndarray,
    widths_s: np.ndarray,
    tids: np.ndarray,
    fs: float,
    config: BeatCounterConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    peaks = np.asarray(peaks, dtype=int)
    proms = np.asarray(proms, dtype=float)
    widths_s = np.asarray(widths_s, dtype=float)
    tids = np.asarray(tids, dtype=int)
    if peaks.size <= 1:
        return peaks, proms, widths_s, tids

    order = np.argsort(peaks)
    peaks = peaks[order]
    proms = proms[order]
    widths_s = widths_s[order]
    tids = tids[order]

    dedup: List[Tuple[int, float, float, int]] = []
    refractory_s = max(float(config.refractory_dedup_window_s), float(config.dedup_min_sep_s))
    refractory_n = max(1, int(round(refractory_s * fs)))

    def _score(pk_i: int, pr_i: float, wd_i: float) -> float:
        amp_i = float(sig[int(pk_i)]) if 0 <= int(pk_i) < len(sig) else 0.0
        return float(pr_i * np.sqrt(max(wd_i, 1e-9)) * np.sqrt(max(abs(amp_i), 1e-9)))

    def _is_true_double_lobe(
        left_pk: int,
        left_pr: float,
        left_wd: float,
        left_tid: int,
        right_pk: int,
        right_pr: float,
        right_wd: float,
        right_tid: int,
    ) -> bool:
        if not bool(config.main_double_lobe_exception_enabled):
            return False
        if int(left_tid) != int(right_tid):
            return False
        dt_s = float((int(right_pk) - int(left_pk)) / fs)
        if dt_s <= 0 or dt_s > float(config.main_double_lobe_max_dt_s):
            return False
        if int(right_pk) - int(left_pk) < 2:
            return False

        left_amp = float(sig[int(left_pk)]) if 0 <= int(left_pk) < len(sig) else np.nan
        right_amp = float(sig[int(right_pk)]) if 0 <= int(right_pk) < len(sig) else np.nan
        if not (np.isfinite(left_amp) and np.isfinite(right_amp)):
            return False
        weaker_amp = min(left_amp, right_amp)
        weaker_prom = min(float(left_pr), float(right_pr))
        weaker_width = min(float(left_wd), float(right_wd))
        stronger_amp = max(left_amp, right_amp)

        if weaker_width < float(config.main_double_lobe_min_weak_width_s):
            return False
        if weaker_prom < float(config.main_double_lobe_min_weak_prom_abs):
            return False
        if weaker_amp < float(config.main_double_lobe_min_weak_amp_rel_strong) * max(stronger_amp, 1e-12):
            return False

        l = int(min(left_pk, right_pk))
        r = int(max(left_pk, right_pk))
        valley = float(np.min(np.asarray(sig[l : r + 1], dtype=float)))
        valley_ratio = (weaker_amp - valley) / max(abs(weaker_amp), 1e-12)
        rise_left = left_amp - valley
        rise_right = right_amp - valley
        reascent_ratio = min(rise_left, rise_right) / max(max(rise_left, rise_right), 1e-12)
        if valley_ratio < float(config.main_double_lobe_min_valley_ratio):
            return False
        if reascent_ratio < float(config.main_double_lobe_min_reascent_ratio):
            return False
        return True

    for pk, pr, wd, tid in zip(peaks, proms, widths_s, tids):
        pk = int(pk)
        pr = float(pr)
        wd = float(wd)
        tid = int(tid)
        if not dedup:
            dedup.append((pk, pr, wd, tid))
            continue

        prev_pk, prev_pr, prev_wd, prev_tid = dedup[-1]
        dt_n = int(pk - prev_pk)
        if dt_n > refractory_n:
            dedup.append((pk, pr, wd, tid))
            continue

        if _is_true_double_lobe(prev_pk, prev_pr, prev_wd, prev_tid, pk, pr, wd, tid):
            dedup.append((pk, pr, wd, tid))
            continue

        if _score(pk, pr, wd) >= _score(prev_pk, prev_pr, prev_wd):
            dedup[-1] = (pk, pr, wd, tid)
        # else: keep previous and drop current by refractory rule

    out_peaks = np.asarray([x[0] for x in dedup], dtype=int)
    out_proms = np.asarray([x[1] for x in dedup], dtype=float)
    out_widths = np.asarray([x[2] for x in dedup], dtype=float)
    out_tids = np.asarray([x[3] for x in dedup], dtype=int)
    return out_peaks, out_proms, out_widths, out_tids

def prune_short_gap_weak_mains(
    *,
    sig: np.ndarray,
    fs: float,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    main_tids: np.ndarray,
    config: BeatCounterConfig,
    protected_peaks: Optional[Set[int]] = None,
    enable_close_pair_preserve: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not bool(config.main_short_gap_filter_enabled):
        return main_peaks, main_proms, main_widths, main_tids, 0
    if main_peaks.size < 3:
        return main_peaks, main_proms, main_widths, main_tids, 0

    main_peaks = np.asarray(main_peaks, dtype=int)
    main_proms = np.asarray(main_proms, dtype=float)
    main_widths = np.asarray(main_widths, dtype=float)
    main_tids = np.asarray(main_tids, dtype=int)
    protected = set(int(x) for x in (protected_peaks or set()))

    ibi = np.diff(main_peaks) / max(fs, 1e-9)
    med_ibi = float(np.median(ibi)) if ibi.size else np.nan
    if not np.isfinite(med_ibi) or med_ibi <= 0:
        return main_peaks, main_proms, main_widths, main_tids, 0

    short_gap_s = max(float(config.main_short_gap_floor_s), float(config.main_short_gap_rel_ibi) * med_ibi)
    short_gap_n = max(1, int(round(short_gap_s * fs)))
    keep = np.ones(main_peaks.size, dtype=bool)

    for i in range(main_peaks.size - 1):
        if not keep[i]:
            continue
        j = i + 1
        if not keep[j]:
            continue
        dt = int(main_peaks[j] - main_peaks[i])
        if dt > short_gap_n:
            continue

        score_i = float(main_proms[i] * np.sqrt(max(main_widths[i], 1e-9)) * np.sqrt(max(float(sig[int(main_peaks[i])]), 1e-9)))
        score_j = float(main_proms[j] * np.sqrt(max(main_widths[j], 1e-9)) * np.sqrt(max(float(sig[int(main_peaks[j])]), 1e-9)))
        weak_idx, strong_idx = (i, j) if score_i <= score_j else (j, i)

        weak_prom = float(main_proms[weak_idx])
        strong_prom = float(main_proms[strong_idx])
        weak_w = float(main_widths[weak_idx])
        strong_w = float(main_widths[strong_idx])
        weak_amp = float(sig[int(main_peaks[weak_idx])])
        strong_amp = float(sig[int(main_peaks[strong_idx])])
        weak_pk = int(main_peaks[weak_idx])
        strong_pk = int(main_peaks[strong_idx])
        if weak_pk in protected or strong_pk in protected:
            continue

        prom_weak = weak_prom <= float(config.main_short_gap_weak_prom_ratio) * max(strong_prom, 1e-12)
        width_weak = weak_w <= float(config.main_short_gap_weak_width_ratio) * max(strong_w, 1e-12)
        amp_weak = weak_amp <= float(config.main_short_gap_weak_amp_ratio) * max(strong_amp, 1e-12)
        if (prom_weak and width_weak) or (prom_weak and amp_weak):
            # Preserve true close double-lobe morphology: meaningful valley + re-ascent
            # with non-tiny width/prominence on the weaker crest.
            left = min(weak_pk, strong_pk)
            right = max(weak_pk, strong_pk)
            keep_as_doublet = False
            if enable_close_pair_preserve and right - left >= 2:
                window = np.asarray(sig[left : right + 1], dtype=float)
                valley = float(np.min(window))
                smaller_peak = min(weak_amp, strong_amp)
                valley_ratio = (smaller_peak - valley) / max(abs(smaller_peak), 1e-12)
                rise_weak = weak_amp - valley
                rise_strong = strong_amp - valley
                rise_ratio = min(rise_weak, rise_strong) / max(max(rise_weak, rise_strong), 1e-12)
                keep_as_doublet = (
                    weak_w >= 0.070
                    and weak_prom >= 0.030
                    and weak_amp >= 0.65 * max(strong_amp, 1e-12)
                    and valley_ratio >= 0.06
                    and rise_ratio >= 0.16
                )
            if keep_as_doublet:
                continue
            keep[weak_idx] = False

    removed = int(np.sum(~keep))
    return main_peaks[keep], main_proms[keep], main_widths[keep], main_tids[keep], removed

def prune_local_weak_mains(
    *,
    sig: np.ndarray,
    fs: float,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    main_tids: np.ndarray,
    config: BeatCounterConfig,
    protected_peaks: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not bool(config.main_local_weak_filter_enabled):
        return main_peaks, main_proms, main_widths, main_tids, 0
    if main_peaks.size < 3:
        return main_peaks, main_proms, main_widths, main_tids, 0

    main_peaks = np.asarray(main_peaks, dtype=int)
    main_proms = np.asarray(main_proms, dtype=float)
    main_widths = np.asarray(main_widths, dtype=float)
    main_tids = np.asarray(main_tids, dtype=int)
    protected = set(int(x) for x in (protected_peaks or set()))
    ibi = np.diff(main_peaks) / max(fs, 1e-9)
    med_ibi = float(np.median(ibi)) if ibi.size else np.nan
    if not np.isfinite(med_ibi) or med_ibi <= 0:
        return main_peaks, main_proms, main_widths, main_tids, 0

    max_local_gap_n = max(1, int(round(float(config.main_local_weak_min_gap_rel_ibi) * med_ibi * fs)))
    keep = np.ones(main_peaks.size, dtype=bool)

    for i in range(1, main_peaks.size - 1):
        prev_pk = int(main_peaks[i - 1])
        cur_pk = int(main_peaks[i])
        next_pk = int(main_peaks[i + 1])
        if cur_pk in protected:
            continue
        dt_prev = int(cur_pk - prev_pk)
        dt_next = int(next_pk - cur_pk)
        if min(dt_prev, dt_next) > max_local_gap_n:
            continue

        ref_prom = float(np.median([main_proms[i - 1], main_proms[i + 1]]))
        ref_amp = float(np.median([sig[prev_pk], sig[next_pk]]))
        ref_width = float(np.median([main_widths[i - 1], main_widths[i + 1]]))
        if ref_prom <= 1e-12 or ref_amp <= 1e-12 or ref_width <= 1e-12:
            continue

        if (
            float(main_proms[i]) <= float(config.main_local_weak_rel_prom_max) * ref_prom
            and float(sig[cur_pk]) <= float(config.main_local_weak_rel_amp_max) * ref_amp
            and float(main_widths[i]) <= float(config.main_local_weak_rel_width_max) * ref_width
        ):
            keep[i] = False
            continue

        if bool(config.main_local_tiny_filter_enabled):
            tiny_profile = (
                float(main_proms[i]) <= float(config.main_local_tiny_rel_prom_max) * ref_prom
                and float(sig[cur_pk]) <= float(config.main_local_tiny_rel_amp_max) * ref_amp
                and float(main_widths[i]) <= float(config.main_local_tiny_rel_width_max) * ref_width
            )
            if tiny_profile:
                keep[i] = False

    removed = int(np.sum(~keep))
    return main_peaks[keep], main_proms[keep], main_widths[keep], main_tids[keep], removed


def prune_interbeat_tiny_bumps(
    *,
    sig: np.ndarray,
    fs: float,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    main_tids: np.ndarray,
    config: BeatCounterConfig,
    protected_peaks: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not bool(config.main_interbeat_tiny_filter_enabled):
        return main_peaks, main_proms, main_widths, main_tids, 0
    if main_peaks.size < 3:
        return main_peaks, main_proms, main_widths, main_tids, 0

    main_peaks = np.asarray(main_peaks, dtype=int)
    main_proms = np.asarray(main_proms, dtype=float)
    main_widths = np.asarray(main_widths, dtype=float)
    main_tids = np.asarray(main_tids, dtype=int)
    keep = np.ones(main_peaks.size, dtype=bool)
    protected = set(int(x) for x in (protected_peaks or set()))
    min_gap_n = max(1, int(round(float(config.main_interbeat_tiny_min_gap_s) * fs)))
    amp_vals = np.asarray(sig, dtype=float)[main_peaks]
    global_amp_ref = float(np.quantile(amp_vals, 0.80)) if amp_vals.size else np.nan
    global_prom_ref = float(np.quantile(main_proms, 0.80)) if main_proms.size else np.nan

    for i in range(1, main_peaks.size - 1):
        if not keep[i]:
            continue
        left = int(main_peaks[i - 1])
        cur = int(main_peaks[i])
        right = int(main_peaks[i + 1])
        if cur in protected:
            continue
        local_tiny = False
        if (cur - left) >= min_gap_n and (right - cur) >= min_gap_n:
            ref_amp = float(np.median([sig[left], sig[right]]))
            ref_prom = float(np.median([main_proms[i - 1], main_proms[i + 1]]))
            if ref_amp > 1e-12 and ref_prom > 1e-12:
                local_tiny = (
                    float(sig[cur]) <= float(config.main_interbeat_tiny_rel_amp_max) * ref_amp
                    and float(main_proms[i]) <= float(config.main_interbeat_tiny_rel_prom_max) * ref_prom
                    and float(main_widths[i]) <= float(config.main_interbeat_tiny_max_width_s)
                )
        global_tiny = False
        if np.isfinite(global_amp_ref) and np.isfinite(global_prom_ref) and global_amp_ref > 1e-12 and global_prom_ref > 1e-12:
            global_tiny = (
                float(sig[cur]) <= float(config.main_interbeat_tiny_global_amp_rel_max) * global_amp_ref
                and float(main_proms[i]) <= float(config.main_interbeat_tiny_global_prom_rel_max) * global_prom_ref
                and float(main_widths[i]) <= float(config.main_interbeat_tiny_max_width_s)
            )
        very_weak_local = False
        if (cur - left) >= max(1, min_gap_n // 2) and (right - cur) >= max(1, min_gap_n // 2):
            ref_amp = float(np.median([sig[left], sig[right]]))
            ref_prom = float(np.median([main_proms[i - 1], main_proms[i + 1]]))
            if ref_amp > 1e-12 and ref_prom > 1e-12:
                very_weak_local = (
                    float(sig[cur]) <= 0.38 * ref_amp
                    and float(main_proms[i]) <= 0.40 * ref_prom
                )
        very_weak_global = False
        if np.isfinite(global_amp_ref) and np.isfinite(global_prom_ref) and global_amp_ref > 1e-12 and global_prom_ref > 1e-12:
            very_weak_global = (
                float(sig[cur]) <= 0.40 * global_amp_ref
                and float(main_proms[i]) <= 0.42 * global_prom_ref
            )
        if local_tiny or global_tiny or very_weak_local or very_weak_global:
            keep[i] = False

    if np.isfinite(global_amp_ref) and np.isfinite(global_prom_ref) and global_amp_ref > 1e-12 and global_prom_ref > 1e-12:
        for i in [0, main_peaks.size - 1]:
            if i < 0 or i >= main_peaks.size or not keep[i]:
                continue
            cur = int(main_peaks[i])
            if cur in protected:
                continue
            if i == 0:
                if main_peaks.size < 2 or (int(main_peaks[1]) - cur) < max(1, min_gap_n // 2):
                    continue
            else:
                if main_peaks.size < 2 or (cur - int(main_peaks[-2])) < max(1, min_gap_n // 2):
                    continue
            edge_tiny = (
                float(sig[cur]) <= float(config.main_interbeat_tiny_global_amp_rel_max) * global_amp_ref
                and float(main_proms[i]) <= float(config.main_interbeat_tiny_global_prom_rel_max) * global_prom_ref
                and float(main_widths[i]) <= float(config.main_interbeat_tiny_max_width_s)
            )
            edge_very_weak = (
                float(sig[cur]) <= 0.40 * global_amp_ref
                and float(main_proms[i]) <= 0.42 * global_prom_ref
            )
            if edge_tiny or edge_very_weak:
                keep[i] = False

    removed = int(np.sum(~keep))
    return main_peaks[keep], main_proms[keep], main_widths[keep], main_tids[keep], removed

def prune_weak_primary_near_rescue(
    *,
    sig: np.ndarray,
    fs: float,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    main_tids: np.ndarray,
    rescue_peaks: np.ndarray,
    config: BeatCounterConfig,
    protected_peaks: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not bool(config.rescue_replace_prev_enabled):
        return main_peaks, main_proms, main_widths, main_tids, 0
    if main_peaks.size == 0 or rescue_peaks.size == 0:
        return main_peaks, main_proms, main_widths, main_tids, 0

    main_peaks = np.asarray(main_peaks, dtype=int)
    main_proms = np.asarray(main_proms, dtype=float)
    main_widths = np.asarray(main_widths, dtype=float)
    main_tids = np.asarray(main_tids, dtype=int)
    rescue_peaks = np.asarray(sorted(set(int(x) for x in rescue_peaks.tolist())), dtype=int)
    if rescue_peaks.size == 0:
        return main_peaks, main_proms, main_widths, main_tids, 0

    ibi = np.diff(main_peaks) / max(fs, 1e-9)
    med_ibi = float(np.median(ibi)) if ibi.size else np.nan
    max_dt_s = float(config.rescue_replace_prev_max_dt_s)
    if np.isfinite(med_ibi) and med_ibi > 0:
        max_dt_s = min(max_dt_s, float(config.rescue_replace_prev_max_dt_rel_ibi) * med_ibi)
    max_dt_n = max(1, int(round(max_dt_s * fs)))

    global_prom_ref = float(np.median(main_proms)) if main_proms.size else 0.0
    global_width_ref = float(np.median(main_widths)) if main_widths.size else 0.0
    rescue_proms = peak_prominences(np.asarray(sig, dtype=float), rescue_peaks)[0] if rescue_peaks.size else np.array([], dtype=float)
    protected = set(int(x) for x in (protected_peaks or set()))

    remove_idx: Set[int] = set()

    def _maybe_remove(i: int, rp: int, rpr: float) -> None:
        if i < 0 or i >= main_peaks.size:
            return
        if int(main_peaks[i]) in protected or int(rp) in protected:
            return
        dt = abs(int(rp) - int(main_peaks[i]))
        if dt <= 0 or dt > max_dt_n:
            return
        prom_i = float(main_proms[i])
        width_i = float(main_widths[i])
        amp_i = float(sig[int(main_peaks[i])])
        prom_ref = max(float(rpr), global_prom_ref, 1e-12)
        width_ref = max(global_width_ref, 1e-12)
        amp_ref = max(float(sig[int(rp)]), 1e-12)
        if (
            prom_i <= float(config.rescue_replace_prev_rel_prom) * prom_ref
            and width_i <= float(config.rescue_replace_prev_rel_width) * width_ref
            and amp_i <= float(config.rescue_replace_prev_rel_amp) * amp_ref
        ):
            remove_idx.add(int(i))

    for rp, rpr in zip(rescue_peaks.tolist(), np.asarray(rescue_proms, dtype=float).tolist()):
        prev = np.where(main_peaks < int(rp))[0]
        if prev.size > 0:
            _maybe_remove(int(prev[-1]), int(rp), float(rpr))
        nxt = np.where(main_peaks > int(rp))[0]
        if nxt.size > 0:
            _maybe_remove(int(nxt[0]), int(rp), float(rpr))

    if not remove_idx:
        return main_peaks, main_proms, main_widths, main_tids, 0

    keep = np.ones(main_peaks.size, dtype=bool)
    for i in remove_idx:
        keep[int(i)] = False
    return main_peaks[keep], main_proms[keep], main_widths[keep], main_tids[keep], int(np.sum(~keep))


def recover_missing_main_peaks_in_large_gaps(
    *,
    raw_peaks: np.ndarray,
    proms: np.ndarray,
    widths_s: np.ndarray,
    main_peaks: np.ndarray,
    fs: float,
    strong_thr: float,
    weak_thr: float,
    periodicity: float,
    corr_med: float,
    config: BeatCounterConfig,
) -> List[int]:
    if not bool(config.main_gap_fill_enabled):
        return []
    if main_peaks.size < 3 or raw_peaks.size == 0:
        return []
    if periodicity < float(config.main_gap_fill_periodicity_min):
        return []
    if np.isfinite(corr_med) and corr_med < float(config.main_gap_fill_corr_min):
        return []

    main_sorted = np.sort(np.asarray(main_peaks, dtype=int))
    ibi = np.diff(main_sorted) / max(fs, 1e-9)
    if ibi.size < 2:
        return []
    med_ibi = float(np.median(ibi))
    if not np.isfinite(med_ibi) or med_ibi <= 0:
        return []

    gap_thr_s = max(float(config.main_gap_fill_gap_mult) * med_ibi, 0.50)
    sep_n = max(1, int(round(float(config.main_gap_fill_min_sep_s) * fs)))
    cand_prom_min = max(float(config.min_secondary_prom), float(config.main_gap_fill_candidate_rel_strong) * float(strong_thr))
    cand_width_min = max(float(config.main_gap_fill_candidate_min_width_s), 0.5 * float(config.min_width_s))
    main_set = set(int(x) for x in main_sorted.tolist())

    added: List[int] = []
    for i in range(main_sorted.size - 1):
        left = int(main_sorted[i])
        right = int(main_sorted[i + 1])
        gap_s = float((right - left) / fs)
        if gap_s <= gap_thr_s:
            continue

        expected = int(round(gap_s / max(med_ibi, 1e-9))) - 1
        expected = max(1, min(int(config.main_gap_fill_max_add_per_gap), expected))

        idx = np.where(
            (raw_peaks > left + sep_n)
            & (raw_peaks < right - sep_n)
            & (~np.isin(raw_peaks, np.asarray(list(main_set), dtype=int)))
            & (proms >= cand_prom_min)
            & (widths_s >= cand_width_min)
        )[0]
        if idx.size == 0:
            continue

        idx_sorted = idx[np.argsort(proms[idx])[::-1]]
        chosen: List[int] = []
        for j in idx_sorted:
            pk = int(raw_peaks[int(j)])
            if all(abs(pk - int(raw_peaks[k])) >= sep_n for k in chosen):
                chosen.append(int(j))
            if len(chosen) >= expected:
                break
        for j in chosen:
            pk = int(raw_peaks[int(j)])
            if pk not in main_set:
                main_set.add(pk)
                added.append(pk)

    return sorted(set(int(x) for x in added))


def recover_close_second_lobe_main_candidates(
    *,
    raw_peaks: np.ndarray,
    proms: np.ndarray,
    widths_s: np.ndarray,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    sig: np.ndarray,
    fs: float,
    strong_thr: float,
    weak_thr: float,
) -> List[int]:
    if raw_peaks.size == 0 or main_peaks.size == 0:
        return []

    raw_peaks = np.asarray(raw_peaks, dtype=int)
    proms = np.asarray(proms, dtype=float)
    widths_s = np.asarray(widths_s, dtype=float)
    main_sorted = np.asarray(sorted(set(int(x) for x in np.asarray(main_peaks, dtype=int).tolist())), dtype=int)
    if main_sorted.size == 0:
        return []
    sig_arr = np.asarray(sig, dtype=float)

    main_prom_map: Dict[int, float] = {}
    for pk, pr in zip(np.asarray(main_peaks, dtype=int).tolist(), np.asarray(main_proms, dtype=float).tolist()):
        main_prom_map[int(pk)] = max(float(main_prom_map.get(int(pk), 0.0)), float(pr))

    main_set = set(int(x) for x in main_sorted.tolist())
    prom_map: Dict[int, float] = {int(pk): float(pr) for pk, pr in zip(raw_peaks.tolist(), proms.tolist())}
    width_map: Dict[int, float] = {int(pk): float(wd) for pk, wd in zip(raw_peaks.tolist(), widths_s.tolist())}

    cand_prom_min = 0.030
    cand_width_min = 0.070
    dt_min_s = 0.10
    dt_max_s = 0.90
    min_valley_ratio = 0.12
    min_reascent_ratio = 0.20

    ref_prom_min = 0.035
    ref_width_min = 0.055
    ref_sorted = np.asarray(sorted(set(int(x) for x in main_sorted.tolist())), dtype=int)
    if ref_sorted.size == 0:
        return []

    added: List[int] = []
    for pk, pr, wd in zip(raw_peaks.tolist(), proms.tolist(), widths_s.tolist()):
        pk_i = int(pk)
        pr_f = float(pr)
        wd_f = float(wd)
        if pk_i in main_set:
            continue
        if wd_f < cand_width_min:
            continue
        if not (pr_f >= cand_prom_min and pr_f < float(strong_thr)):
            continue
        if pk_i <= 0 or pk_i >= (sig_arr.size - 1):
            continue
        cand_amp = float(sig_arr[pk_i])
        if not np.isfinite(cand_amp):
            continue

        ref_idx = np.where(ref_sorted != pk_i)[0]
        if ref_idx.size == 0:
            continue
        neighbor_candidates = sorted(
            [int(ref_sorted[k]) for k in ref_idx.tolist()],
            key=lambda x: abs(int(x) - pk_i),
        )

        accepted = False
        for anchor in neighbor_candidates:
            dt_s = abs(int(anchor) - pk_i) / max(float(fs), 1e-9)
            if dt_s < dt_min_s or dt_s > dt_max_s:
                continue

            anchor_amp = float(sig_arr[int(anchor)]) if 0 <= int(anchor) < sig_arr.size else np.nan
            if not np.isfinite(anchor_amp) or not np.isfinite(cand_amp):
                continue

            anchor_prom = float(prom_map.get(int(anchor), main_prom_map.get(int(anchor), np.nan)))
            if not np.isfinite(anchor_prom) or anchor_prom <= 0:
                continue
            anchor_width = float(width_map.get(int(anchor), np.nan))
            if np.isfinite(anchor_width) and anchor_width < ref_width_min:
                continue
            if int(anchor) not in main_set and anchor_prom < ref_prom_min:
                continue

            left = min(int(anchor), pk_i)
            right = max(int(anchor), pk_i)
            if right - left < 2:
                continue
            window = sig_arr[left : right + 1]
            valley = float(np.min(window))
            smaller_peak = min(anchor_amp, cand_amp)
            valley_ratio = (smaller_peak - valley) / max(abs(smaller_peak), 1e-12)
            if valley_ratio < min_valley_ratio:
                continue

            rise_anchor = anchor_amp - valley
            rise_cand = cand_amp - valley
            rise_max = max(rise_anchor, rise_cand, 1e-12)
            rise_min = min(rise_anchor, rise_cand)
            reascent_ratio = rise_min / rise_max
            if reascent_ratio < min_reascent_ratio:
                continue

            prom_rel = pr_f / max(anchor_prom, 1e-12)
            amp_rel = cand_amp / max(anchor_amp, 1e-12)
            shape_ok = (
                (prom_rel >= 0.20 and amp_rel >= 0.65)
                or (prom_rel >= 0.04 and amp_rel >= 0.80 and dt_s <= 0.85)
            )
            strong_amp_pair_ok = (
                amp_rel >= 0.90
                and valley_ratio >= 0.40
                and reascent_ratio >= 0.55
                and dt_s <= 0.24
            )
            if not (shape_ok or strong_amp_pair_ok):
                continue

            accepted = True
            break

        if accepted:
            added.append(pk_i)

    return sorted(set(int(x) for x in added))


def refine_main_peaks_by_transient_coherence(
    *,
    raw_peaks: np.ndarray,
    proms: np.ndarray,
    widths_s: np.ndarray,
    peak_tids: np.ndarray,
    main_peaks: np.ndarray,
    sig: np.ndarray,
    fs: float,
    strong_thr: float,
    periodicity: float,
    corr_med: float,
    seg_meta: List[Tuple[int, int, int, float, int]],
    config: BeatCounterConfig,
) -> Tuple[List[int], Dict]:
    if not bool(config.main_transient_fill_enabled):
        return sorted(int(x) for x in np.asarray(main_peaks, dtype=int)), {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}
    if raw_peaks.size == 0:
        return [], {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}
    if periodicity < float(config.main_transient_fill_periodicity_min):
        return sorted(int(x) for x in np.asarray(main_peaks, dtype=int)), {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}
    if np.isfinite(corr_med) and corr_med < float(config.main_transient_fill_corr_min):
        return sorted(int(x) for x in np.asarray(main_peaks, dtype=int)), {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}

    raw_peaks = np.asarray(raw_peaks, dtype=int)
    proms = np.asarray(proms, dtype=float)
    widths_s = np.asarray(widths_s, dtype=float)
    peak_tids = np.asarray(peak_tids, dtype=int)
    main_set: Set[int] = set(int(x) for x in np.asarray(main_peaks, dtype=int).tolist())
    if not main_set:
        return [], {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}

    width_min = max(float(config.main_gap_fill_candidate_min_width_s), 0.5 * float(config.min_width_s))
    prom_min = max(float(config.min_secondary_prom), float(config.main_transient_fill_min_prom_rel_strong) * float(strong_thr))
    sep_n = max(1, int(round(float(config.main_transient_fill_min_sep_s) * fs)))
    edge_n = max(1, int(round(float(config.main_transient_edge_window_s) * fs)))

    by_tid: Dict[int, List[int]] = {}
    for i, tid in enumerate(peak_tids):
        by_tid.setdefault(int(tid), []).append(int(i))
    tid_bounds = {int(tid): (int(s), int(e)) for tid, s, e, _, _ in seg_meta}

    def _amp(pk: int) -> float:
        return float(sig[int(pk)])

    def _prom(pk: int) -> float:
        idx = np.where(raw_peaks == int(pk))[0]
        return float(proms[int(idx[0])]) if idx.size > 0 else 0.0

    def _width(pk: int) -> float:
        idx = np.where(raw_peaks == int(pk))[0]
        return float(widths_s[int(idx[0])]) if idx.size > 0 else 0.0

    def _far_enough(pk: int, *, exclude: Optional[int] = None) -> bool:
        for m in main_set:
            if exclude is not None and int(m) == int(exclude):
                continue
            if abs(int(pk) - int(m)) < sep_n:
                return False
        return True

    def _score(pk: int) -> float:
        w = float(_width(int(pk)))
        width_fac = min(max(w / 0.06, 0.0), 1.0)
        return float(_amp(int(pk)) * width_fac + 0.35 * _prom(int(pk)))

    def _candidates_for_tid(tid: int) -> List[int]:
        idxs = by_tid.get(int(tid), [])
        cand = [
            int(raw_peaks[i])
            for i in idxs
            if float(widths_s[i]) >= width_min
        ]
        return cand

    def _is_edge_peak(pk: int, tid: int) -> bool:
        bounds = tid_bounds.get(int(tid), None)
        if bounds is None:
            return False
        s, e = int(bounds[0]), int(bounds[1])
        if e <= s:
            return False
        return (int(pk) - s) <= edge_n or (e - 1 - int(pk)) <= edge_n

    def _pick_best_for_tid(tid: int, local_ref: float) -> Optional[int]:
        cand = _candidates_for_tid(int(tid))
        if not cand:
            return None
        selected: List[int] = []
        for pk in cand:
            is_edge = _is_edge_peak(int(pk), int(tid))
            amp_ok_edge = is_edge and (_amp(int(pk)) >= float(config.main_transient_edge_amp_rel_ref) * max(local_ref, 1e-12))
            if _prom(int(pk)) >= prom_min or amp_ok_edge:
                selected.append(int(pk))
        if not selected:
            return None
        return int(max(selected, key=_score))

    peak_tid_map = {int(pk): int(tid) for pk, tid in zip(raw_peaks.tolist(), peak_tids.tolist())}

    def _main_amp_by_tid() -> Dict[int, float]:
        out: Dict[int, float] = {}
        for pk in main_set:
            tid = peak_tid_map.get(int(pk), None)
            if tid is None:
                continue
            amp = _amp(pk)
            if tid not in out or amp > out[tid]:
                out[tid] = amp
        return out

    replaced = 0
    added_missing = 0
    added_tail = 0

    main_amp_tid = _main_amp_by_tid()
    if not main_amp_tid:
        return sorted(main_set), {"added_missing_transient_main": 0, "replaced_transient_main": 0, "added_tail_main": 0}
    global_ref = float(np.median(list(main_amp_tid.values())))
    n_trans = max(1, int(len(seg_meta)))
    fill_ratio = float(len(main_set) / max(n_trans, 1))
    allow_fill = fill_ratio >= 0.82

    min_tid = int(min(main_amp_tid.keys()))
    max_tid = int(max(main_amp_tid.keys()))

    for tid in range(min_tid, max_tid + 1):
        main_same_tid = [int(pk) for pk in main_set if peak_tid_map.get(int(pk), None) == int(tid)]
        neighbors = [amp for t, amp in main_amp_tid.items() if abs(int(t) - int(tid)) <= 2]
        local_ref = float(np.median(neighbors)) if neighbors else global_ref
        best_pk = _pick_best_for_tid(int(tid), local_ref)
        if best_pk is None:
            continue

        if main_same_tid:
            current_main_pk = max(main_same_tid, key=_amp)
            if int(best_pk) != int(current_main_pk):
                best_amp = _amp(best_pk)
                main_amp = _amp(current_main_pk)
                main_prom = _prom(current_main_pk)
                best_prom = _prom(best_pk)
                main_score = _score(current_main_pk)
                best_score = _score(best_pk)
                if (
                    (
                        best_amp >= float(config.main_transient_replace_amp_ratio) * max(main_amp, 1e-12)
                        or (
                            best_score >= float(config.main_transient_replace_score_ratio) * max(main_score, 1e-12)
                            and main_prom <= max(
                                float(config.main_transient_replace_main_prom_rel_strong_max) * float(strong_thr),
                                0.65 * float(best_prom),
                            )
                        )
                    )
                    and _far_enough(best_pk, exclude=current_main_pk)
                ):
                    main_set.discard(int(current_main_pk))
                    main_set.add(int(best_pk))
                    replaced += 1
            continue

        if allow_fill:
            if _amp(best_pk) >= float(config.main_transient_fill_min_amp_rel_ref) * max(local_ref, 1e-12) and _far_enough(best_pk):
                main_set.add(int(best_pk))
                added_missing += 1

    tail_n = max(0, int(config.main_transient_tail_n))
    if tail_n > 0 and fill_ratio >= 0.86:
        seg_tids = sorted(int(tid) for tid, _, _, _, _ in seg_meta)
        tail_tids = seg_tids[-tail_n:] if seg_tids else []
        for tid in tail_tids:
            has_main = any(peak_tid_map.get(int(pk), None) == int(tid) for pk in main_set)
            if has_main:
                continue
            best_pk = _pick_best_for_tid(int(tid), global_ref)
            if best_pk is None:
                continue
            if _amp(best_pk) >= float(config.main_transient_tail_min_amp_rel_ref) * max(global_ref, 1e-12) and _far_enough(best_pk):
                main_set.add(int(best_pk))
                added_tail += 1

    return sorted(main_set), {
        "added_missing_transient_main": int(added_missing),
        "replaced_transient_main": int(replaced),
        "added_tail_main": int(added_tail),
    }

def rescue_boundary_split_main_peaks(
    *,
    sig: np.ndarray,
    fs: float,
    seg_meta: List[Tuple[int, int, int, float, int]],
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    raw_proms: np.ndarray,
    config: BeatCounterConfig,
) -> Tuple[List[int], Dict]:
    meta = {
        "rescue_windows": 0,
        "rescue_candidates": 0,
        "rescue_added": 0,
        "rescue_boundary_windows": 0,
        "rescue_gap_windows": 0,
        "rescue_event_audit": [],
    }
    if not bool(config.rescue_enabled):
        return [], meta
    if len(seg_meta) < 2 or main_peaks.size < 2:
        return [], meta

    order_main = np.argsort(np.asarray(main_peaks, dtype=int))
    main_sorted = np.asarray(main_peaks, dtype=int)[order_main]
    main_proms = np.asarray(main_proms, dtype=float)[order_main]
    main_widths = np.asarray(main_widths, dtype=float)[order_main]
    if main_sorted.size < 2:
        return [], meta

    ibi = np.diff(main_sorted) / max(fs, 1e-9)
    if ibi.size == 0:
        return [], meta
    med_ibi = float(np.median(ibi))
    if not np.isfinite(med_ibi) or med_ibi <= 0:
        return [], meta

    win_n = max(1, int(round(float(config.rescue_window_s) * fs)))
    min_sep_s = max(float(config.rescue_min_sep_s), float(config.rescue_min_sep_rel_ibi) * med_ibi)
    min_sep_n = max(1, int(round(min_sep_s * fs)))
    start_relaxed_sep_n = max(1, int(round(0.045 * fs)))
    n = int(len(sig))

    windows: List[Dict] = []
    boundaries = [int(seg_meta[i][2]) for i in range(len(seg_meta) - 1)]
    for b in boundaries:
        c = int(np.clip(b, 0, max(n - 1, 0)))
        windows.append({
            "l": max(0, c - win_n),
            "r": min(max(n - 1, 0), c + win_n),
            "center": c,
            "type": "boundary",
            "used": 0,
        })
    meta["rescue_boundary_windows"] = int(len(windows))

    gap_thr_s = max(float(config.rescue_gap_mult) * med_ibi, float(config.rescue_min_gap_s))
    for left, right in zip(main_sorted[:-1], main_sorted[1:]):
        gap_s = float((int(right) - int(left)) / fs)
        if gap_s <= gap_thr_s:
            continue
        c = int(round((int(left) + int(right)) / 2.0))
        span = max(win_n, int(round(0.28 * (int(right) - int(left)))))
        l = max(int(left) + min_sep_n, c - span)
        r = min(int(right) - min_sep_n, c + span)
        if r <= l:
            continue
        windows.append({"l": int(l), "r": int(r), "center": int(c), "type": "gap", "used": 0})
        meta["rescue_gap_windows"] = int(meta["rescue_gap_windows"]) + 1

    if not windows:
        return [], meta
    meta["rescue_windows"] = int(len(windows))

    if raw_proms.size > 0:
        prom_floor = float(max(
            0.75 * float(config.min_secondary_prom),
            0.60 * float(np.quantile(np.asarray(raw_proms, dtype=float), float(np.clip(config.rescue_prom_quantile, 0.05, 0.95)))),
            0.80 * float(config.prom0),
        ))
    else:
        prom_floor = float(max(0.75 * float(config.min_secondary_prom), 0.80 * float(config.prom0)))

    cand, props = find_peaks(
        np.asarray(sig, dtype=float),
        prominence=prom_floor,
        distance=max(1, int(round(0.75 * float(config.min_peak_distance_s) * fs))),
        width=2,
    )
    if cand.size == 0:
        return [], meta
    cand_prom = np.asarray(props.get("prominences", np.array([], dtype=float)), dtype=float)
    cand_width = np.asarray(props.get("widths", np.array([], dtype=float)), dtype=float) / max(fs, 1e-9)
    if cand_prom.size != cand.size or cand_width.size != cand.size:
        return [], meta

    width_ok = (cand_width >= float(config.rescue_min_width_s)) & (cand_width <= float(config.rescue_max_width_s))
    cand = cand[width_ok]
    cand_prom = cand_prom[width_ok]
    cand_width = cand_width[width_ok]
    if cand.size == 0:
        return [], meta

    rescue_spike_cfg = replace(config, main_spike_max_width_s=min(float(config.main_spike_max_width_s), 0.030))
    cand, cand_prom, cand_width, _ = filter_needle_spike_peaks(sig, cand, cand_prom, cand_width, rescue_spike_cfg)
    if cand.size == 0:
        return [], meta

    global_ref_prom = float(np.median(main_proms)) if main_proms.size else float(np.median(cand_prom))
    global_ref_amp = float(np.median(np.asarray(sig, dtype=float)[main_sorted])) if main_sorted.size else float(np.median(np.asarray(sig, dtype=float)[cand]))
    global_ref_width = float(np.median(main_widths)) if main_widths.size else float(np.median(cand_width))
    local_span = max(win_n, int(round(2.0 * med_ibi * fs)))

    def _nearest_window_idx(pk: int) -> Optional[int]:
        idxs = [i for i, w in enumerate(windows) if int(w["l"]) <= int(pk) <= int(w["r"])]
        if not idxs:
            return None
        return min(idxs, key=lambda i: abs(int(pk) - int(windows[i]["center"])))

    scored_candidates: List[Dict] = []
    for pk, pr, wd in zip(cand.tolist(), cand_prom.tolist(), cand_width.tolist()):
        pk_i = int(pk)
        w_idx = _nearest_window_idx(pk_i)
        if w_idx is None:
            continue
        if main_sorted.size:
            nearest_idx = int(np.argmin(np.abs(main_sorted - pk_i)))
            nearest_dist = int(abs(int(main_sorted[nearest_idx]) - pk_i))
        else:
            nearest_idx = -1
            nearest_dist = np.inf
        if nearest_idx >= 0 and main_widths.size > nearest_idx:
            nearest_main_w = float(main_widths[nearest_idx])
            overlap_n = max(
                min_sep_n,
                int(round(float(config.rescue_overlap_width_factor) * nearest_main_w * fs)),
            )
            if nearest_main_w >= float(config.rescue_overlap_min_main_width_s) and nearest_dist <= overlap_n:
                continue
        first_main = int(main_sorted[0]) if main_sorted.size else None
        near_start_allow = bool(
            first_main is not None
            and int(pk_i) < first_main
            and (first_main - int(pk_i)) >= start_relaxed_sep_n
        )
        if nearest_dist < min_sep_n and not near_start_allow:
            continue

        nb_mask = np.abs(main_sorted - pk_i) <= local_span
        if int(np.sum(nb_mask)) >= 2:
            nb_idx = np.where(nb_mask)[0]
            ref_prom = float(np.median(main_proms[nb_idx]))
            ref_amp = float(np.median(np.asarray(sig, dtype=float)[main_sorted[nb_idx]]))
            ref_width = float(np.median(main_widths[nb_idx]))
        else:
            ref_prom = global_ref_prom
            ref_amp = global_ref_amp
            ref_width = global_ref_width

        amp_rel = float(sig[pk_i]) / max(ref_amp, 1e-12)
        prom_rel = float(pr) / max(ref_prom, 1e-12)
        width_rel = float(wd) / max(ref_width, 1e-12)
        strict_small_ok = (
            amp_rel >= float(config.rescue_min_rel_amp_strict)
            and prom_rel >= float(config.rescue_min_rel_prom_strict)
            and width_rel >= float(config.rescue_min_rel_width_strict)
        )
        boundary_override_ok = (
            windows[int(w_idx)]["type"] == "boundary"
            and wd >= float(config.rescue_boundary_min_width_s)
            and amp_rel >= float(config.rescue_boundary_amp_rel_ref)
            and prom_rel >= float(config.rescue_boundary_min_prom_rel_ref)
        )
        if not (strict_small_ok or boundary_override_ok):
            continue
        if (not boundary_override_ok) and (
            amp_rel < float(config.rescue_min_amp_rel_ref) or prom_rel < float(config.rescue_min_prom_rel_ref)
        ):
            continue

        left_idx = int(np.searchsorted(main_sorted, pk_i) - 1)
        right_idx = int(np.searchsorted(main_sorted, pk_i))
        left_gap = np.nan
        right_gap = np.nan
        if left_idx >= 0:
            left_gap = float((pk_i - int(main_sorted[left_idx])) / fs)
        if right_idx < main_sorted.size:
            right_gap = float((int(main_sorted[right_idx]) - pk_i) / fs)
        if left_idx >= 0 and right_idx < main_sorted.size:
            min_rel = min(left_gap, right_gap) / max(med_ibi, 1e-9)
            split_strong = bool(
                amp_rel >= 0.92 and prom_rel >= 0.80 and width_rel >= 0.78
            )
            if min_rel < float(config.rescue_min_neighbor_gap_rel_ibi) and not split_strong:
                continue

        score = float(pr * np.sqrt(max(wd, 1e-9)))
        window_type = str(windows[int(w_idx)].get("type", "other")).strip().lower()
        if window_type == "gap":
            rescue_type = "large_gap"
        elif window_type == "boundary":
            rescue_type = "boundary"
        else:
            rescue_type = "other"
        scored_candidates.append(
            {
                "score": score,
                "peak_index": int(pk_i),
                "prominence": float(pr),
                "width_s": float(wd),
                "window_index": int(w_idx),
                "rescue_type": rescue_type,
                "gap_before_s": float(left_gap) if np.isfinite(left_gap) else np.nan,
                "gap_after_s": float(right_gap) if np.isfinite(right_gap) else np.nan,
                "amp": float(sig[pk_i]),
                "amp_rel_ref": float(amp_rel),
                "prom_rel_ref": float(prom_rel),
                "width_rel_ref": float(width_rel),
            }
        )

    if not scored_candidates:
        return [], meta
    scored_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
    meta["rescue_candidates"] = int(len(scored_candidates))

    accepted: List[int] = []
    accepted_audit: List[Dict] = []
    for item in scored_candidates:
        pk_i = int(item["peak_index"])
        w_idx = int(item["window_index"])
        if windows[int(w_idx)]["used"] >= int(config.rescue_max_candidates_per_window):
            continue
        if accepted and np.min(np.abs(np.asarray(accepted, dtype=int) - int(pk_i))) < min_sep_n:
            continue
        nearest_dist = int(np.min(np.abs(main_sorted - int(pk_i)))) if main_sorted.size else np.inf
        first_main = int(main_sorted[0]) if main_sorted.size else None
        near_start_allow = bool(
            first_main is not None
            and int(pk_i) < first_main
            and (first_main - int(pk_i)) >= start_relaxed_sep_n
        )
        if nearest_dist < min_sep_n and not near_start_allow:
            continue
        accepted.append(int(pk_i))
        accepted_audit.append(item)
        windows[int(w_idx)]["used"] += 1

    accepted = sorted(set(int(x) for x in accepted))
    accepted_audit_map = {int(d["peak_index"]): d for d in accepted_audit}
    ordered_audit = []
    for pk in accepted:
        d = accepted_audit_map.get(int(pk), {})
        ordered_audit.append(
            {
                "peak_index": int(pk),
                "rescue_type": str(d.get("rescue_type", "other")),
                "gap_before_s": float(d.get("gap_before_s", np.nan)) if np.isfinite(float(d.get("gap_before_s", np.nan))) else np.nan,
                "gap_after_s": float(d.get("gap_after_s", np.nan)) if np.isfinite(float(d.get("gap_after_s", np.nan))) else np.nan,
                "amp": float(d.get("amp", np.nan)) if np.isfinite(float(d.get("amp", np.nan))) else np.nan,
                "prominence": float(d.get("prominence", np.nan)) if np.isfinite(float(d.get("prominence", np.nan))) else np.nan,
                "width_s": float(d.get("width_s", np.nan)) if np.isfinite(float(d.get("width_s", np.nan))) else np.nan,
                "amp_rel_ref": float(d.get("amp_rel_ref", np.nan)) if np.isfinite(float(d.get("amp_rel_ref", np.nan))) else np.nan,
                "prom_rel_ref": float(d.get("prom_rel_ref", np.nan)) if np.isfinite(float(d.get("prom_rel_ref", np.nan))) else np.nan,
                "width_rel_ref": float(d.get("width_rel_ref", np.nan)) if np.isfinite(float(d.get("width_rel_ref", np.nan))) else np.nan,
                "rescue_stage": "boundary_split",
            }
        )
    meta["rescue_added"] = int(len(accepted))
    meta["rescue_event_audit"] = ordered_audit
    return accepted, meta

def _quality_score(prom_snr: float, n_main: int, ibi_cv: float) -> float:
    q = float(np.log1p(max(prom_snr, 0.0)) + 0.14 * min(n_main, 12))
    if np.isfinite(ibi_cv):
        q -= 0.90 * max(0.0, ibi_cv - 0.12)
    if n_main == 0:
        q -= 1.5
    return q

def _compute_ibi_cv(main_peaks: np.ndarray, fs: float) -> float:
    if main_peaks.size < 3:
        return np.nan
    ibi = np.diff(main_peaks) / fs
    m = float(np.mean(ibi))
    if m <= 1e-9:
        return np.nan
    return float(np.std(ibi) / m)

def build_events_dataframe(
    time: np.ndarray,
    sig: np.ndarray,
    main_peaks: np.ndarray,
    main_proms: np.ndarray,
    main_widths: np.ndarray,
    main_tids: np.ndarray,
    *,
    rescue_peaks: Optional[np.ndarray] = None,
    rescue_audit_by_peak: Optional[Dict[int, Dict]] = None,
    rescue_peak_tid_map: Optional[Dict[int, int]] = None,
    rescue_peak_prom_map: Optional[Dict[int, float]] = None,
    rescue_peak_width_map: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for pk, pr, wd, tid in zip(main_peaks, main_proms, main_widths, main_tids):
        rows.append(
            {
                "Time_s": float(time[int(pk)]),
                "Type": "Main Beat",
                "Amp": float(sig[int(pk)]),
                "Prom": float(pr),
                "Width_s": float(wd),
                "Transient": int(tid) + 1,
                "RescueType": "",
                "RescueGapBefore_s": np.nan,
                "RescueGapAfter_s": np.nan,
                "RescueAmpRel": np.nan,
                "RescuePromRel": np.nan,
                "RescueWidthRel": np.nan,
            }
        )

    rescue_idxs = np.asarray(rescue_peaks if rescue_peaks is not None else np.array([], dtype=int), dtype=int)
    rescue_audit_by_peak = rescue_audit_by_peak or {}
    rescue_peak_tid_map = rescue_peak_tid_map or {}
    rescue_peak_prom_map = rescue_peak_prom_map or {}
    rescue_peak_width_map = rescue_peak_width_map or {}
    for pk in rescue_idxs.tolist():
        if int(pk) < 0 or int(pk) >= time.size:
            continue
        audit = rescue_audit_by_peak.get(int(pk), {})
        tid = rescue_peak_tid_map.get(int(pk), -1)
        rows.append(
            {
                "Time_s": float(time[int(pk)]),
                "Type": "Rescue",
                "Amp": float(sig[int(pk)]),
                "Prom": float(rescue_peak_prom_map.get(int(pk), audit.get("prominence", np.nan))),
                "Width_s": float(rescue_peak_width_map.get(int(pk), audit.get("width_s", np.nan))),
                "Transient": (int(tid) + 1) if int(tid) >= 0 else np.nan,
                "RescueType": str(audit.get("rescue_type", "other")),
                "RescueGapBefore_s": float(audit.get("gap_before_s", np.nan)),
                "RescueGapAfter_s": float(audit.get("gap_after_s", np.nan)),
                "RescueAmpRel": float(audit.get("amp_rel_ref", np.nan)),
                "RescuePromRel": float(audit.get("prom_rel_ref", np.nan)),
                "RescueWidthRel": float(audit.get("width_rel_ref", np.nan)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Time_s",
                "Type",
                "Amp",
                "Prom",
                "Width_s",
                "Transient",
                "RescueType",
                "RescueGapBefore_s",
                "RescueGapAfter_s",
                "RescueAmpRel",
                "RescuePromRel",
                "RescueWidthRel",
            ]
        )
    return pd.DataFrame(rows).sort_values("Time_s").reset_index(drop=True)


def _estimate_fs_from_time(time_array: np.ndarray) -> float:
    t = np.asarray(time_array, dtype=float)
    if t.size < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        return np.nan
    return float(1.0 / med_dt)


def _resolve_afc_line(mode: str, main_peak_amp: float, value: float) -> float:
    m = str(mode).strip().lower()
    if m in {"absolute", "fixed", "value"}:
        return float(value)
    if m in {"fraction_of_main", "relative_to_main", "relative"}:
        return float(main_peak_amp) * float(value)
    return float(main_peak_amp) + float(value)


def detect_secondary_candidates_in_window(
    time_array,
    signal_array,
    main_peak_time_s,
    window_start_s,
    window_end_s,
    lower_line,
    upper_line,
    min_prominence,
    min_width_s,
    max_width_s,
    min_distance_s,
):
    time = np.asarray(time_array, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    out = {
        "indices": np.array([], dtype=int),
        "times_s": np.array([], dtype=float),
        "amps": np.array([], dtype=float),
        "prominences": np.array([], dtype=float),
        "widths_s": np.array([], dtype=float),
        "window_start_s": float(window_start_s),
        "window_end_s": float(window_end_s),
        "lower_line": float(min(lower_line, upper_line)),
        "upper_line": float(max(lower_line, upper_line)),
    }
    if time.size == 0 or sig.size == 0 or time.size != sig.size:
        return out

    lo_line = float(min(lower_line, upper_line))
    hi_line = float(max(lower_line, upper_line))
    win_start = float(min(window_start_s, window_end_s))
    win_end = float(max(window_start_s, window_end_s))
    mp_t = float(main_peak_time_s)
    mask = np.isfinite(time) & np.isfinite(sig)
    mask &= time >= win_start
    mask &= time <= win_end
    mask &= time > mp_t
    idx = np.where(mask)[0]
    if idx.size < 3:
        return out

    fs = _estimate_fs_from_time(time)
    if not np.isfinite(fs) or fs <= 0:
        fs = 250.0
    min_dist_n = max(1, int(round(float(max(min_distance_s, 0.0)) * fs)))
    min_w_n = max(1, int(round(float(max(min_width_s, 0.0)) * fs)))
    max_w = float(max_width_s) if np.isfinite(max_width_s) else np.nan
    max_w_n = int(round(max_w * fs)) if np.isfinite(max_w) and max_w > 0 else None
    if max_w_n is not None and max_w_n < min_w_n:
        max_w_n = min_w_n
    width_arg = (min_w_n, max_w_n)

    local_sig = sig[idx]
    peaks_local, props = find_peaks(
        local_sig,
        prominence=max(float(min_prominence), 0.0),
        distance=min_dist_n,
        width=width_arg,
    )
    if peaks_local.size == 0:
        return out

    cand_idx = idx[np.asarray(peaks_local, dtype=int)]
    cand_t = time[cand_idx]
    cand_a = sig[cand_idx]
    cand_p = np.asarray(props.get("prominences", np.zeros(peaks_local.size, dtype=float)), dtype=float)
    cand_w = np.asarray(props.get("widths", np.zeros(peaks_local.size, dtype=float)), dtype=float) / max(fs, 1e-9)

    keep = (
        np.isfinite(cand_t)
        & np.isfinite(cand_a)
        & np.isfinite(cand_p)
        & np.isfinite(cand_w)
        & (cand_t > mp_t)
        & (cand_t >= win_start)
        & (cand_t <= win_end)
        & (cand_a >= lo_line)
        & (cand_a <= hi_line)
        & (cand_p >= float(min_prominence))
        & (cand_w >= float(min_width_s))
    )
    if np.isfinite(max_width_s):
        keep &= cand_w <= float(max_width_s)
    if not np.any(keep):
        return out

    cand_idx = cand_idx[keep]
    cand_t = cand_t[keep]
    cand_a = cand_a[keep]
    cand_p = cand_p[keep]
    cand_w = cand_w[keep]
    order = np.argsort(cand_t)
    out.update(
        {
            "indices": np.asarray(cand_idx[order], dtype=int),
            "times_s": np.asarray(cand_t[order], dtype=float),
            "amps": np.asarray(cand_a[order], dtype=float),
            "prominences": np.asarray(cand_p[order], dtype=float),
            "widths_s": np.asarray(cand_w[order], dtype=float),
            "window_start_s": float(win_start),
            "window_end_s": float(win_end),
            "lower_line": float(lo_line),
            "upper_line": float(hi_line),
        }
    )
    return out


def recompute_afc_candidates_for_item(
    review_item: AFCReviewItem,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    config: AFCReviewConfig,
) -> AFCReviewItem:
    detected = detect_secondary_candidates_in_window(
        time_array=time_array,
        signal_array=signal_array,
        main_peak_time_s=review_item.main_peak_time_s,
        window_start_s=review_item.window_start_s,
        window_end_s=review_item.window_end_s,
        lower_line=review_item.lower_line,
        upper_line=review_item.upper_line,
        min_prominence=float(config.min_secondary_prominence),
        min_width_s=float(config.min_secondary_width_s),
        max_width_s=float(config.max_secondary_width_s),
        min_distance_s=float(config.min_secondary_distance_s),
    )
    return replace(
        review_item,
        auto_candidate_times_s=[float(x) for x in detected["times_s"].tolist()],
        auto_candidate_amps=[float(x) for x in detected["amps"].tolist()],
    )


def build_afc_review_items(
    segment_results: Sequence[Dict],
    afc_config: AFCReviewConfig,
) -> List[AFCReviewItem]:
    items: List[AFCReviewItem] = []
    for i, seg in enumerate(segment_results):
        seg_name = str(seg.get("sheet_name", f"Segment {i + 1}"))
        seg_idx = int(seg.get("segment_no", i + 1))
        meta = seg.get("meta", {}) or {}
        events = seg.get("events", pd.DataFrame())
        if events is None or events.empty:
            continue
        if not bool(meta.get("qc_pass", False)):
            continue

        time_arr = np.asarray(meta.get("_time_plot", []), dtype=float)
        sig_arr = np.asarray(meta.get("_sig_plot", []), dtype=float)
        if time_arr.size == 0 or sig_arr.size == 0 or time_arr.size != sig_arr.size:
            continue

        main_df = events.loc[events["Type"].astype(str) == "Main Beat"].reset_index(drop=True)
        if main_df.empty:
            continue

        for main_i, row in main_df.iterrows():
            main_t = float(row.get("Time_s", np.nan))
            main_amp = float(row.get("Amp", np.nan))
            if not np.isfinite(main_t) or not np.isfinite(main_amp):
                continue

            win_start = float(main_t + float(afc_config.window_start_delay_s))
            win_end = float(main_t + float(afc_config.window_end_delay_s))
            if win_end <= win_start:
                continue
            win_start = max(win_start, float(time_arr[0]))
            win_end = min(win_end, float(time_arr[-1]))
            if win_end <= win_start:
                continue

            lower = _resolve_afc_line(str(afc_config.lower_line_mode), main_amp, float(afc_config.default_lower_line_offset))
            upper = _resolve_afc_line(str(afc_config.upper_line_mode), main_amp, float(afc_config.default_upper_line_offset))
            if lower > upper:
                lower, upper = upper, lower

            detected = detect_secondary_candidates_in_window(
                time_array=time_arr,
                signal_array=sig_arr,
                main_peak_time_s=main_t,
                window_start_s=win_start,
                window_end_s=win_end,
                lower_line=lower,
                upper_line=upper,
                min_prominence=float(afc_config.min_secondary_prominence),
                min_width_s=float(afc_config.min_secondary_width_s),
                max_width_s=float(afc_config.max_secondary_width_s),
                min_distance_s=float(afc_config.min_secondary_distance_s),
            )
            has_candidates = int(np.asarray(detected["times_s"]).size) > 0
            if (not bool(afc_config.review_all_main_peaks)) and (not has_candidates):
                continue
            if bool(afc_config.only_review_windows_with_candidates) and (not has_candidates):
                continue

            items.append(
                AFCReviewItem(
                    segment_name=seg_name,
                    segment_index=seg_idx,
                    main_peak_index=int(main_i),
                    main_peak_time_s=float(main_t),
                    main_peak_amp=float(main_amp),
                    window_start_s=float(win_start),
                    window_end_s=float(win_end),
                    lower_line=float(lower),
                    upper_line=float(upper),
                    auto_candidate_times_s=[float(x) for x in np.asarray(detected["times_s"], dtype=float).tolist()],
                    auto_candidate_amps=[float(x) for x in np.asarray(detected["amps"], dtype=float).tolist()],
                    status="pending",
                )
            )
    items.sort(key=lambda x: (int(x.segment_index), float(x.main_peak_time_s), int(x.main_peak_index)))
    return items


def _nearest_signal_amplitude(time_array: np.ndarray, signal_array: np.ndarray, at_time_s: float) -> float:
    t = np.asarray(time_array, dtype=float)
    s = np.asarray(signal_array, dtype=float)
    if t.size == 0 or s.size == 0 or t.size != s.size or (not np.isfinite(at_time_s)):
        return np.nan
    idx = int(np.argmin(np.abs(t - float(at_time_s))))
    return float(s[idx]) if 0 <= idx < s.size else np.nan


def _nearest_signal_peak_features(time_array: np.ndarray, signal_array: np.ndarray, at_time_s: float) -> Tuple[float, float]:
    t = np.asarray(time_array, dtype=float)
    s = np.asarray(signal_array, dtype=float)
    if t.size < 3 or s.size < 3 or t.size != s.size or (not np.isfinite(at_time_s)):
        return np.nan, np.nan
    idx = int(np.argmin(np.abs(t - float(at_time_s))))
    if idx <= 0 or idx >= (s.size - 1):
        return np.nan, np.nan
    try:
        prom = float(peak_prominences(s, np.asarray([idx], dtype=int))[0][0])
    except Exception:
        prom = np.nan
    try:
        dt = np.diff(t)
        fs = float(1.0 / np.median(dt)) if dt.size else np.nan
        width_samples = float(peak_widths(s, np.asarray([idx], dtype=int), rel_height=0.5)[0][0])
        width_s = float(width_samples / fs) if np.isfinite(fs) and fs > 0 else np.nan
    except Exception:
        width_s = np.nan
    return prom, width_s


def _dedup_times(values: Sequence[float], *, ndigits: int = 6) -> List[float]:
    out: List[float] = []
    seen: Set[float] = set()
    for x in values:
        try:
            xf = float(x)
        except Exception:
            continue
        if not np.isfinite(xf):
            continue
        key = float(round(xf, ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(xf))
    out.sort()
    return out


def merge_afc_decisions_with_results(
    segment_results: Sequence[Dict],
    decisions: Sequence[AFCReviewDecision],
) -> Tuple[List[Dict], List[AFCEvent], pd.DataFrame]:
    merged_results: List[Dict] = []
    seg_by_index: Dict[int, Dict] = {}
    seg_by_name: Dict[str, Dict] = {}
    for i, seg in enumerate(segment_results):
        seg_copy = dict(seg)
        seg_copy["events"] = seg.get("events", pd.DataFrame()).copy()
        seg_copy["meta"] = dict(seg.get("meta", {}) or {})
        seg_copy["afc_events_df"] = pd.DataFrame(
            columns=[
                "segment_name",
                "segment_index",
                "main_peak_index",
                "time_s",
                "amplitude",
                "source",
                "review_id",
            ]
        )
        seg_copy["events_with_afc"] = seg_copy["events"].copy()
        merged_results.append(seg_copy)
        seg_idx = int(seg_copy.get("segment_no", i + 1))
        seg_name = str(seg_copy.get("sheet_name", f"Segment {seg_idx}"))
        seg_by_index[seg_idx] = seg_copy
        seg_by_name[seg_name] = seg_copy

    afc_events: List[AFCEvent] = []
    per_segment_rows: Dict[int, List[Dict]] = {}
    review_rows: List[Dict] = []

    for dec in decisions:
        seg = seg_by_index.get(int(dec.segment_index)) or seg_by_name.get(str(dec.segment_name))
        if seg is None:
            continue
        seg_idx = int(seg.get("segment_no", dec.segment_index))
        seg_name = str(seg.get("sheet_name", dec.segment_name))
        review_id = dec.review_id

        auto_times = _dedup_times(dec.accepted_times_s)
        manual_times = _dedup_times(dec.manual_added_times_s)
        rejected_times = _dedup_times(dec.rejected_times_s)
        review_rows.append(
            {
                "segment_name": seg_name,
                "segment_index": seg_idx,
                "main_peak_index": int(dec.main_peak_index),
                "lower_line": float(dec.lower_line),
                "upper_line": float(dec.upper_line),
                "window_start_s": float(dec.window_start_s),
                "window_end_s": float(dec.window_end_s),
                "accepted_times_s": ",".join(f"{x:.6f}" for x in auto_times),
                "rejected_times_s": ",".join(f"{x:.6f}" for x in rejected_times),
                "manual_added_times_s": ",".join(f"{x:.6f}" for x in manual_times),
                "notes": str(dec.notes),
                "status": str(dec.status),
            }
        )

        if str(dec.status).lower() == "skipped":
            continue

        time_arr = np.asarray(seg.get("meta", {}).get("_time_plot", []), dtype=float)
        sig_arr = np.asarray(seg.get("meta", {}).get("_sig_plot", []), dtype=float)

        for src, times in (("auto_accepted", auto_times), ("manual_added", manual_times)):
            for t in times:
                amp = _nearest_signal_amplitude(time_arr, sig_arr, t)
                prom, width_s = _nearest_signal_peak_features(time_arr, sig_arr, t)
                event = AFCEvent(
                    segment_name=seg_name,
                    segment_index=seg_idx,
                    main_peak_index=int(dec.main_peak_index),
                    time_s=float(t),
                    amplitude=float(amp) if np.isfinite(amp) else np.nan,
                    source=src,
                    prominence=float(prom) if np.isfinite(prom) else np.nan,
                    width_s=float(width_s) if np.isfinite(width_s) else np.nan,
                    review_id=review_id,
                )
                afc_events.append(event)
                per_segment_rows.setdefault(seg_idx, []).append(event.to_dict())

    for seg_idx, seg in seg_by_index.items():
        rows = per_segment_rows.get(int(seg_idx), [])
        if rows:
            afc_df = pd.DataFrame(rows).sort_values(["main_peak_index", "time_s"]).reset_index(drop=True)
        else:
            afc_df = pd.DataFrame(
                columns=[
                    "segment_name",
                    "segment_index",
                    "main_peak_index",
                    "time_s",
                    "amplitude",
                    "prominence",
                    "width_s",
                    "source",
                    "review_id",
                ]
            )
        seg["afc_events_df"] = afc_df

        main_df = seg.get("events", pd.DataFrame()).copy()
        if not afc_df.empty:
            afc_as_events = pd.DataFrame(
                {
                    "Time_s": pd.to_numeric(afc_df["time_s"], errors="coerce"),
                    "Type": "AFC",
                    "Amp": pd.to_numeric(afc_df["amplitude"], errors="coerce"),
                    "Prom": pd.to_numeric(afc_df.get("prominence", np.nan), errors="coerce"),
                    "Width_s": pd.to_numeric(afc_df.get("width_s", np.nan), errors="coerce"),
                    "Transient": np.nan,
                    "Source": afc_df["source"].astype(str),
                    "ReviewID": afc_df["review_id"].astype(str),
                    "MainPeakIndex": pd.to_numeric(afc_df["main_peak_index"], errors="coerce"),
                }
            )
            main_with_meta = main_df.copy()
            for col in ["Source", "ReviewID", "MainPeakIndex"]:
                if col not in main_with_meta.columns:
                    main_with_meta[col] = np.nan
            events_with_afc = (
                pd.concat([main_with_meta, afc_as_events], ignore_index=True, sort=False)
                .sort_values("Time_s")
                .reset_index(drop=True)
            )
        else:
            events_with_afc = main_df
        seg["events_with_afc"] = events_with_afc

    review_log_df = pd.DataFrame(review_rows)
    if not review_log_df.empty:
        review_log_df = review_log_df.sort_values(["segment_index", "main_peak_index"]).reset_index(drop=True)
    return merged_results, afc_events, review_log_df


def detect_secondary_candidates_in_segment(
    time_array,
    signal_array,
    main_peak_times_s: Sequence[float],
    afc_lower_left_value,
    afc_lower_right_value,
    afc_upper_left_value,
    afc_upper_right_value,
    min_prominence,
    min_width_s,
    max_width_s,
    min_distance_s,
    main_peak_exclusion_window_s: float = 0.0,
    x_start_s: Optional[float] = None,
    x_end_s: Optional[float] = None,
):
    time = np.asarray(time_array, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    lower_left = float(afc_lower_left_value)
    lower_right = float(afc_lower_right_value)
    upper_left = float(afc_upper_left_value)
    upper_right = float(afc_upper_right_value)
    out = {
        "indices": np.array([], dtype=int),
        "times_s": np.array([], dtype=float),
        "amps": np.array([], dtype=float),
        "prominences": np.array([], dtype=float),
        "widths_s": np.array([], dtype=float),
        "afc_lower_values": np.array([], dtype=float),
        "afc_upper_values": np.array([], dtype=float),
        "x_start_s": float(x_start_s) if x_start_s is not None else np.nan,
        "x_end_s": float(x_end_s) if x_end_s is not None else np.nan,
        "afc_lower_left_value": float(lower_left),
        "afc_lower_right_value": float(lower_right),
        "afc_upper_left_value": float(upper_left),
        "afc_upper_right_value": float(upper_right),
    }
    if time.size == 0 or sig.size == 0 or time.size != sig.size:
        return out
    if not (np.isfinite(lower_left) and np.isfinite(lower_right) and np.isfinite(upper_left) and np.isfinite(upper_right)):
        return out

    x_start = float(x_start_s) if x_start_s is not None and np.isfinite(x_start_s) else float(time[0])
    x_end = float(x_end_s) if x_end_s is not None and np.isfinite(x_end_s) else float(time[-1])
    if x_end < x_start:
        x_start, x_end = x_end, x_start
    x_start = max(float(time[0]), x_start)
    x_end = min(float(time[-1]), x_end)
    if x_end <= x_start:
        return out

    mask = np.isfinite(time) & np.isfinite(sig) & (time >= x_start) & (time <= x_end)
    idx = np.where(mask)[0]
    if idx.size < 3:
        return out

    fs = _estimate_fs_from_time(time)
    if not np.isfinite(fs) or fs <= 0:
        fs = 250.0
    min_dist_n = max(1, int(round(float(max(min_distance_s, 0.0)) * fs)))
    min_w_n = max(1, int(round(float(max(min_width_s, 0.0)) * fs)))
    max_w = float(max_width_s) if np.isfinite(max_width_s) else np.nan
    max_w_n = int(round(max_w * fs)) if np.isfinite(max_w) and max_w > 0 else None
    if max_w_n is not None and max_w_n < min_w_n:
        max_w_n = min_w_n

    local_sig = sig[idx]
    peaks_local, props = find_peaks(
        local_sig,
        prominence=max(float(min_prominence), 0.0),
        distance=min_dist_n,
        width=(min_w_n, max_w_n),
    )
    if peaks_local.size == 0:
        return out

    cand_idx = idx[np.asarray(peaks_local, dtype=int)]
    cand_t = time[cand_idx]
    cand_a = sig[cand_idx]
    cand_p = np.asarray(props.get("prominences", np.zeros(peaks_local.size, dtype=float)), dtype=float)
    cand_w = np.asarray(props.get("widths", np.zeros(peaks_local.size, dtype=float)), dtype=float) / max(fs, 1e-9)
    lower_line = build_sloped_afc_threshold(cand_t, x_start, x_end, lower_left, lower_right)
    upper_line = build_sloped_afc_threshold(cand_t, x_start, x_end, upper_left, upper_right)
    low_bound = np.minimum(lower_line, upper_line)
    high_bound = np.maximum(lower_line, upper_line)

    keep = (
        np.isfinite(cand_t)
        & np.isfinite(cand_a)
        & np.isfinite(cand_p)
        & np.isfinite(cand_w)
        & (cand_t >= x_start)
        & (cand_t <= x_end)
        & np.isfinite(low_bound)
        & np.isfinite(high_bound)
        & (cand_a >= low_bound)
        & (cand_a <= high_bound)
        & (cand_p >= float(min_prominence))
        & (cand_w >= float(min_width_s))
    )
    if np.isfinite(max_width_s):
        keep &= cand_w <= float(max_width_s)
    main_times = np.asarray([float(x) for x in main_peak_times_s if np.isfinite(float(x))], dtype=float)
    excl = float(max(main_peak_exclusion_window_s, 0.0))
    if excl > 0 and main_times.size > 0:
        nearest_main_dist = np.min(np.abs(cand_t[:, None] - main_times[None, :]), axis=1)
        keep &= nearest_main_dist >= excl
    if not np.any(keep):
        return out

    cand_idx = cand_idx[keep]
    cand_t = cand_t[keep]
    cand_a = cand_a[keep]
    cand_p = cand_p[keep]
    cand_w = cand_w[keep]
    lower_line = lower_line[keep]
    upper_line = upper_line[keep]
    order = np.argsort(cand_t)
    out.update(
        {
            "indices": np.asarray(cand_idx[order], dtype=int),
            "times_s": np.asarray(cand_t[order], dtype=float),
            "amps": np.asarray(cand_a[order], dtype=float),
            "prominences": np.asarray(cand_p[order], dtype=float),
            "widths_s": np.asarray(cand_w[order], dtype=float),
            "afc_lower_values": np.asarray(lower_line[order], dtype=float),
            "afc_upper_values": np.asarray(upper_line[order], dtype=float),
            "x_start_s": float(x_start),
            "x_end_s": float(x_end),
            "afc_lower_left_value": float(lower_left),
            "afc_lower_right_value": float(lower_right),
            "afc_upper_left_value": float(upper_left),
            "afc_upper_right_value": float(upper_right),
        }
    )
    return out


def recompute_afc_candidates_for_segment(
    review_item: AFCSegmentReviewItem,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    config: AFCReviewConfig,
) -> AFCSegmentReviewItem:
    time = np.asarray(time_array, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    if time.size == 0 or sig.size == 0 or time.size != sig.size:
        return replace(
            review_item,
            helper_candidate_times_s=[],
            helper_candidate_amps=[],
            auto_candidate_times_s=[],
            auto_candidate_amps=[],
        )
    x_start = float(review_item.x_start_s) if np.isfinite(review_item.x_start_s) else float(time[0])
    x_end = float(review_item.x_end_s) if np.isfinite(review_item.x_end_s) else float(time[-1])
    if x_end < x_start:
        x_start, x_end = x_end, x_start
    x_start = max(float(time[0]), x_start)
    x_end = min(float(time[-1]), x_end)
    if x_end <= x_start:
        x_start = float(time[0])
        x_end = float(time[-1])
    lower_left = float(review_item.afc_lower_left_value)
    lower_right = float(review_item.afc_lower_right_value)
    upper_left = float(review_item.afc_upper_left_value)
    upper_right = float(review_item.afc_upper_right_value)
    if not np.isfinite(upper_left):
        upper_left = _fallback_upper_cap_from_signal(sig, config)
    if not np.isfinite(upper_right):
        upper_right = upper_left
    detected = detect_secondary_candidates_in_segment(
        time_array=time,
        signal_array=sig,
        main_peak_times_s=review_item.main_peak_times_s,
        afc_lower_left_value=lower_left,
        afc_lower_right_value=lower_right,
        afc_upper_left_value=upper_left,
        afc_upper_right_value=upper_right,
        min_prominence=float(config.min_secondary_prominence),
        min_width_s=float(config.min_secondary_width_s),
        max_width_s=float(config.max_secondary_width_s),
        min_distance_s=float(config.min_secondary_distance_s),
        main_peak_exclusion_window_s=float(config.main_peak_exclusion_window_s),
        x_start_s=float(x_start),
        x_end_s=float(x_end),
    )
    det_times = [float(x) for x in np.asarray(detected["times_s"], dtype=float).tolist()]
    det_amps = [float(x) for x in np.asarray(detected["amps"], dtype=float).tolist()]
    return replace(
        review_item,
        x_start_s=float(x_start),
        x_end_s=float(x_end),
        afc_upper_left_value=float(upper_left),
        afc_upper_right_value=float(upper_right),
        helper_candidate_times_s=det_times,
        helper_candidate_amps=det_amps,
        auto_candidate_times_s=det_times,
        auto_candidate_amps=det_amps,
    )


def _fallback_upper_cap_from_signal(sig_values: np.ndarray, afc_config: AFCReviewConfig) -> float:
    s = np.asarray(sig_values, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        fallback = float(afc_config.default_afc_upper_cap_fallback)
        return float(fallback if np.isfinite(fallback) else 1.0)
    cfg_cap = float(afc_config.default_afc_upper_cap_value)
    if np.isfinite(cfg_cap):
        return float(cfg_cap)
    q = float(np.clip(afc_config.default_afc_upper_cap_quantile, 0.0, 1.0))
    cap_q = float(np.quantile(s, q))
    if np.isfinite(cap_q):
        return float(cap_q)
    max_sig = float(np.max(s))
    if np.isfinite(max_sig):
        return float(max_sig)
    fallback = float(afc_config.default_afc_upper_cap_fallback)
    return float(fallback if np.isfinite(fallback) else 1.0)


def infer_afc_upper_cap_from_segment_meta(meta: Dict, sig_values: np.ndarray, afc_config: AFCReviewConfig) -> float:
    candidates = [
        "afc_upper_cap",
        "upper_line",
        "upper_threshold",
        "strong_thr",
        "weak_thr",
    ]
    if bool(afc_config.prefer_report_upper_line):
        for key in candidates:
            val = meta.get(key, np.nan)
            try:
                vf = float(val)
            except Exception:
                continue
            if np.isfinite(vf):
                return float(vf)
    return _fallback_upper_cap_from_signal(np.asarray(sig_values, dtype=float), afc_config)


def extract_report_peaks_for_reviewer(
    *,
    events_df: pd.DataFrame,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    rescue_indices: Sequence[int],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    main_times, main_amps = extract_fixed_main_peaks_from_events(events_df)
    t = np.asarray(time_array, dtype=float)
    s = np.asarray(signal_array, dtype=float)
    rescue_times: List[float] = []
    rescue_amps: List[float] = []
    if t.size and s.size and t.size == s.size:
        for idx in rescue_indices:
            try:
                ii = int(idx)
            except Exception:
                continue
            if 0 <= ii < t.size:
                rescue_times.append(float(t[ii]))
                rescue_amps.append(float(s[ii]))
    return main_times, main_amps, rescue_times, rescue_amps


def build_afc_segment_review_items(
    segment_results: Sequence[Dict],
    afc_config: AFCReviewConfig,
) -> List[AFCSegmentReviewItem]:
    items: List[AFCSegmentReviewItem] = []
    for i, seg in enumerate(segment_results):
        seg_name = str(seg.get("sheet_name", f"Segment {i + 1}"))
        seg_idx = int(seg.get("segment_no", i + 1))
        meta = seg.get("meta", {}) or {}
        events = seg.get("events", pd.DataFrame())
        if events is None or events.empty:
            continue
        if not bool(meta.get("qc_pass", False)):
            continue

        time_arr = np.asarray(meta.get("_time_plot", []), dtype=float)
        sig_arr = np.asarray(meta.get("_sig_plot", []), dtype=float)
        if time_arr.size == 0 or sig_arr.size == 0 or time_arr.size != sig_arr.size:
            continue

        main_times, main_amps, rescue_times, rescue_amps = extract_report_peaks_for_reviewer(
            events_df=events,
            time_array=time_arr,
            signal_array=sig_arr,
            rescue_indices=meta.get("_rescue_peaks_plot", []),
        )
        if not main_times:
            continue
        afc_left = float(afc_config.default_afc_lower_left_value)
        if not np.isfinite(afc_left):
            afc_left = float(afc_config.default_afc_left_value)
        afc_right = float(afc_config.default_afc_lower_right_value)
        if not np.isfinite(afc_right):
            afc_right = float(afc_config.default_afc_right_value)
        afc_upper_cap = infer_afc_upper_cap_from_segment_meta(meta, sig_arr, afc_config)
        afc_upper_left = float(afc_config.default_afc_upper_left_value)
        afc_upper_right = float(afc_config.default_afc_upper_right_value)
        if not np.isfinite(afc_upper_left):
            afc_upper_left = float(afc_upper_cap)
        if not np.isfinite(afc_upper_right):
            afc_upper_right = float(afc_upper_cap)

        item = AFCSegmentReviewItem(
            segment_name=seg_name,
            segment_index=seg_idx,
            afc_lower_left_value=float(afc_left),
            afc_lower_right_value=float(afc_right),
            afc_upper_left_value=float(afc_upper_left),
            afc_upper_right_value=float(afc_upper_right),
            x_start_s=float(time_arr[0]),
            x_end_s=float(time_arr[-1]),
            status="pending",
            main_peak_times_s=main_times,
            main_peak_amps=main_amps,
            rescue_peak_times_s=rescue_times,
            rescue_peak_amps=rescue_amps,
            helper_candidate_times_s=[],
            helper_candidate_amps=[],
            manual_afc_times_s=[],
            manual_afc_amps=[],
            auto_candidate_times_s=[],
            auto_candidate_amps=[],
            accepted_times_s=[],
            accepted_amps=[],
            manual_added_times_s=[],
            manual_added_amps=[],
            rejected_times_s=[],
            rejected_amps=[],
            notes="",
        )
        items.append(item)

    items.sort(key=lambda x: int(x.segment_index))
    return items


def assign_afc_events_to_previous_main_peak(
    afc_times_s: Sequence[float],
    main_peak_times_s: Sequence[float],
) -> List[int]:
    main_times = np.asarray([float(x) for x in main_peak_times_s if np.isfinite(float(x))], dtype=float)
    if main_times.size == 0:
        return [-1 for _ in afc_times_s]
    main_times = np.sort(main_times)
    out: List[int] = []
    for t in afc_times_s:
        try:
            tt = float(t)
        except Exception:
            out.append(-1)
            continue
        if not np.isfinite(tt):
            out.append(-1)
            continue
        idx = int(np.searchsorted(main_times, tt, side="right") - 1)
        out.append(idx if idx >= 0 else -1)
    return out


def extract_fixed_main_peaks_from_events(events_df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    if events_df is None or events_df.empty:
        return [], []
    if "Type" not in events_df.columns or "Time_s" not in events_df.columns:
        return [], []
    m = events_df.loc[events_df["Type"].astype(str) == "Main Beat"].copy()
    if m.empty:
        return [], []
    m["Time_s"] = pd.to_numeric(m["Time_s"], errors="coerce")
    m["Amp"] = pd.to_numeric(m.get("Amp", np.nan), errors="coerce")
    m = m.dropna(subset=["Time_s"]).sort_values("Time_s").reset_index(drop=True)
    if m.empty:
        return [], []
    times = [float(x) for x in m["Time_s"].tolist()]
    amps = [float(x) for x in m["Amp"].tolist()] if "Amp" in m.columns else [np.nan for _ in times]
    return times, amps


def build_sloped_afc_threshold(
    x_values: np.ndarray,
    x_start_s: float,
    x_end_s: float,
    afc_left_value: float,
    afc_right_value: float,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if x.size == 0:
        return out
    xs = float(x_start_s)
    xe = float(x_end_s)
    if not np.isfinite(xs) or not np.isfinite(xe):
        return out
    if xe == xs:
        out[:] = float(afc_left_value)
        return out
    if xe < xs:
        xs, xe = xe, xs
        lv = float(afc_right_value)
        rv = float(afc_left_value)
    else:
        lv = float(afc_left_value)
        rv = float(afc_right_value)
    frac = (x - xs) / max((xe - xs), 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    out = lv + frac * (rv - lv)
    return out


def merge_afc_segment_decisions_with_results(
    segment_results: Sequence[Dict],
    decisions: Sequence[AFCSegmentReviewDecision],
) -> Tuple[List[Dict], List[AFCEvent], pd.DataFrame]:
    merged_results: List[Dict] = []
    seg_by_index: Dict[int, Dict] = {}
    for i, seg in enumerate(segment_results):
        seg_copy = dict(seg)
        seg_copy["events"] = seg.get("events", pd.DataFrame()).copy()
        seg_copy["meta"] = dict(seg.get("meta", {}) or {})
        seg_copy["afc_events_df"] = pd.DataFrame(
            columns=[
                "segment_name",
                "segment_index",
                "main_peak_index",
                "time_s",
                "amplitude",
                "source",
                "review_id",
            ]
        )
        seg_copy["events_with_afc"] = seg_copy["events"].copy()
        merged_results.append(seg_copy)
        seg_idx = int(seg_copy.get("segment_no", i + 1))
        seg_by_index[seg_idx] = seg_copy

    afc_events: List[AFCEvent] = []
    per_segment_rows: Dict[int, List[Dict]] = {}
    review_rows: List[Dict] = []

    for dec in decisions:
        seg = seg_by_index.get(int(dec.segment_index))
        if seg is None:
            continue
        seg_idx = int(seg.get("segment_no", dec.segment_index))
        seg_name = str(seg.get("sheet_name", dec.segment_name))

        manual_times = _dedup_times(
            list(dec.manual_afc_times_s)
            if dec.manual_afc_times_s
            else (list(dec.manual_added_times_s) + list(dec.accepted_times_s))
        )
        manual_amp_sources = [
            (dec.manual_afc_times_s, dec.manual_afc_amps),
            (dec.manual_added_times_s, dec.manual_added_amps),
            (dec.accepted_times_s, dec.accepted_amps),
        ]
        amp_lookup: Dict[float, float] = {}
        for ts, amps in manual_amp_sources:
            for t, a in zip(ts, amps):
                try:
                    tf = round(float(t), 6)
                    af = float(a)
                except Exception:
                    continue
                if np.isfinite(tf) and np.isfinite(af):
                    amp_lookup[tf] = af
        manual_amps_log = [float(amp_lookup.get(round(float(t), 6), np.nan)) for t in manual_times]
        review_rows.append(
            {
                "segment_name": seg_name,
                "segment_index": seg_idx,
                "afc_lower_left_value": float(dec.afc_lower_left_value),
                "afc_lower_right_value": float(dec.afc_lower_right_value),
                "afc_upper_left_value": float(dec.afc_upper_left_value),
                "afc_upper_right_value": float(dec.afc_upper_right_value),
                "x_start_s": float(dec.x_start_s),
                "x_end_s": float(dec.x_end_s),
                "manual_afc_times_s": ",".join(f"{x:.6f}" for x in manual_times),
                "manual_afc_amps": ",".join(
                    "" if not np.isfinite(float(x)) else f"{float(x):.6f}" for x in manual_amps_log
                ),
                "status": str(dec.status),
            }
        )

        if str(dec.status).lower() == "skipped":
            continue

        time_arr = np.asarray(seg.get("meta", {}).get("_time_plot", []), dtype=float)
        sig_arr = np.asarray(seg.get("meta", {}).get("_sig_plot", []), dtype=float)
        main_times, _ = extract_fixed_main_peaks_from_events(seg.get("events", pd.DataFrame()))

        assigned = assign_afc_events_to_previous_main_peak(manual_times, main_times)
        for t, pk_idx in zip(manual_times, assigned):
            amp = _nearest_signal_amplitude(time_arr, sig_arr, t)
            prom, width_s = _nearest_signal_peak_features(time_arr, sig_arr, t)
            event = AFCEvent(
                segment_name=seg_name,
                segment_index=seg_idx,
                main_peak_index=int(pk_idx),
                time_s=float(t),
                amplitude=float(amp) if np.isfinite(amp) else np.nan,
                source="manual_curated",
                prominence=float(prom) if np.isfinite(prom) else np.nan,
                width_s=float(width_s) if np.isfinite(width_s) else np.nan,
                review_id=str(seg_idx),
            )
            afc_events.append(event)
            per_segment_rows.setdefault(seg_idx, []).append(event.to_dict())

    for seg_idx, seg in seg_by_index.items():
        rows = per_segment_rows.get(int(seg_idx), [])
        if rows:
            afc_df = pd.DataFrame(rows).sort_values(["time_s"]).reset_index(drop=True)
        else:
            afc_df = pd.DataFrame(
                columns=[
                    "segment_name",
                    "segment_index",
                    "main_peak_index",
                    "time_s",
                    "amplitude",
                    "prominence",
                    "width_s",
                    "source",
                    "review_id",
                ]
            )
        seg["afc_events_df"] = afc_df

        main_df = seg.get("events", pd.DataFrame()).copy()
        if not afc_df.empty:
            afc_as_events = pd.DataFrame(
                {
                    "Time_s": pd.to_numeric(afc_df["time_s"], errors="coerce"),
                    "Type": "AFC",
                    "Amp": pd.to_numeric(afc_df["amplitude"], errors="coerce"),
                    "Prom": pd.to_numeric(afc_df.get("prominence", np.nan), errors="coerce"),
                    "Width_s": pd.to_numeric(afc_df.get("width_s", np.nan), errors="coerce"),
                    "Transient": np.nan,
                    "Source": afc_df["source"].astype(str),
                    "ReviewID": afc_df["review_id"].astype(str),
                    "MainPeakIndex": pd.to_numeric(afc_df["main_peak_index"], errors="coerce"),
                }
            )
            main_with_meta = main_df.copy()
            for col in ["Source", "ReviewID", "MainPeakIndex"]:
                if col not in main_with_meta.columns:
                    main_with_meta[col] = np.nan
            seg["events_with_afc"] = (
                pd.concat([main_with_meta, afc_as_events], ignore_index=True, sort=False)
                .sort_values("Time_s")
                .reset_index(drop=True)
            )
        else:
            seg["events_with_afc"] = main_df

    review_log_df = pd.DataFrame(review_rows)
    if not review_log_df.empty:
        review_log_df = review_log_df.sort_values(["segment_index"]).reset_index(drop=True)
    return merged_results, afc_events, review_log_df

def _analyze_prebuilt_signal(
    sig: np.ndarray,
    fs: float,
    t0: float,
    seg_meta: List[Tuple[int, int, int, float, int]],
    orient_meta: Dict,
    config: BeatCounterConfig,
    file_path: str,
    display_name: str,
    sheet_name: Optional[str],
    candidate_label: str,
    df_source: pd.DataFrame,
    y_cols_source: List[str],
    *,
    enable_close_peak_recovery: bool = True,
    debug_timing: bool = False,
) -> Tuple[float, int, pd.DataFrame, Dict]:
    empty = pd.DataFrame(
        columns=[
            "Time_s",
            "Type",
            "Amp",
            "Prom",
            "Width_s",
            "Transient",
            "RescueType",
            "RescueGapBefore_s",
            "RescueGapAfter_s",
            "RescueAmpRel",
            "RescuePromRel",
            "RescueWidthRel",
        ]
    )
    duration_s = float(sig.size / fs) if fs > 0 else 0.0
    time_origin_s = float(orient_meta.get("time_origin_s", t0))
    stitched_gap_ranges_raw = orient_meta.get("stitched_gap_ranges_samples", [])
    stitched_gap_ranges_samples: List[Tuple[int, int]] = []
    if isinstance(stitched_gap_ranges_raw, (list, tuple)):
        for item in stitched_gap_ranges_raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                s_i = int(item[0])
                e_i = int(item[1])
            except Exception:
                continue
            if e_i <= s_i:
                continue
            stitched_gap_ranges_samples.append((int(s_i), int(e_i)))
    meta = {
        "fs": fs,
        "t0": t0,
        "time_origin_s": float(time_origin_s),
        "duration_s": duration_s,
        "orientation": orient_meta,
        "sheet_name": sheet_name,
        "display_name": display_name,
        "file": os.path.basename(file_path),
        "candidate_label": candidate_label,
        "forced_invert": orient_meta.get("invert", None),
        "config": asdict(config),
        "_seg_meta_plot": seg_meta,
        "_peak_debug_rows": [],
        "n_main_primary": 0,
        "n_main_rescue": 0,
        "rescue_fraction": 0.0,
        "stitched_gap_ranges_samples": [[int(a), int(b)] for a, b in stitched_gap_ranges_samples],
        "stitched_gap_samples": int(sum(max(0, int(b) - int(a)) for a, b in stitched_gap_ranges_samples)),
    }
    timing: Optional[Dict[str, float]] = (
        {
            "drift_normalisation_s": 0.0,
            "raw_peak_detection_s": 0.0,
            "qc_s": 0.0,
            "dedup_s": 0.0,
            "rescue_s": 0.0,
        }
        if bool(debug_timing)
        else None
    )

    def _acc_t(name: str, t_start: float) -> None:
        if timing is None:
            return
        timing[name] = float(timing.get(name, 0.0) + (perf_counter() - float(t_start)))

    def _ret(
        bpm_out: float,
        n_main_out: int,
        events_out: pd.DataFrame,
    ) -> Tuple[float, int, pd.DataFrame, Dict]:
        if timing is not None:
            meta["timing"] = {str(k): float(v) for k, v in timing.items()}
        return float(bpm_out), int(n_main_out), events_out, meta
    if sig.size == 0:
        meta.update({"qc_pass": False, "qc_reason": "empty_signal", "hard_noise": True, "hard_reject_reason": "empty_signal", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "strong_thr": np.nan, "weak_thr": np.nan, "snr": np.nan})
        return _ret(0.0, 0, empty)

    sig_raw = np.asarray(sig, dtype=float)
    sig_work = sig_raw.copy()

    vert_meta = detect_vertical_line_artifacts(sig_work, config) if config.vertical_artifact_clean_enabled else {"centers": np.array([], dtype=int), "n_artifacts": 0, "jump_thr": 0.0}
    n_vert = int(vert_meta.get("n_artifacts", 0))
    n_vert_cleaned = 0
    vert_mode = "disabled"
    if n_vert > 0:
        if n_vert <= int(config.vertical_artifact_sparse_max_count):
            sig_work, touched = suppress_vertical_line_artifacts(sig_work, np.asarray(vert_meta.get("centers", np.array([], dtype=int))), halfwin=int(config.vertical_artifact_interp_halfwin))
            n_vert_cleaned = int(np.sum(touched))
            vert_mode = "cleaned_sparse"
        else:
            vert_mode = "dense_detected"

    pre_features = compute_sheet_structure_features(
        df_source,
        y_cols_source,
        fs,
        bool(orient_meta.get("invert", True)),
        orient_meta,
        np.asarray(sig_work, dtype=float),
        config,
    )
    t_drift = perf_counter()
    sig, drift_meta = normalize_slow_trend(
        np.asarray(sig_work, dtype=float),
        fs,
        config,
        strength=float(config.drift_strength),
        min_ratio=float(config.drift_apply_min_ratio),
        row_corr=float(pre_features.get("corr_med", np.nan)),
        force=False,
    )
    _acc_t("drift_normalisation_s", t_drift)
    time = time_origin_s + np.arange(sig.size) / fs

    snr, noise_mad, signal_mad = compute_global_snr(sig)
    t_qc = perf_counter()
    qc_features = compute_sheet_structure_features(df_source, y_cols_source, fs, bool(orient_meta.get("invert", True)), orient_meta, sig, config)
    qc_features["snr"] = snr
    qc_pass, qc_reason = evaluate_segment_qc(qc_features, n_vert, config)
    _acc_t("qc_s", t_qc)

    meta.update({
        "snr": float(snr),
        "noise_mad": float(noise_mad),
        "signal_mad": float(signal_mad),
        "sheet_qc": qc_features,
        "qc_pass": bool(qc_pass),
        "qc_reason": str(qc_reason),
        "hard_noise": not bool(qc_pass),
        "hard_reject_reason": None if qc_pass else str(qc_reason),
        "n_vertical_artifacts": int(n_vert),
        "n_vertical_cleaned": int(n_vert_cleaned),
        "vertical_artifact_mode": vert_mode,
        "drift_norm": drift_meta,
        "n_rows": int(qc_features.get("n_rows", 0)),
    })

    if not qc_pass:
        meta.update({"n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "strong_thr": np.nan, "weak_thr": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
        return _ret(0.0, 0, empty)

    trans_id = build_transient_id_vector(sig.size, seg_meta)

    t_raw = perf_counter()
    raw_peaks, proms, widths_s, peak_tid = detect_raw_peaks_transientwise(sig=sig, seg_meta=seg_meta, fs=fs, config=config)
    _acc_t("raw_peak_detection_s", t_raw)
    if raw_peaks.size == 0:
        meta.update({"qc_pass": False, "qc_reason": "no_detectable_peaks", "hard_noise": True, "hard_reject_reason": "no_detectable_peaks", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "strong_thr": np.nan, "weak_thr": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
        return _ret(0.0, 0, empty)

    seg_name_for_debug = str(sheet_name or display_name)
    seg_idx_for_debug = _infer_segment_index_from_name(sheet_name or display_name)
    raw_peaks_all = np.asarray(raw_peaks, dtype=int).copy()
    raw_proms_all = np.asarray(proms, dtype=float).copy()
    raw_widths_all = np.asarray(widths_s, dtype=float).copy()
    raw_tids_all = np.asarray(peak_tid, dtype=int).copy()
    raw_survivor_set: Set[int] = set(int(x) for x in raw_peaks_all.tolist())
    main_candidate_set: Set[int] = set()
    main_after_dedup_set: Set[int] = set()
    main_after_short_gap_set: Set[int] = set()
    main_after_local_set: Set[int] = set()
    main_after_interbeat_set: Set[int] = set()
    final_main_set: Set[int] = set()
    rescue_set: Set[int] = set()
    removed_by_rescue_set: Set[int] = set()
    promoted_gap_set: Set[int] = set()
    promoted_transient_set: Set[int] = set()
    rejected_in_stitched_gap_set: Set[int] = set()
    strong_thr_debug = np.nan

    def _store_peak_debug_rows() -> None:
        rows = _build_peak_debug_rows(
            segment_name=seg_name_for_debug,
            segment_index=seg_idx_for_debug,
            time=time,
            sig=sig,
            raw_peaks_all=raw_peaks_all,
            raw_proms_all=raw_proms_all,
            raw_widths_all=raw_widths_all,
            raw_tids_all=raw_tids_all,
            raw_survivors=raw_survivor_set,
            main_candidates=main_candidate_set,
            main_after_dedup=main_after_dedup_set,
            main_after_short_gap=main_after_short_gap_set,
            main_after_local=main_after_local_set,
            main_after_interbeat=main_after_interbeat_set,
            final_main=final_main_set,
            rescue=rescue_set,
            removed_by_rescue=removed_by_rescue_set,
            promoted_gap=promoted_gap_set,
            promoted_transient=promoted_transient_set,
            rejected_in_stitched_gap=rejected_in_stitched_gap_set,
            strong_thr=strong_thr_debug,
        )
        meta["_peak_debug_rows"] = rows
        if rows:
            df_rows = pd.DataFrame(rows)
            label_counts = df_rows["final_label"].value_counts(dropna=False).to_dict()
            meta["_peak_debug_counts"] = {str(k): int(v) for k, v in label_counts.items()}
        else:
            meta["_peak_debug_counts"] = {}

    if config.main_spike_filter_enabled and (snr >= float(config.main_spike_filter_min_snr) or n_vert > 0):
        raw_peaks, proms, widths_s, n_spike = filter_needle_spike_peaks(sig, raw_peaks, proms, widths_s, config)
        peak_tid = trans_id[raw_peaks]
        raw_after_spike = set(int(x) for x in raw_peaks.tolist())
        raw_survivor_set = set(raw_after_spike)
        meta["n_main_spike_filtered"] = int(n_spike)
        if raw_peaks.size == 0:
            _store_peak_debug_rows()
            meta.update({"qc_pass": False, "qc_reason": "all_peaks_filtered_as_spikes", "hard_noise": True, "hard_reject_reason": "all_peaks_filtered_as_spikes", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "strong_thr": np.nan, "weak_thr": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
            return _ret(0.0, 0, empty)

    discont_meta = detect_discontinuity_artifact_centers(sig=sig, config=config)
    discont_centers = np.asarray(discont_meta.get("centers", np.array([], dtype=int)), dtype=int)
    discont_mask = np.zeros(sig.size, dtype=bool)
    discont_guard = max(1, int(round(float(config.discontinuity_peak_exclusion_s) * fs)))
    for c in discont_centers:
        l = max(0, int(c) - discont_guard)
        r = min(sig.size, int(c) + discont_guard + 1)
        discont_mask[l:r] = True
    keep = ~discont_mask[raw_peaks]
    n_artifact_filtered = int(np.sum(~keep))
    if n_artifact_filtered > 0:
        raw_peaks = raw_peaks[keep]
        proms = proms[keep]
        widths_s = widths_s[keep]
        peak_tid = peak_tid[keep]
    raw_survivor_set = set(int(x) for x in raw_peaks.tolist())
    if stitched_gap_ranges_samples:
        stitched_gap_mask = np.zeros(sig.size, dtype=bool)
        for s_i, e_i in stitched_gap_ranges_samples:
            l = max(0, int(s_i))
            r = min(sig.size, int(e_i))
            if r > l:
                stitched_gap_mask[l:r] = True
        if raw_peaks.size > 0:
            keep_stitched = ~stitched_gap_mask[raw_peaks]
            n_stitched_gap_filtered = int(np.sum(~keep_stitched))
            if n_stitched_gap_filtered > 0:
                rejected_in_stitched_gap_set = set(int(x) for x in raw_peaks[~keep_stitched].tolist())
                raw_peaks = raw_peaks[keep_stitched]
                proms = proms[keep_stitched]
                widths_s = widths_s[keep_stitched]
                peak_tid = peak_tid[keep_stitched]
                raw_survivor_set = set(int(x) for x in raw_peaks.tolist())
            meta["n_stitched_gap_filtered_peaks"] = int(n_stitched_gap_filtered)
        else:
            meta["n_stitched_gap_filtered_peaks"] = 0
    else:
        meta["n_stitched_gap_filtered_peaks"] = 0
    meta.update({
        "n_discontinuity_centers": int(discont_centers.size),
        "discontinuity_jump_thr": float(discont_meta.get("jump_thr", 0.0)),
        "n_artifact_filtered_peaks": int(n_artifact_filtered),
    })
    if raw_peaks.size == 0:
        _store_peak_debug_rows()
        meta.update({"qc_pass": False, "qc_reason": "all_peaks_filtered_as_artifacts", "hard_noise": True, "hard_reject_reason": "all_peaks_filtered_as_artifacts", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "strong_thr": np.nan, "weak_thr": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
        return _ret(0.0, 0, empty)

    strong_thr, weak_thr, thr_meta = compute_prominence_thresholds(proms, config)
    weak_thr = max(float(weak_thr), float(config.min_secondary_prom), 3.5 * float(noise_mad))
    strong_thr = max(float(strong_thr), float(config.min_main_prom), 4.5 * float(noise_mad))
    drift_ratio = float(meta.get("drift_norm", {}).get("drift_ratio", 0.0))
    if bool(meta.get("drift_norm", {}).get("applied", False)) and np.isfinite(drift_ratio) and drift_ratio > 0:
        strong_sens_factor = max(0.75, 1.0 - 0.18 * float(drift_ratio))
        weak_sens_factor = max(0.80, 1.0 - 0.12 * float(drift_ratio))
        strong_thr *= float(strong_sens_factor)
        weak_thr *= float(weak_sens_factor)
        meta["detrend_main_sensitivity_factor"] = float(strong_sens_factor)
    strong_thr_debug = float(strong_thr)

    strong_mask = proms >= strong_thr

    meta.update({"strong_thr": float(strong_thr), "weak_thr": float(weak_thr), "threshold_meta": thr_meta, "n_raw_peaks": int(raw_peaks.size)})

    if np.sum(strong_mask) == 0 and bool(config.use_narrow_retry):
        retry_thr = max(float(config.min_main_prom), float(np.quantile(proms, 0.85)) * 0.45, 3.8 * float(noise_mad))
        strong_thr = float(retry_thr)
        strong_mask = proms >= strong_thr
        meta["strong_thr_retry"] = float(strong_thr)
        meta["strong_thr"] = float(strong_thr)
        strong_thr_debug = float(strong_thr)

    # Restored baseline detector path.
    if np.sum(strong_mask) == 0:
        _store_peak_debug_rows()
        meta.update({"qc_pass": False, "qc_reason": "no_main_candidates", "hard_noise": True, "hard_reject_reason": "no_main_candidates", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
        return _ret(0.0, 0, empty)

    main_candidate_set = set(int(x) for x in raw_peaks[strong_mask].tolist())
    main_peaks = raw_peaks[strong_mask]
    main_proms = proms[strong_mask]
    main_widths = widths_s[strong_mask]
    main_tids = peak_tid[strong_mask]
    t_dedup = perf_counter()
    main_peaks, main_proms, main_widths, main_tids = deduplicate_main_candidates(
        sig=sig,
        peaks=main_peaks,
        proms=main_proms,
        widths_s=main_widths,
        tids=main_tids,
        fs=fs,
        config=config,
    )
    _acc_t("dedup_s", t_dedup)
    main_after_dedup_set = set(int(x) for x in main_peaks.tolist())

    periodicity_score = float(qc_features.get("periodicity_score", 0.0))
    corr_med = float(qc_features.get("corr_med", np.nan))
    promoted = recover_missing_main_peaks_in_large_gaps(
        raw_peaks=raw_peaks,
        proms=proms,
        widths_s=widths_s,
        main_peaks=main_peaks,
        fs=fs,
        strong_thr=float(strong_thr),
        weak_thr=float(weak_thr),
        periodicity=periodicity_score,
        corr_med=corr_med,
        config=config,
    )
    promoted_close: List[int] = []
    if enable_close_peak_recovery:
        promoted_close = recover_close_second_lobe_main_candidates(
            raw_peaks=raw_peaks,
            proms=proms,
            widths_s=widths_s,
            main_peaks=main_peaks,
            main_proms=main_proms,
            sig=sig,
            fs=fs,
            strong_thr=float(strong_thr),
            weak_thr=float(weak_thr),
        )
    promoted_close_unique = sorted(set(int(x) for x in promoted_close) - set(int(x) for x in promoted))
    promoted_all = sorted(set(int(x) for x in promoted) | set(int(x) for x in promoted_close_unique))
    if promoted_all:
        promoted_gap_set = set(int(x) for x in promoted_all)
        main_set_aug = set(int(x) for x in main_peaks.tolist()) | set(int(x) for x in promoted_all)
        main_sorted = np.asarray(sorted(main_set_aug), dtype=int)
        idx_map = {int(pk): i for i, pk in enumerate(raw_peaks)}
        idx_sel = [idx_map[int(pk)] for pk in main_sorted if int(pk) in idx_map]
        idx_sel_arr = np.asarray(idx_sel, dtype=int)
        main_peaks = raw_peaks[idx_sel_arr]
        main_proms = proms[idx_sel_arr]
        main_widths = widths_s[idx_sel_arr]
        main_tids = peak_tid[idx_sel_arr]
        t_dedup = perf_counter()
        main_peaks, main_proms, main_widths, main_tids = deduplicate_main_candidates(
            sig=sig,
            peaks=main_peaks,
            proms=main_proms,
            widths_s=main_widths,
            tids=main_tids,
            fs=fs,
            config=config,
        )
        _acc_t("dedup_s", t_dedup)
    else:
        promoted_gap_set = set()
    meta["n_gap_promoted"] = int(len(promoted))
    meta["n_close_second_lobe_promoted"] = int(len(promoted_close_unique))

    main_before_refine_set = set(int(x) for x in main_peaks.tolist())
    refined_main, trans_refine_meta = refine_main_peaks_by_transient_coherence(
        raw_peaks=raw_peaks,
        proms=proms,
        widths_s=widths_s,
        peak_tids=peak_tid,
        main_peaks=main_peaks,
        sig=sig,
        fs=fs,
        strong_thr=float(strong_thr),
        periodicity=periodicity_score,
        corr_med=corr_med,
        seg_meta=seg_meta,
        config=config,
    )
    if refined_main:
        idx_map = {int(pk): i for i, pk in enumerate(raw_peaks)}
        idx_sel = [idx_map[int(pk)] for pk in refined_main if int(pk) in idx_map]
        idx_sel_arr = np.asarray(idx_sel, dtype=int)
        main_peaks = raw_peaks[idx_sel_arr]
        main_proms = proms[idx_sel_arr]
        main_widths = widths_s[idx_sel_arr]
        main_tids = peak_tid[idx_sel_arr]
        order = np.argsort(main_peaks)
        main_peaks = main_peaks[order]
        main_proms = main_proms[order]
        main_widths = main_widths[order]
        main_tids = main_tids[order]
    main_after_refine_set = set(int(x) for x in main_peaks.tolist())
    promoted_transient_set = main_after_refine_set - main_before_refine_set
    # Stage 3/4 consolidated close-peak handling:
    # primary refractory-window dedup + strict true double-lobe exception.
    t_dedup = perf_counter()
    main_peaks, main_proms, main_widths, main_tids = deduplicate_main_candidates(
        sig=sig,
        peaks=main_peaks,
        proms=main_proms,
        widths_s=main_widths,
        tids=main_tids,
        fs=fs,
        config=config,
    )
    _acc_t("dedup_s", t_dedup)
    main_after_dedup_set = set(int(x) for x in main_peaks.tolist())
    meta.update(trans_refine_meta)
    main_after_short_gap_set = set(int(x) for x in main_peaks.tolist())
    main_after_local_set = set(int(x) for x in main_peaks.tolist())
    main_after_interbeat_set = set(int(x) for x in main_peaks.tolist())
    meta["n_short_gap_weak_removed"] = 0
    meta["n_local_weak_removed"] = 0
    meta["n_interbeat_tiny_removed"] = 0

    n_trans = max(len(seg_meta), 1)
    if main_peaks.size > float(config.max_main_per_transient_ratio) * n_trans:
        final_main_set = set(int(x) for x in main_peaks.tolist())
        _store_peak_debug_rows()
        meta.update({"qc_pass": False, "qc_reason": "too_many_main_candidates_noise", "hard_noise": True, "hard_reject_reason": "too_many_main_candidates_noise", "n_main": 0, "n_main_primary": 0, "n_main_rescue": 0, "quality_score": -np.inf, "prom_snr": 0.0, "ibi_cv": np.nan, "_sig_plot": sig, "_time_plot": time, "_rescue_peaks_plot": []})
        return _ret(0.0, 0, empty)

    t_rescue = perf_counter()
    rescue_peaks, rescue_meta = rescue_boundary_split_main_peaks(
        sig=sig,
        fs=fs,
        seg_meta=seg_meta,
        main_peaks=main_peaks,
        main_proms=main_proms,
        main_widths=main_widths,
        raw_proms=proms,
        config=config,
    )
    meta.update(rescue_meta)
    rescue_peaks_arr = np.asarray(sorted(set(int(x) for x in rescue_peaks)), dtype=int)
    rescue_event_audit = rescue_meta.get("rescue_event_audit", [])
    rescue_audit_by_peak = {
        int(row.get("peak_index")): dict(row)
        for row in (rescue_event_audit if isinstance(rescue_event_audit, list) else [])
        if isinstance(row, dict) and np.isfinite(float(row.get("peak_index", np.nan)))
    }
    rescue_set = set(int(x) for x in rescue_peaks_arr.tolist())
    main_before_rescue_prune_set = set(int(x) for x in main_peaks.tolist())
    main_peaks, main_proms, main_widths, main_tids, n_rescue_replaced = prune_weak_primary_near_rescue(
        sig=sig,
        fs=fs,
        main_peaks=main_peaks,
        main_proms=main_proms,
        main_widths=main_widths,
        main_tids=main_tids,
        rescue_peaks=rescue_peaks_arr,
        config=config,
    )
    meta["n_rescue_replaced_primary"] = int(n_rescue_replaced)
    main_after_rescue_prune_set = set(int(x) for x in main_peaks.tolist())
    removed_by_rescue_set = main_before_rescue_prune_set - main_after_rescue_prune_set
    _acc_t("rescue_s", t_rescue)
    t_dedup = perf_counter()
    main_peaks, main_proms, main_widths, main_tids = deduplicate_main_candidates(
        sig=sig,
        peaks=main_peaks,
        proms=main_proms,
        widths_s=main_widths,
        tids=main_tids,
        fs=fs,
        config=config,
    )
    _acc_t("dedup_s", t_dedup)
    rescue_set = set(int(x) for x in rescue_peaks_arr.tolist())
    final_main_set = set(int(x) for x in main_peaks.tolist())
    _store_peak_debug_rows()

    raw_idx_map = {int(pk): i for i, pk in enumerate(raw_peaks.tolist())}
    rescue_peak_tid_map: Dict[int, int] = {}
    rescue_peak_prom_map: Dict[int, float] = {}
    rescue_peak_width_map: Dict[int, float] = {}
    for rp in rescue_peaks_arr.tolist():
        ridx = raw_idx_map.get(int(rp), None)
        if ridx is None:
            continue
        rescue_peak_tid_map[int(rp)] = int(peak_tid[ridx])
        rescue_peak_prom_map[int(rp)] = float(proms[ridx])
        rescue_peak_width_map[int(rp)] = float(widths_s[ridx])

    events = build_events_dataframe(
        time=time,
        sig=sig,
        main_peaks=main_peaks,
        main_proms=main_proms,
        main_widths=main_widths,
        main_tids=main_tids,
        rescue_peaks=rescue_peaks_arr,
        rescue_audit_by_peak=rescue_audit_by_peak,
        rescue_peak_tid_map=rescue_peak_tid_map,
        rescue_peak_prom_map=rescue_peak_prom_map,
        rescue_peak_width_map=rescue_peak_width_map,
    )

    primary_main = int((events.Type == "Main Beat").sum()) if not events.empty else 0
    total_main = int(len(set(int(x) for x in main_peaks.tolist()) | set(int(x) for x in rescue_peaks_arr.tolist())))
    bpm = float((total_main / duration_s) * 60.0) if duration_s > 0 else 0.0
    prom_snr = float(np.median(main_proms) / max(float(noise_mad), 1e-12)) if main_proms.size else 0.0
    all_main_for_cv = np.asarray(sorted(set(int(x) for x in main_peaks.tolist()) | set(int(x) for x in rescue_peaks_arr.tolist())), dtype=int)
    ibi_cv = _compute_ibi_cv(all_main_for_cv, fs)
    q_score = _quality_score(prom_snr, total_main, ibi_cv)

    corr_med = float(qc_features.get("corr_med", np.nan))
    periodicity = float(qc_features.get("periodicity_score", 0.0))
    row_transition = float(qc_features.get("orientation_transition_score", 0.0))
    minor_orient = min(
        float(qc_features.get("orientation_down_fraction", 0.0)),
        float(qc_features.get("orientation_up_fraction", 0.0)),
    )
    flip_transition = float(qc_features.get("mixed_direction_flip_transition", 0.0))
    flip_spread = float(qc_features.get("mixed_direction_flip_spread", 0.0))
    main_amp_med = float(np.median(np.asarray(sig, dtype=float)[main_peaks])) if main_peaks.size else np.nan
    main_prom_med = float(np.median(main_proms)) if main_proms.size else np.nan
    if (
        total_main >= 8
        and np.isfinite(main_amp_med)
        and np.isfinite(main_prom_med)
        and main_amp_med < 0.060
        and main_prom_med < 0.040
        and periodicity < 0.35
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "low_amplitude_dense_noise",
            "hard_noise": True,
            "hard_reject_reason": "low_amplitude_dense_noise",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    if (
        total_main <= 4
        and periodicity >= 0.55
        and minor_orient >= 0.15
        and row_transition >= 0.35
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "mixed_orientation_inconsistent_morphology",
            "hard_noise": True,
            "hard_reject_reason": "mixed_orientation_inconsistent_morphology",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    if (
        total_main >= 8
        and periodicity >= 0.60
        and np.isfinite(corr_med)
        and corr_med >= 0.05
        and row_transition >= 0.50
        and 0.22 <= minor_orient <= 0.45
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "mixed_orientation_inconsistent_morphology",
            "hard_noise": True,
            "hard_reject_reason": "mixed_orientation_inconsistent_morphology",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    if (
        total_main >= 10
        and periodicity >= 0.60
        and flip_transition >= 0.34
        and flip_spread >= 0.11
        and float(qc_features.get("mixed_direction_flip_rel_strength", 0.0)) >= 0.65
        and minor_orient >= 0.18
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "mixed_orientation_inconsistent_morphology",
            "hard_noise": True,
            "hard_reject_reason": "mixed_orientation_inconsistent_morphology",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    # Use rescue-only events for irregular-noise gating to avoid inflating rescue ratio
    # when a boundary marker overlaps an already-accepted main peak.
    rescue_only_set = set(int(x) for x in rescue_peaks_arr.tolist()) - set(int(x) for x in main_peaks.tolist())
    rescue_ratio = float(len(rescue_only_set) / max(total_main, 1))
    if (
        total_main >= 18
        and rescue_ratio >= 0.22
        and np.isfinite(ibi_cv)
        and float(ibi_cv) >= 0.55
        and periodicity <= 0.52
        and row_transition <= 0.18
        and (
            minor_orient >= 0.10
            and (np.isfinite(corr_med) and corr_med <= 0.70)
        )
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "incoherent_irregular_high_cv_noise",
            "hard_noise": True,
            "hard_reject_reason": "incoherent_irregular_high_cv_noise",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    if (
        total_main >= 10
        and np.isfinite(ibi_cv)
        and float(ibi_cv) >= 0.75
        and 0.22 <= periodicity <= 0.28
        and np.isfinite(corr_med)
        and corr_med <= 0.20
        and np.isfinite(main_prom_med)
        and float(main_prom_med) < 0.08
    ):
        meta.update({
            "qc_pass": False,
            "qc_reason": "incoherent_irregular_high_cv_noise",
            "hard_noise": True,
            "hard_reject_reason": "incoherent_irregular_high_cv_noise",
            "n_main": 0,
            "n_main_primary": 0,
            "n_main_rescue": 0,
            "prom_snr": float(prom_snr),
            "ibi_cv": float(ibi_cv),
            "quality_score": float(q_score),
            "_sig_plot": sig,
            "_time_plot": time,
            "_rescue_peaks_plot": [],
            "_rescue_times_s": [],
        })
        return _ret(0.0, 0, empty)

    meta.update({
        "n_main": int(total_main),
        "n_main_primary": int(primary_main),
        "n_main_rescue": int(rescue_peaks_arr.size),
        "rescue_fraction": float(rescue_peaks_arr.size / max(total_main, 1)),
        "bpm_file_duration": float(bpm),
        "prom_snr": float(prom_snr),
        "ibi_cv": float(ibi_cv) if np.isfinite(ibi_cv) else np.nan,
        "quality_score": float(q_score),
        "_rescue_peaks_plot": rescue_peaks_arr.astype(int).tolist(),
        "_rescue_times_s": [float(time[int(pk)]) for pk in rescue_peaks_arr.tolist()],
        "_rescue_event_audit": list(rescue_event_audit) if isinstance(rescue_event_audit, list) else [],
        "_sig_plot": sig,
        "_time_plot": time,
        "main_only_pipeline": True,
    })
    return _ret(bpm, total_main, events)

def rescue_candidate_is_plausible(meta: Dict) -> bool:
    if not bool(meta.get("qc_pass", False)):
        return False
    if int(meta.get("n_main", 0)) <= 0:
        return False
    if float(meta.get("snr", 0.0)) < 1.8:
        return False
    return True


def _candidate_orientation_sanity_score(
    result: Tuple[float, int, pd.DataFrame, Dict],
    config: BeatCounterConfig,
) -> Tuple[float, Dict[str, float]]:
    _, n_main_res, events_res, meta_res = result
    if not bool(meta_res.get("qc_pass", False)):
        return -np.inf, {"rescue_ratio": np.nan, "prom_med": np.nan, "amp_med": np.nan, "promoted_fail": np.nan}

    q = float(meta_res.get("quality_score", -np.inf))
    n_main = int(meta_res.get("n_main", n_main_res))
    n_rescue = int(meta_res.get("n_main_rescue", 0))
    rescue_ratio = float(n_rescue / max(n_main, 1))
    legacy_fail_key = "n_" + "promoted" + "_main" + "_floor_failed"
    promoted_fail = float(meta_res.get(legacy_fail_key, 0))

    prom_med = np.nan
    amp_med = np.nan
    if isinstance(events_res, pd.DataFrame) and (not events_res.empty):
        m = events_res.copy()
        if "Type" in m.columns:
            m = m[m["Type"].astype(str) == "Main Beat"]
        if not m.empty:
            prom_med = float(pd.to_numeric(m.get("Prom", np.nan), errors="coerce").dropna().median()) if "Prom" in m.columns else np.nan
            amp_med = float(pd.to_numeric(m.get("Amp", np.nan), errors="coerce").dropna().median()) if "Amp" in m.columns else np.nan

    score = float(q)
    if np.isfinite(prom_med):
        score += 0.35 * float(prom_med)
    if np.isfinite(amp_med):
        score += 0.20 * float(amp_med)
    score -= float(config.orientation_sanity_rescue_ratio_penalty) * rescue_ratio
    score -= float(config.orientation_sanity_promoted_fail_penalty) * promoted_fail

    return score, {
        "rescue_ratio": rescue_ratio,
        "prom_med": float(prom_med) if np.isfinite(prom_med) else np.nan,
        "amp_med": float(amp_med) if np.isfinite(amp_med) else np.nan,
        "promoted_fail": float(promoted_fail),
    }

def count_main_beats_from_excel(
    file_path: str,
    *,
    sheet_name: Optional[str] = None,
    config: BeatCounterConfig = BeatCounterConfig(),
    show_plot: bool = True,
    debug: bool = False,
    xl: Optional[pd.ExcelFile] = None,
) -> Tuple[float, int, pd.DataFrame, Dict]:
    if debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    timing: Optional[Dict[str, float]] = (
        {
            "sheet_load_s": 0.0,
            "stitching_s": 0.0,
        }
        if bool(debug)
        else None
    )
    excel_io = xl if xl is not None else file_path
    t_load = perf_counter()
    df, y_cols, fs, t0 = load_cytocypher_excel(excel_io, sheet_name=sheet_name)
    if timing is not None:
        timing["sheet_load_s"] = float(perf_counter() - t_load)
    sample_id = np.nan
    sample_id_col = next((c for c in df.columns if str(c).strip().lower() == "sample id"), None)
    if sample_id_col is not None:
        sample_series = df[sample_id_col].dropna()
        if not sample_series.empty:
            sample_id = sample_series.iloc[0]
    display_name = str(sheet_name) if sheet_name else os.path.basename(file_path)

    def cand_rank(meta_i: Dict) -> Tuple[int, float, int]:
        qc_i = 1 if bool(meta_i.get("qc_pass", False)) else 0
        q_i = float(meta_i.get("quality_score", -np.inf))
        if not np.isfinite(q_i):
            q_i = -np.inf
        plaus_i = 1 if rescue_candidate_is_plausible(meta_i) else 0
        return qc_i, q_i, plaus_i

    candidate_specs: List[Tuple[str, bool]] = [
        ("invert_false", False),
        ("invert_true", True),
    ]
    candidates: List[Dict] = []
    for cand_label, cand_invert in candidate_specs:
        t_stitch = perf_counter()
        sig_c, seg_meta_c, orient_c = build_concatenated_signal(df, y_cols, fs, config, force_invert=cand_invert)
        if timing is not None:
            timing["stitching_s"] = float(timing.get("stitching_s", 0.0) + (perf_counter() - t_stitch))
        cand_result = _analyze_prebuilt_signal(
            sig=sig_c,
            fs=fs,
            t0=t0,
            seg_meta=seg_meta_c,
            orient_meta=orient_c,
            config=config,
            file_path=file_path,
            display_name=display_name,
            sheet_name=sheet_name,
            candidate_label=cand_label,
            df_source=df,
            y_cols_source=y_cols,
            enable_close_peak_recovery=False,
            debug_timing=bool(debug),
        )
        cand_meta = dict(cand_result[3] or {})
        candidates.append(
            {
                "label": cand_label,
                "invert": bool(cand_invert),
                "sig": sig_c,
                "seg_meta": seg_meta_c,
                "orient_meta": orient_c,
                "result": cand_result,
                "rank": cand_rank(cand_meta),
            }
        )

    best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["rank"])
    best_candidate = candidates[best_idx]
    selected_label = str(best_candidate["label"])
    selected_rank = tuple(best_candidate["rank"])

    bpm, n_main, events, meta = _analyze_prebuilt_signal(
        sig=np.asarray(best_candidate["sig"], dtype=float),
        fs=fs,
        t0=t0,
        seg_meta=best_candidate["seg_meta"],
        orient_meta=best_candidate["orient_meta"],
        config=config,
        file_path=file_path,
        display_name=display_name,
        sheet_name=sheet_name,
        candidate_label=selected_label,
        df_source=df,
        y_cols_source=y_cols,
        enable_close_peak_recovery=True,
        debug_timing=bool(debug),
    )
    tested_labels = [str(c["label"]) for c in candidates]
    candidate_metrics: Dict[str, Dict] = {}
    candidate_ranks: Dict[str, Tuple[int, float, int]] = {}
    for cand in candidates:
        _, _, _, cand_meta = cand["result"]
        label = str(cand["label"])
        candidate_metrics[label] = {
            "qc_pass": bool(cand_meta.get("qc_pass", False)),
            "qc_reason": str(cand_meta.get("qc_reason", "")),
            "quality_score": float(cand_meta.get("quality_score", np.nan)),
            "n_main_primary": int(cand_meta.get("n_main_primary", 0)),
            "n_main_rescue": int(cand_meta.get("n_main_rescue", 0)),
            "periodicity": float(cand_meta.get("periodicity", np.nan)),
            "ibi_cv": float(cand_meta.get("ibi_cv", np.nan)),
            "prom_snr": float(cand_meta.get("prom_snr", np.nan)),
        }
        candidate_ranks[label] = tuple(cand["rank"])
    meta["candidate_tested"] = tested_labels
    meta["candidate_selected"] = selected_label
    meta["candidate_selection_rank"] = selected_rank
    meta["candidate_selection_reason"] = f"max_rank={selected_rank}"
    meta["candidate_rankings"] = candidate_ranks
    meta["candidate_metrics"] = candidate_metrics
    meta["sample_id"] = sample_id
    if timing is not None:
        timing_out = dict(meta.get("timing", {}) if isinstance(meta.get("timing"), dict) else {})
        for k, v in timing.items():
            timing_out[str(k)] = float(v)
        meta["timing"] = timing_out

    if show_plot:
        plot_sig = np.asarray(meta.get("_sig_plot", np.array([], dtype=float)), dtype=float)
        plot_time = np.asarray(meta.get("_time_plot", np.array([], dtype=float)), dtype=float)
        if plot_sig.size == 0 or plot_time.size != plot_sig.size:
            plot_sig = np.asarray(best_candidate["sig"], dtype=float)
            plot_time = float(best_candidate["orient_meta"].get("time_origin_s", t0)) + np.arange(plot_sig.size) / fs
        plot_events(
            time=plot_time,
            sig=plot_sig,
            events=events,
            file_name=display_name,
            bpm=bpm,
            n_main=n_main,
            n_rescue=int(meta.get("n_main_rescue", 0)),
            rescue_peaks=meta.get("_rescue_peaks_plot", []),
            snr=float(meta.get("snr", np.nan)),
            strong_thr=float(meta.get("strong_thr", np.nan)),
            weak_thr=float(meta.get("weak_thr", np.nan)),
            config=config,
            show=True,
            return_fig=False,
            meta=meta,
        )

    return bpm, n_main, events, meta


