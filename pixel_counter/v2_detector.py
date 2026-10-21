"""Experimental v2 contraction detector (relative-prominence cycles + robust amplitude reference).

Input is the baseline-relative contraction amplitude A(t) (contractions upward, resting ~0)
produced by :mod:`pixel_counter.v2_signal`.  The detector never sees raw units or polarity.

Definitions (all on the lightly smoothed waveform  Â = G_sigma * A, sigma = 12 ms)
---------------------------------------------------------------------------------
candidate    local maximum of Â whose prominence is >= 4 sigma_noise
H_i          height of candidate i above the resting baseline, H_i = Â(t_i)
P_i          topographic prominence of candidate i (drop to the higher of the two
             saddles that connect it to a taller peak, or to the record boundary)
recovery r_i P_i / H_i   (fraction of its height the waveform recovers on BOTH sides)

independent  r_i >= rho     (min_rel_recovery)   -> the candidate is a separate mechanical cycle
             otherwise it is a shoulder / secondary oscillation of a larger neighbour, a
             duplicate maximum inside one contraction, or an incomplete excursion.
A_ref        90th percentile of H over the independent candidates whose H >= snr*sigma_noise
major        independent and H_i >= alpha * A_ref  (min_rel_amplitude)  and  H_i >= snr * sigma
event time   time-to-peak: the earliest local maximum of the cycle that reaches >= 90 % of the
             cycle maximum (cycle = region around the dominant maximum where Â stays above
             (1 - rho) * H); this prevents a narrow late ripple on a sustained contraction from
             hijacking the timestamp of the contraction.  The marker is finally snapped to the
             true local maximum of A within +-2.5 sigma.

Outcome per candidate: ACCEPTED or REJECTED (with a diagnostic reason).  No third class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, grey_opening, median_filter
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .v2_signal import estimate_noise_sigma, valid_blocks

# Internal numerical constants (documented; not user-facing strictness controls)
SMOOTH_SIGMA_S = 0.012     # Gaussian smoothing (sigma) before peak analysis
CANDIDATE_FLOOR_SNR = 4.0  # candidates need prominence >= 4 sigma_noise to be listed at all
REF_QUANTILE = 90.0        # amplitude reference = 90th percentile of independent candidates
TIME_TO_PEAK_FRACTION = 0.90
SNAP_SIGMA_MULT = 2.5

ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"

EVENT_COLUMNS = [
    "index", "time_rel_s", "amplitude", "height", "prominence", "recovery", "snr", "fwhm_s",
    "cycle_id", "status", "reason",
]



def separation_strictness_to_recovery(strictness: float) -> float:
    """Map user separation_strictness in [0.0, 2.0] continuously to min_rel_recovery in [0.02, 0.95]."""
    s = max(0.0, min(2.0, float(strictness)))
    return float(min(0.95, max(0.02, 0.50 * s)))


def amplitude_strictness_to_amplitude(strictness: float) -> float:
    """Map user amplitude_strictness in [0.0, 2.0] continuously to min_rel_amplitude in [0.001, 0.98]."""
    s = max(0.0, min(2.0, float(strictness)))
    return float(min(0.98, max(0.001, 0.50 * s)))


def strictness_to_params(
    separation_strictness: float = 1.0,
    amplitude_strictness: float = 1.0,
    min_snr: float = 8.0,
    enable_impulse_filter: bool = True,
) -> DetectorParams:
    """Construct DetectorParams from the two continuous user-facing strictness coefficients."""
    return DetectorParams(
        min_rel_amplitude=amplitude_strictness_to_amplitude(amplitude_strictness),
        min_rel_recovery=separation_strictness_to_recovery(separation_strictness),
        min_snr=min_snr,
        enable_impulse_filter=enable_impulse_filter,
        separation_strictness=float(separation_strictness),
        amplitude_strictness=float(amplitude_strictness),
    ).validate()


@dataclass(frozen=True)
class DetectorParams:
    """The user-adjustable strictness controls."""
    min_rel_amplitude: float = 0.50   # alpha: a major contraction is >= alpha * typical major amplitude
    min_rel_recovery: float = 0.50    # rho:   separate cycles must recover >= rho of their height
    min_snr: float = 8.0              # absolute floor: height >= min_snr * noise sigma
    enable_impulse_filter: bool = True
    separation_strictness: float = 1.0  # continuous separation strictness in [0.0, 2.0]
    amplitude_strictness: float = 1.0   # continuous amplitude strictness in [0.0, 2.0]

    def validate(self) -> "DetectorParams":
        if not (0.0 < self.min_rel_amplitude < 1.0):
            raise ValueError("min_rel_amplitude must be in (0, 1)")
        if not (0.0 < self.min_rel_recovery < 1.0):
            raise ValueError("min_rel_recovery must be in (0, 1)")
        if not (self.min_snr > 0.0):
            raise ValueError("min_snr must be > 0")
        if not (0.0 <= self.separation_strictness <= 2.0):
            raise ValueError("separation_strictness must be in [0.0, 2.0]")
        if not (0.0 <= self.amplitude_strictness <= 2.0):
            raise ValueError("amplitude_strictness must be in [0.0, 2.0]")
        return self


def params_from_config(config) -> DetectorParams:
    sep_s = getattr(config, "v2_separation_strictness", None)
    amp_s = getattr(config, "v2_amplitude_strictness", None)
    min_rec = float(getattr(config, "v2_min_rel_recovery", 0.50))
    min_amp = float(getattr(config, "v2_min_rel_amplitude", 0.50))

    if sep_s is not None and sep_s != 1.0:
        min_rec = separation_strictness_to_recovery(sep_s)
    else:
        sep_s = 1.0 if min_rec == 0.50 else min_rec / 0.50

    if amp_s is not None and amp_s != 1.0:
        min_amp = amplitude_strictness_to_amplitude(amp_s)
    else:
        amp_s = 1.0 if min_amp == 0.50 else min_amp / 0.50

    return DetectorParams(
        min_rel_amplitude=min_amp,
        min_rel_recovery=min_rec,
        min_snr=float(getattr(config, "v2_min_snr", 8.0)),
        enable_impulse_filter=bool(getattr(config, "v2_enable_impulse_filter", True)),
        separation_strictness=float(sep_s),
        amplitude_strictness=float(amp_s),
    ).validate()


def _empty_events() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in EVENT_COLUMNS})
    df["index"] = df["index"].astype(int)
    return df


def _trough_bounded_width(
    trace: np.ndarray, idx: int, level: float, fs: float, l_bound: int, r_bound: int
) -> tuple[float, int, int]:
    """Width (seconds) of excursion around ``idx`` above ``level``, bounded by local troughs."""
    l = idx
    while l > l_bound and np.isfinite(trace[l - 1]) and trace[l - 1] >= level:
        l -= 1
    r = idx
    while r < r_bound and np.isfinite(trace[r + 1]) and trace[r + 1] >= level:
        r += 1
    return (r - l + 1) / fs, l, r


def _refine_events_for_impulses(
    a: np.ndarray,
    fs: float,
    sn: float,
    ev: pd.DataFrame,
    params: DetectorParams,
    out: Dict,
) -> pd.DataFrame:
    """Two-stage morphological impulse & embedded-spike evaluation.

    Distinguishes:
      1. Isolated technical impulse artefacts (narrow trough-bounded width across W25/W50/W75,
         abrupt single-sample step jump, and lack of sustained 60 ms contraction body) ->
         rejected as ``isolated_impulse_artefact`` and excluded from ``reference_amplitude``.
      2. Vertical impulse spikes embedded inside genuine contractions (wide contraction base
         W25 >= 140 ms with razor-thin spike tip W75 <= 48 ms and large overshoot above the
         sustained 60 ms opening / 48 ms median body) -> contraction is preserved as ACCEPTED,
         its effective height in ``reference_amplitude`` is replaced by the sustained plateau
         height (preventing spike-induced threshold inflation), and its timestamp & amplitude
         are localized on the non-spike contraction plateau.
      3. Borderline post-contraction secondary excursions (within 8 % above threshold, occurring
         <= 360 ms after a >= 1.8x taller contraction with < 50 % median width and < 70 % recovery)
         -> rejected as ``borderline_secondary_excursion``.
    """
    nc = len(ev)
    if nc == 0 or not params.enable_impulse_filter:
        return ev

    n = a.size
    sigma_s = max(1.0, (0.035 * fs) / 2.355)
    k_open = max(5, int(round(0.060 * fs)) | 1)
    k_med = max(5, int(round(0.048 * fs)) | 1)
    amp_clean = np.where(np.isfinite(a), a, 0.0)
    sm35 = gaussian_filter1d(amp_clean, sigma_s)
    body60 = grey_opening(amp_clean, size=k_open)
    med48 = median_filter(amp_clean, size=k_med, mode="nearest")

    pks = ev["index"].to_numpy(dtype=int)
    w25_arr = np.zeros(nc)
    w50_arr = np.zeros(nc)
    w75_arr = np.zeros(nc)
    w50_raw_arr = np.zeros(nc)
    w75_raw_arr = np.zeros(nc)
    step1_arr = np.zeros(nc)
    b60_val_arr = np.zeros(nc)
    sust_val_arr = np.zeros(nc)
    sust_idx_arr = np.zeros(nc, dtype=int)

    for k in range(nc):
        idx = int(pks[k])
        lo_s = max(0, pks[k - 1] if k > 0 and 0 < idx - pks[k - 1] < int(0.4 * fs) else idx - int(0.25 * fs))
        hi_s = min(n - 1, pks[k + 1] if k + 1 < nc and 0 < pks[k + 1] - idx < int(0.4 * fs) else idx + int(0.25 * fs))
        lo_s = min(lo_s, idx)
        hi_s = max(hi_s, idx)
        l_tr = lo_s + int(np.argmin(sm35[lo_s : idx + 1]))
        r_tr = idx + int(np.argmin(sm35[idx : hi_s + 1]))

        h_sm = float(ev.iloc[k]["height"])
        a_raw = float(ev.iloc[k]["amplitude"])

        w25, l25, r25 = _trough_bounded_width(sm35, idx, 0.25 * h_sm, fs, l_tr, r_tr)
        w50, l50, r50 = _trough_bounded_width(sm35, idx, 0.50 * h_sm, fs, l_tr, r_tr)
        w75, _, _ = _trough_bounded_width(sm35, idx, 0.75 * h_sm, fs, l_tr, r_tr)
        w50_raw, _, _ = _trough_bounded_width(a, idx, 0.50 * a_raw, fs, l_tr, r_tr)
        w75_raw, _, _ = _trough_bounded_width(a, idx, 0.75 * a_raw, fs, l_tr, r_tr)

        w25_arr[k] = w25
        w50_arr[k] = w50
        w75_arr[k] = w75
        w50_raw_arr[k] = w50_raw
        w75_raw_arr[k] = w75_raw

        b_slice = body60[l25 : r25 + 1]
        m_slice = med48[l25 : r25 + 1]
        a_slice = a[l25 : r25 + 1]
        b60_val_arr[k] = float(np.max(b_slice)) if len(b_slice) else h_sm
        sust_trace = 0.5 * (b_slice + m_slice)
        sust_val = float(np.max(sust_trace)) if len(sust_trace) else h_sm
        sust_val_arr[k] = sust_val

        # Localize plateau timestamp on non-spike samples within [l25, r25]
        non_spike = np.where(np.isfinite(a_slice) & (a_slice <= 1.15 * sust_val))[0]
        if non_spike.size > 0:
            best_sub = int(non_spike[int(np.argmax(0.5 * b_slice[non_spike] + 0.5 * a_slice[non_spike]))])
            sust_idx_arr[k] = int(l25 + best_sub)
        else:
            sust_idx_arr[k] = int(l25 + int(np.argmax(b_slice))) if len(b_slice) else idx

        win_a = a[max(0, l50 - 2) : min(n, r50 + 3)]
        valid_win = win_a[np.isfinite(win_a)]
        step1_arr[k] = float(np.max(np.abs(np.diff(valid_win)))) / max(a_raw, 1e-9) if valid_win.size >= 2 else 0.0

    prelim_major = (ev["status"] == ACCEPTED).to_numpy()
    med_w50 = float(np.median(w50_arr[prelim_major])) if np.any(prelim_major) else float(np.median(w50_arr))

    is_isolated_impulse = np.zeros(nc, dtype=bool)
    is_embedded_spike = np.zeros(nc, dtype=bool)
    eff_heights = ev["height"].to_numpy(dtype=float).copy()

    for k in range(nc):
        a_raw = float(ev.iloc[k]["amplitude"])
        b60_ratio = b60_val_arr[k] / max(a_raw, 1e-9)

        iso = (
            (w50_raw_arr[k] <= 0.040 and w50_arr[k] <= 0.056 and w50_arr[k] < 0.55 * med_w50)
            or (step1_arr[k] >= 0.38 and w50_arr[k] < 0.60 * med_w50 and w75_arr[k] <= 0.048)
            or (
                w25_arr[k] <= 0.090
                and w50_arr[k] <= 0.064
                and w75_arr[k] <= 0.036
                and w50_arr[k] < 0.60 * med_w50
                and (step1_arr[k] >= 0.22 or b60_ratio <= 0.45)
            )
        )
        if not iso and w50_raw_arr[k] <= 0.048 and step1_arr[k] >= 0.30 and b60_ratio <= 0.50:
            for m in range(max(0, k - 2), min(nc, k + 3)):
                if m == k:
                    continue
                dt_m = abs(pks[m] - pks[k]) / fs
                if dt_m <= 0.22 and w50_raw_arr[m] >= 2.2 * w50_raw_arr[k] and b60_val_arr[m] >= 1.5 * b60_val_arr[k]:
                    iso = True
                    break

        emb = (
            (not iso)
            and w25_arr[k] >= 0.140
            and w50_raw_arr[k] <= 0.068
            and b60_ratio <= 0.58
            and (
                (w75_arr[k] <= 0.048 and (w75_arr[k] / max(w25_arr[k], 1e-9)) <= 0.25)
                or (w75_raw_arr[k] <= 0.024 and step1_arr[k] >= 0.25)
            )
        )
        is_isolated_impulse[k] = iso
        is_embedded_spike[k] = emb
        if emb:
            eff_heights[k] = sust_val_arr[k]

    if np.any(is_isolated_impulse & prelim_major) or np.any(is_embedded_spike & prelim_major):
        non_amp_reject_reasons = {
            "baseline_noise",
            "secondary_morphological_feature",
            "incomplete_excursion",
            "incomplete_boundary_excursion",
            "duplicate_peak_within_cycle",
        }
        eligible_for_ref = np.array(
            [
                (ev.iloc[k]["reason"] not in non_amp_reject_reasons) and (not is_isolated_impulse[k])
                for k in range(nc)
            ],
            dtype=bool,
        )
        if np.any(eligible_for_ref):
            new_ref_amp = float(np.percentile(eff_heights[eligible_for_ref], 80.0))
            new_amp_thr = max(params.min_rel_amplitude * new_ref_amp, params.min_snr * sn)
            out["reference_amplitude"] = new_ref_amp
            out["amplitude_threshold"] = new_amp_thr
        else:
            new_ref_amp = float("nan")
            new_amp_thr = float("inf")
            out["reference_amplitude"] = new_ref_amp
            out["amplitude_threshold"] = new_amp_thr

        half_snap = max(1, int(round(SNAP_SIGMA_MULT * SMOOTH_SIGMA_S * fs)))
        for k in range(nc):
            cur_status = ev.iloc[k]["status"]
            cur_reason = ev.iloc[k]["reason"]
            if is_isolated_impulse[k] and cur_status == ACCEPTED:
                ev.at[ev.index[k], "status"] = REJECTED
                ev.at[ev.index[k], "reason"] = "isolated_impulse_artefact"
            elif is_embedded_spike[k] and cur_status == ACCEPTED:
                new_idx = int(sust_idx_arr[k])
                ev.at[ev.index[k], "index"] = new_idx
                ev.at[ev.index[k], "time_rel_s"] = float(new_idx / fs)
                ev.at[ev.index[k], "amplitude"] = float(a[new_idx])
                ev.at[ev.index[k], "height"] = float(eff_heights[k])
                if float(eff_heights[k]) < new_amp_thr or float(a[new_idx]) < new_amp_thr:
                    ev.at[ev.index[k], "status"] = REJECTED
                    ev.at[ev.index[k], "reason"] = "insufficient_amplitude"
            elif cur_status == REJECTED and cur_reason == "insufficient_amplitude" and not is_isolated_impulse[k]:
                if eff_heights[k] >= new_amp_thr and w50_arr[k] >= 0.65 * med_w50:
                    pk_idx = int(ev.iloc[k]["index"])
                    lo = max(0, pk_idx - half_snap)
                    hi = min(n, pk_idx + half_snap + 1)
                    finite_sub = a[lo:hi]
                    if np.isfinite(finite_sub).any():
                        snapped_idx = lo + int(np.nanargmax(finite_sub))
                    else:
                        snapped_idx = pk_idx
                    if float(a[snapped_idx]) >= new_amp_thr:
                        ev.at[ev.index[k], "index"] = snapped_idx
                        ev.at[ev.index[k], "time_rel_s"] = float(snapped_idx / fs)
                        ev.at[ev.index[k], "amplitude"] = float(a[snapped_idx])
                        ev.at[ev.index[k], "status"] = ACCEPTED
                        ev.at[ev.index[k], "reason"] = ""

    # Check for borderline post-contraction secondary excursions near threshold (Problem E)
    cur_thr = float(out["amplitude_threshold"])
    acc_after = (ev["status"] == ACCEPTED).to_numpy()
    if np.any(acc_after) and np.isfinite(cur_thr):
        med_w50_acc = float(np.median(w50_arr[acc_after]))
        pks_cur = ev["index"].to_numpy(dtype=int)
        for k in range(1, nc):
            if ev.iloc[k]["status"] != ACCEPTED:
                continue
            prev_acc = [m for m in range(k) if ev.iloc[m]["status"] == ACCEPTED]
            if not prev_acc:
                continue
            m_prev = prev_acc[-1]
            dt_prev = (pks_cur[k] - pks_cur[m_prev]) / fs
            h_cur = float(ev.iloc[k]["height"])
            h_prev = float(ev.iloc[m_prev]["height"])
            rec_cur = float(ev.iloc[k]["recovery"])
            if (
                h_cur <= 1.08 * cur_thr
                and dt_prev <= 0.36
                and h_prev >= 1.80 * h_cur
                and w50_arr[k] < 0.50 * med_w50_acc
                and rec_cur < 0.70
            ):
                ev.at[ev.index[k], "status"] = REJECTED
                ev.at[ev.index[k], "reason"] = "borderline_secondary_excursion"

    # Enforce strict threshold consistency invariant: no accepted event may fall below cur_thr
    if np.isfinite(cur_thr):
        for k in range(nc):
            if ev.iloc[k]["status"] == ACCEPTED:
                if float(ev.iloc[k]["amplitude"]) < cur_thr or float(ev.iloc[k]["height"]) < cur_thr:
                    ev.at[ev.index[k], "status"] = REJECTED
                    ev.at[ev.index[k], "reason"] = "insufficient_amplitude"

    return ev


def detect_contractions(
    amplitude: np.ndarray,
    fs: float,
    params: Optional[DetectorParams] = None,
) -> Dict:
    """Detect contractions in a baseline-relative amplitude waveform.

    Returns {"events": DataFrame(EVENT_COLUMNS), "reference_amplitude", "noise_sigma",
             "amplitude_threshold", "params"}.  Events contain every listed candidate with
    status ACCEPTED / REJECTED; ``time_rel_s`` is index / fs from the first sample.
    """
    params = (params or DetectorParams()).validate()
    a = np.asarray(amplitude, dtype=float)
    out = {"events": _empty_events(), "reference_amplitude": float("nan"),
           "noise_sigma": float("nan"), "amplitude_threshold": float("nan"), "params": params}
    if a.size == 0 or not np.isfinite(fs) or fs <= 0 or not np.isfinite(a).any():
        return out
    sn = estimate_noise_sigma(a)
    out["noise_sigma"] = sn
    if not np.isfinite(sn) or sn <= 0:
        sn = 1e-9
    sigma_n = SMOOTH_SIGMA_S * fs

    sm = np.full_like(a, np.nan)
    for s, e in valid_blocks(a, 3):
        sm[s:e] = gaussian_filter1d(a[s:e], sigma_n, mode="nearest")

    # ---- 1. candidates -----------------------------------------------------------
    cands: List[dict] = []
    for s, e in valid_blocks(sm, 5):
        b = sm[s:e]
        pk, _ = find_peaks(b)
        if pk.size == 0:
            continue
        prom, lbase, rbase = peak_prominences(b, pk)
        widths = peak_widths(b, pk, rel_height=0.5, prominence_data=(prom, lbase, rbase))[0]
        for p, P, w in zip(pk, prom, widths):
            if P >= CANDIDATE_FLOOR_SNR * sn and b[p] > 0:
                cands.append({"index": int(s + p), "block": (s, e), "height": float(b[p]),
                              "prominence": float(P), "fwhm_s": float(w / fs)})
    if not cands:
        return out

    for c in cands:
        c["recovery"] = c["prominence"] / c["height"]
        c["snr"] = c["height"] / sn
        c["independent"] = c["recovery"] >= params.min_rel_recovery
        c["status"], c["reason"], c["cycle_id"] = REJECTED, "", -1

    # ---- 2. robust amplitude reference ---------------------------------------------
    ref_pool = [c["height"] for c in cands if c["independent"] and c["snr"] >= params.min_snr]
    if ref_pool:
        a_ref = float(np.percentile(ref_pool, REF_QUANTILE))
        out["reference_amplitude"] = a_ref
        thr = max(params.min_rel_amplitude * a_ref, params.min_snr * sn)
        out["amplitude_threshold"] = thr
    else:
        a_ref, thr = float("nan"), float("inf")

    # ---- 3. classification ---------------------------------------------------------
    majors: List[dict] = []
    for c in cands:
        if c["snr"] < params.min_snr:
            c["reason"] = "baseline_noise"
        elif not c["independent"]:
            c["reason"] = "_dependent"          # refined below
        elif c["height"] < thr:
            c["reason"] = "insufficient_amplitude"
        else:
            c["status"], c["reason"] = ACCEPTED, ""
            majors.append(c)
    majors.sort(key=lambda c: c["index"])

    # ---- 4. cycles, time-to-peak localisation -----------------------------------------
    by_block: Dict[tuple, List[dict]] = {}
    for c in cands:
        by_block.setdefault(c["block"], []).append(c)
    claimed = set()
    for cid, m in enumerate(majors):
        s, e = m["block"]
        b = sm[s:e]
        i0 = m["index"] - s
        level = (1.0 - params.min_rel_recovery) * m["height"]
        l = i0
        while l > 0 and b[l - 1] >= level:
            l -= 1
        r = i0
        while r < b.size - 1 and b[r + 1] >= level:
            r += 1
        members = [c for c in by_block[m["block"]] if l <= c["index"] - s <= r]
        near = [c for c in members if c["height"] >= TIME_TO_PEAK_FRACTION * m["height"]]
        chosen = min(near, key=lambda c: c["index"]) if near else m
        # transfer decision to the time-to-peak candidate; keep amplitude/height of the chosen one
        if chosen is not m:
            m["status"], m["reason"] = REJECTED, "duplicate_peak_within_cycle"
            chosen["status"], chosen["reason"] = ACCEPTED, ""
        chosen["cycle_id"] = cid
        claimed.add(id(chosen))
        for c in members:
            c["cycle_id"] = cid if c["cycle_id"] < 0 else c["cycle_id"]

    # refine reasons of dependent / non-chosen candidates
    for c in cands:
        if c["status"] == ACCEPTED:
            continue
        if c["reason"] == "_dependent":
            s, e = c["block"]
            i = c["index"]
            lvl = (1.0 - params.min_rel_recovery) * c["height"]
            taller = [o["index"] for o in by_block[c["block"]] if o["height"] > c["height"]]
            taller_left = any(t < i for t in taller)
            taller_right = any(t > i for t in taller)
            # incomplete: the record boundary is reached before the waveform recovers on a side
            # where no taller peak (i.e. no observed saddle) exists
            left_open = (not taller_left) and np.nanmin(sm[s:i + 1]) > lvl
            right_open = (not taller_right) and np.nanmin(sm[i:e]) > lvl
            if left_open or right_open:
                c["reason"] = "incomplete_excursion"
            elif c["height"] >= thr:
                c["reason"] = "duplicate_peak_within_cycle"
            else:
                c["reason"] = "secondary_morphological_feature"
        elif c["reason"] == "":
            c["reason"] = "duplicate_peak_within_cycle"

    # ---- 5. snap accepted markers to the local maximum of A, assemble output ---------
    half = max(1, int(round(SNAP_SIGMA_MULT * sigma_n)))
    rows = []
    for c in sorted(cands, key=lambda c: c["index"]):
        idx = c["index"]
        if c["status"] == ACCEPTED:
            s, e = c["block"]
            lo, hi = max(s, idx - half), min(e, idx + half + 1)
            idx = lo + int(np.argmax(a[lo:hi]))
        rows.append({
            "index": int(idx), "time_rel_s": idx / fs, "amplitude": float(a[idx]),
            "height": c["height"], "prominence": c["prominence"], "recovery": c["recovery"],
            "snr": c["snr"], "fwhm_s": c["fwhm_s"], "cycle_id": int(c["cycle_id"]),
            "status": c["status"], "reason": c["reason"],
        })
    rows.sort(key=lambda r: (r["index"], 0 if r["status"] == ACCEPTED else 1))
    ev_df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    out["events"] = _refine_events_for_impulses(a, fs, sn, ev_df, params, out)
    return out


# Named strictness presets used by the review utility (all share the same internal constants).
# See research/out/param_study.md for how these behave on the development recordings.
STRICTNESS_PRESETS = {
    "permissive": DetectorParams(min_rel_amplitude=0.30, min_rel_recovery=0.30, separation_strictness=0.60, amplitude_strictness=0.60),
    "default": DetectorParams(),
    "strict": DetectorParams(min_rel_amplitude=0.70, min_rel_recovery=0.70, separation_strictness=1.40, amplitude_strictness=1.40),
}

