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
from scipy.ndimage import gaussian_filter1d
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


@dataclass(frozen=True)
class DetectorParams:
    """The three user-adjustable strictness controls."""
    min_rel_amplitude: float = 0.50   # alpha: a major contraction is >= alpha * typical major amplitude
    min_rel_recovery: float = 0.50    # rho:   separate cycles must recover >= rho of their height
    min_snr: float = 8.0              # absolute floor: height >= min_snr * noise sigma

    def validate(self) -> "DetectorParams":
        if not (0.0 < self.min_rel_amplitude < 1.0):
            raise ValueError("min_rel_amplitude must be in (0, 1)")
        if not (0.0 < self.min_rel_recovery < 1.0):
            raise ValueError("min_rel_recovery must be in (0, 1)")
        if not (self.min_snr > 0.0):
            raise ValueError("min_snr must be > 0")
        return self


def params_from_config(config) -> DetectorParams:
    return DetectorParams(
        min_rel_amplitude=float(getattr(config, "v2_min_rel_amplitude", 0.50)),
        min_rel_recovery=float(getattr(config, "v2_min_rel_recovery", 0.50)),
        min_snr=float(getattr(config, "v2_min_snr", 8.0)),
    ).validate()


def _empty_events() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in EVENT_COLUMNS})
    df["index"] = df["index"].astype(int)
    return df


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
    out["events"] = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return out


# Named strictness presets used by the review utility (all share the same internal constants).
# See research/out/param_study.md for how these behave on the development recordings.
STRICTNESS_PRESETS = {
    "permissive": DetectorParams(min_rel_amplitude=0.30, min_rel_recovery=0.30),
    "default": DetectorParams(),
    "strict": DetectorParams(min_rel_amplitude=0.70, min_rel_recovery=0.70),
}

