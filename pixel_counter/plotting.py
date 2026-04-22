from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BeatCounterConfig
from .results import AFCSegmentReviewItem

def plot_events(*, time: np.ndarray, sig: np.ndarray, events: pd.DataFrame, file_name: str, bpm: float, n_main: int, n_rescue: int, rescue_peaks: Sequence[int], snr: float, strong_thr: float, weak_thr: float, config: BeatCounterConfig, show: bool = True, return_fig: bool = False, meta: Optional[Dict] = None):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(time, np.asarray(sig, dtype=float), color="black", alpha=0.65, lw=1.1, label="Oriented PixelCorrelation")
    if not events.empty:
        m = events[events.Type == "Main Beat"]
        if not m.empty:
            ax.scatter(m.Time_s, m.Amp, s=60, c="#1f77b4", label="Main", zorder=3)
    rescue_idx = np.asarray(sorted(set(int(x) for x in rescue_peaks)), dtype=int)
    rescue_idx = rescue_idx[(rescue_idx >= 0) & (rescue_idx < len(sig))]
    if rescue_idx.size > 0:
        ax.scatter(np.asarray(time, dtype=float)[rescue_idx], np.asarray(sig, dtype=float)[rescue_idx], s=52, c="#2ca02c", label="Rescue", zorder=3)

    if config.show_threshold_guides and np.isfinite(strong_thr) and np.isfinite(weak_thr):
        ax.axhline(strong_thr, ls="--", alpha=0.25, color="#1f77b4")
        ax.axhline(weak_thr, ls=":", alpha=0.25, color="#2ca02c")

    qc_flag, qc_reason, invert, conf = "PASS", "pass", None, np.nan
    if meta:
        qc_flag = "PASS" if bool(meta.get("qc_pass", True)) else "REJECT"
        qc_reason = str(meta.get("qc_reason", "pass"))
        orient = meta.get("orientation", {})
        invert = orient.get("invert", None)
        conf = float(orient.get("confidence", np.nan))

    ax.set_title(f"{file_name}\nQC={qc_flag} ({qc_reason}) | main={n_main} rescue={n_rescue} bpm={bpm:.2f} | invert={invert} conf={conf:.2f}")
    ax.set_xlabel("Concatenated Time (s)")
    ax.set_ylabel("Amplitude (oriented, baseline-shifted)")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right")

    if show:
        plt.show()
    if return_fig:
        return fig
    if not show:
        plt.close(fig)


def _nearest_amp_for_times(time_array: np.ndarray, signal_array: np.ndarray, times_s: Sequence[float]) -> np.ndarray:
    time = np.asarray(time_array, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    if time.size == 0 or sig.size == 0 or time.size != sig.size:
        return np.array([], dtype=float)
    vals = []
    for t in times_s:
        try:
            tt = float(t)
        except Exception:
            continue
        if not np.isfinite(tt):
            continue
        idx = int(np.argmin(np.abs(time - tt)))
        vals.append(float(sig[idx]))
    return np.asarray(vals, dtype=float)


def _build_sloped_line(
    x_values: np.ndarray,
    x_start_s: float,
    x_end_s: float,
    left_value: float,
    right_value: float,
) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    xs = float(x_start_s)
    xe = float(x_end_s)
    if xe == xs:
        return np.full(x.shape, float(left_value), dtype=float)
    if xe < xs:
        xs, xe = xe, xs
        left_value, right_value = float(right_value), float(left_value)
    frac = np.clip((x - xs) / max(xe - xs, 1e-12), 0.0, 1.0)
    return float(left_value) + frac * (float(right_value) - float(left_value))


def plot_afc_review_item(
    *,
    ax,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    review_item: AFCSegmentReviewItem,
    helper_candidate_times_s: Sequence[float] = (),
    manual_afc_times_s: Sequence[float] = (),
    selected_time_s: Optional[float] = None,
    title: Optional[str] = None,
    show_helper_candidates: bool = True,
):
    time = np.asarray(time_array, dtype=float)
    sig = np.asarray(signal_array, dtype=float)
    ax.clear()
    if time.size == 0 or sig.size == 0 or time.size != sig.size:
        ax.set_title("AFC review: empty signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return ax

    ax.plot(time, sig, color="black", lw=1.1, alpha=0.8, label="Signal (full segment)")

    ax.axvline(float(review_item.x_start_s), color="#9467bd", lw=1.0, ls="--", alpha=0.85, label="x start")
    ax.axvline(float(review_item.x_end_s), color="#9467bd", lw=1.0, ls="--", alpha=0.85, label="x end")
    x_line = np.linspace(float(review_item.x_start_s), float(review_item.x_end_s), 120)
    if np.isfinite(float(review_item.afc_lower_left_value)) and np.isfinite(float(review_item.afc_lower_right_value)):
        y_line_low = _build_sloped_line(
            x_line,
            float(review_item.x_start_s),
            float(review_item.x_end_s),
            float(review_item.afc_lower_left_value),
            float(review_item.afc_lower_right_value),
        )
        ax.plot(
            x_line,
            y_line_low,
            color="#17becf",
            lw=1.3,
            ls="--",
            alpha=0.95,
            label="AFC lower guide",
        )
    if np.isfinite(float(review_item.afc_upper_left_value)) and np.isfinite(float(review_item.afc_upper_right_value)):
        y_line_high = _build_sloped_line(
            x_line,
            float(review_item.x_start_s),
            float(review_item.x_end_s),
            float(review_item.afc_upper_left_value),
            float(review_item.afc_upper_right_value),
        )
        ax.plot(
            x_line,
            y_line_high,
            color="#9467bd",
            lw=1.2,
            ls="-.",
            alpha=0.95,
            label="AFC upper cap",
        )

    main_times = [float(x) for x in review_item.main_peak_times_s]
    main_amps = np.asarray(review_item.main_peak_amps, dtype=float)
    if main_times:
        if main_amps.size != len(main_times):
            main_amps = _nearest_amp_for_times(time, sig, main_times)
        ax.scatter(main_times, main_amps, c="#1f77b4", s=46, marker="o", zorder=4, label="Main peaks")

    rescue_times = [float(x) for x in review_item.rescue_peak_times_s]
    rescue_amps = np.asarray(review_item.rescue_peak_amps, dtype=float)
    if rescue_times:
        if rescue_amps.size != len(rescue_times):
            rescue_amps = _nearest_amp_for_times(time, sig, rescue_times)
        ax.scatter(rescue_times, rescue_amps, c="#2ca02c", s=44, marker="^", zorder=4, label="Rescue peaks")

    if bool(show_helper_candidates):
        helper_times = [
            float(x)
            for x in (
                helper_candidate_times_s
                if helper_candidate_times_s
                else (review_item.helper_candidate_times_s if review_item.helper_candidate_times_s else review_item.auto_candidate_times_s)
            )
        ]
        if helper_times:
            helper_amps = np.asarray(review_item.helper_candidate_amps, dtype=float)
            if helper_amps.size != len(helper_times):
                helper_amps = np.asarray(review_item.auto_candidate_amps, dtype=float)
            if helper_amps.size != len(helper_times):
                helper_amps = _nearest_amp_for_times(time, sig, helper_times)
            ax.scatter(
                helper_times,
                helper_amps,
                c="#ff7f0e",
                s=52,
                marker="x",
                linewidths=1.1,
                alpha=0.95,
                zorder=4,
                label="AFC helpers (Recompute)",
            )

    man_times = [float(x) for x in (manual_afc_times_s if manual_afc_times_s else review_item.manual_afc_times_s)]
    if man_times:
        man_amps = np.asarray(review_item.manual_afc_amps, dtype=float)
        if man_amps.size != len(man_times):
            man_amps = _nearest_amp_for_times(time, sig, man_times)
        ax.scatter(
            man_times,
            man_amps,
            c="#cc5c00",
            s=84,
            marker="D",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label="Manual AFC (final)",
        )

    if selected_time_s is not None and np.isfinite(float(selected_time_s)):
        ax.axvline(float(selected_time_s), color="#ff1493", lw=1.2, alpha=0.85, label="Selected")

    ttl = title or f"{review_item.segment_name} | Segment-level AFC review"
    ax.set_title(ttl)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (oriented, baseline-shifted)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


def save_afc_review_plot(
    path: str,
    *,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    review_item: AFCSegmentReviewItem,
    manual_afc_times_s: Sequence[float] = (),
    show_helper_candidates: bool = True,
    dpi: int = 180,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    plot_afc_review_item(
        ax=ax,
        time_array=time_array,
        signal_array=signal_array,
        review_item=review_item,
        manual_afc_times_s=manual_afc_times_s,
        show_helper_candidates=show_helper_candidates,
    )
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def save_afc_report_plot(
    path: str,
    *,
    time_array: np.ndarray,
    signal_array: np.ndarray,
    review_item: AFCSegmentReviewItem,
    manual_afc_times_s: Sequence[float] = (),
    dpi: int = 180,
) -> None:
    save_afc_review_plot(
        path,
        time_array=time_array,
        signal_array=signal_array,
        review_item=review_item,
        manual_afc_times_s=manual_afc_times_s,
        show_helper_candidates=False,
        dpi=dpi,
    )
