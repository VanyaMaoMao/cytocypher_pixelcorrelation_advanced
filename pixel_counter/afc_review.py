from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from .analysis import recompute_afc_candidates_for_segment
from .config import AFCReviewConfig
from .io_utils import load_afc_review_session_json, save_afc_review_session_json
from .plotting import plot_afc_review_item, save_afc_review_plot
from .results import AFCSegmentReviewDecision, AFCSegmentReviewItem, AFCReviewSession

CURRENT_REVIEW_SCHEMA = "afc_segment_manual_v3"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _item_key(item: AFCSegmentReviewItem) -> int:
    return int(item.segment_index)


def _decision_key(decision: AFCSegmentReviewDecision) -> int:
    return int(decision.segment_index)


def _dedup_sorted(values: Sequence[float], ndigits: int = 6) -> List[float]:
    out: List[float] = []
    seen = set()
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


def _parse_float(text: str, fallback: float) -> float:
    try:
        v = float(str(text).strip())
    except Exception:
        return float(fallback)
    return float(v) if np.isfinite(v) else float(fallback)


def _parse_optional_float(text: str, fallback: float = np.nan) -> float:
    s = str(text).strip()
    if s == "":
        return float(fallback)
    try:
        v = float(s)
    except Exception:
        return float(fallback)
    return float(v) if np.isfinite(v) else float(fallback)


def _nearest_amp(time: np.ndarray, signal: np.ndarray, at_time: float) -> float:
    if time.size == 0 or signal.size == 0 or time.size != signal.size or (not np.isfinite(at_time)):
        return np.nan
    idx = int(np.argmin(np.abs(time - float(at_time))))
    return float(signal[idx]) if 0 <= idx < signal.size else np.nan


def _align_times_and_amps(
    times: Sequence[float],
    amps: Sequence[float],
    time_array: np.ndarray,
    signal_array: np.ndarray,
) -> tuple[list[float], list[float]]:
    aligned_times = list(_dedup_sorted(times))
    amp_map: Dict[float, float] = {}
    for t, a in zip(times, amps):
        try:
            tt = round(float(t), 6)
            aa = float(a)
        except Exception:
            continue
        if np.isfinite(tt) and np.isfinite(aa):
            amp_map[tt] = aa
    aligned_amps: List[float] = []
    for t in aligned_times:
        key = round(float(t), 6)
        if key in amp_map:
            aligned_amps.append(float(amp_map[key]))
        else:
            aligned_amps.append(_nearest_amp(time_array, signal_array, float(t)))
    return aligned_times, aligned_amps


def _pairs_from_times_amps(
    times: Sequence[float],
    amps: Sequence[float],
    time_array: np.ndarray,
    signal_array: np.ndarray,
) -> list[tuple[float, float]]:
    out: Dict[float, tuple[float, float]] = {}
    amps_list = list(amps)
    for i, t in enumerate(times):
        try:
            tf = float(t)
        except Exception:
            continue
        if not np.isfinite(tf):
            continue
        af = np.nan
        if i < len(amps_list):
            try:
                af = float(amps_list[i])
            except Exception:
                af = np.nan
        if not np.isfinite(af):
            af = _nearest_amp(time_array, signal_array, tf)
        key = round(tf, 6)
        out[key] = (float(tf), float(af))
    return sorted(out.values(), key=lambda p: p[0])


def _pairs_to_times_amps(points: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    times = [float(p[0]) for p in points]
    amps = [float(p[1]) for p in points]
    return times, amps


def _upsert_point(points: Sequence[tuple[float, float]], t: float, a: float) -> list[tuple[float, float]]:
    out: Dict[float, tuple[float, float]] = {}
    for pt in points:
        try:
            tt = float(pt[0])
            aa = float(pt[1])
        except Exception:
            continue
        if np.isfinite(tt):
            out[round(tt, 6)] = (tt, aa)
    if np.isfinite(float(t)):
        out[round(float(t), 6)] = (float(t), float(a))
    return sorted(out.values(), key=lambda p: p[0])


def _remove_point(points: Sequence[tuple[float, float]], target_t: float) -> list[tuple[float, float]]:
    keep: list[tuple[float, float]] = []
    target_key = round(float(target_t), 6)
    for pt in points:
        try:
            tt = float(pt[0])
            aa = float(pt[1])
        except Exception:
            continue
        if round(tt, 6) != target_key:
            keep.append((tt, aa))
    return sorted(keep, key=lambda p: p[0])


def _fallback_upper_from_signal(signal: np.ndarray, cfg: AFCReviewConfig) -> float:
    s = np.asarray(signal, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        fb = float(cfg.default_afc_upper_cap_fallback)
        return float(fb if np.isfinite(fb) else 1.0)
    q = float(np.clip(cfg.default_afc_upper_cap_quantile, 0.0, 1.0))
    v = float(np.quantile(s, q))
    if np.isfinite(v):
        return float(v)
    return float(np.max(s)) if s.size else 1.0


def _merge_loaded_over_current(
    current: AFCSegmentReviewItem,
    loaded: AFCSegmentReviewItem,
    *,
    allow_threshold_restore: bool,
) -> AFCSegmentReviewItem:
    # Keep report-derived peaks from current auto results as source of truth.
    merged = replace(
        current,
        helper_candidate_times_s=[float(x) for x in loaded.helper_candidate_times_s],
        helper_candidate_amps=[float(x) for x in loaded.helper_candidate_amps],
        auto_candidate_times_s=[float(x) for x in loaded.auto_candidate_times_s],
        auto_candidate_amps=[float(x) for x in loaded.auto_candidate_amps],
        manual_afc_times_s=[float(x) for x in loaded.manual_afc_times_s],
        manual_afc_amps=[float(x) for x in loaded.manual_afc_amps],
        manual_added_times_s=[float(x) for x in loaded.manual_added_times_s],
        manual_added_amps=[float(x) for x in loaded.manual_added_amps],
        status=str(loaded.status),
    )
    if allow_threshold_restore:
        merged = replace(
            merged,
            afc_lower_left_value=float(loaded.afc_lower_left_value),
            afc_lower_right_value=float(loaded.afc_lower_right_value),
            afc_upper_left_value=float(loaded.afc_upper_left_value),
            afc_upper_right_value=float(loaded.afc_upper_right_value),
            x_start_s=float(loaded.x_start_s),
            x_end_s=float(loaded.x_end_s),
        )
    return merged


def _segment_signal_lookup(segment_results: Sequence[Dict]) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for i, seg in enumerate(segment_results):
        seg_idx = int(seg.get("segment_no", i + 1))
        meta = seg.get("meta", {}) or {}
        out[seg_idx] = {
            "segment_name": str(seg.get("sheet_name", f"Segment {seg_idx}")),
            "time": np.asarray(meta.get("_time_plot", []), dtype=float),
            "signal": np.asarray(meta.get("_sig_plot", []), dtype=float),
        }
    return out


def _apply_decisions_to_items(items: List[AFCSegmentReviewItem], decisions: Sequence[AFCSegmentReviewDecision]) -> None:
    done = {_decision_key(d): str(d.status).lower() for d in decisions}
    for i, item in enumerate(items):
        k = _item_key(item)
        if k in done:
            items[i] = replace(item, status=("skipped" if done[k] == "skipped" else "completed"))
        else:
            items[i] = replace(item, status="pending")


def _next_pending_index(items: Sequence[AFCSegmentReviewItem], start: int = 0) -> int:
    if not items:
        return 0
    for i in range(max(0, int(start)), len(items)):
        if str(items[i].status).lower() == "pending":
            return int(i)
    for i in range(0, max(0, int(start))):
        if str(items[i].status).lower() == "pending":
            return int(i)
    return int(np.clip(start, 0, max(0, len(items) - 1)))


def _normalize_item_to_signal(item: AFCSegmentReviewItem, t: np.ndarray, s: np.ndarray, cfg: AFCReviewConfig) -> AFCSegmentReviewItem:
    if t.size == 0 or s.size == 0 or t.size != s.size:
        return item
    x_start = float(item.x_start_s) if np.isfinite(item.x_start_s) else float(t[0])
    x_end = float(item.x_end_s) if np.isfinite(item.x_end_s) else float(t[-1])
    if x_end < x_start:
        x_start, x_end = x_end, x_start
    x_start = max(float(t[0]), x_start)
    x_end = min(float(t[-1]), x_end)
    if x_end <= x_start:
        x_start = float(t[0])
        x_end = float(t[-1])

    upper_left = float(item.afc_upper_left_value)
    upper_right = float(item.afc_upper_right_value)
    if not np.isfinite(upper_left):
        upper_left = _fallback_upper_from_signal(s, cfg)
    if not np.isfinite(upper_right):
        upper_right = upper_left

    return replace(
        item,
        x_start_s=float(x_start),
        x_end_s=float(x_end),
        afc_upper_left_value=float(upper_left),
        afc_upper_right_value=float(upper_right),
    )


def launch_afc_review_session(
    *,
    input_workbook: str,
    review_items: Sequence[AFCSegmentReviewItem],
    segment_results: Sequence[Dict],
    afc_config: AFCReviewConfig,
    session_json_path: str,
    review_plots_dir: Optional[str] = None,
    interactive: bool = True,
) -> AFCReviewSession:
    session_path = Path(session_json_path)
    signal_lookup = _segment_signal_lookup(segment_results)
    resume_existing_session = bool(getattr(afc_config, "resume_existing_session", False))

    items = [replace(x) for x in review_items]
    decisions: List[AFCSegmentReviewDecision] = []
    created_at = _now_iso()
    if session_path.exists() and resume_existing_session:
        loaded = load_afc_review_session_json(str(session_path))
        created_at = loaded.created_at or created_at
        allow_threshold_restore = str(getattr(loaded, "schema_version", "")).strip() == CURRENT_REVIEW_SCHEMA
        loaded_items = {_item_key(x): x for x in loaded.items}
        merged: List[AFCSegmentReviewItem] = []
        for cur in items:
            k = _item_key(cur)
            if k in loaded_items:
                merged.append(
                    _merge_loaded_over_current(
                        cur,
                        loaded_items[k],
                        allow_threshold_restore=allow_threshold_restore,
                    )
                )
            else:
                merged.append(cur)
        items = merged
        if allow_threshold_restore:
            decisions = list(loaded.decisions)
        else:
            decisions = [
                replace(
                    d,
                    afc_lower_left_value=np.nan,
                    afc_lower_right_value=np.nan,
                    afc_upper_left_value=np.nan,
                    afc_upper_right_value=np.nan,
                    x_start_s=np.nan,
                    x_end_s=np.nan,
                )
                for d in loaded.decisions
            ]

    _apply_decisions_to_items(items, decisions)

    def build_session() -> AFCReviewSession:
        return AFCReviewSession(
            input_workbook=str(input_workbook),
            created_at=str(created_at),
            updated_at=_now_iso(),
            items=items,
            decisions=decisions,
            schema_version=CURRENT_REVIEW_SCHEMA,
        )

    def persist_session(force: bool = False) -> None:
        if force or bool(afc_config.save_review_json) or bool(afc_config.save_partial_progress):
            save_afc_review_session_json(str(session_path), build_session())

    if not interactive or not items:
        persist_session(force=True)
        return build_session()

    decisions_map: Dict[int, AFCSegmentReviewDecision] = {_decision_key(d): d for d in decisions}
    start_idx = int(_next_pending_index(items, start=0)) if resume_existing_session else 0
    state = {
        "idx": int(np.clip(start_idx, 0, max(0, len(items) - 1))),
        "selected_time": None,
        "add_mode": False,
        "helper_points": [],
        "manual_points": [],
        "can_advance": False,
        "review_completed": False,
    }

    fig, ax = plt.subplots(figsize=(15.8, 9.4))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.36)

    # Row 1 labels (manual text labels above empty-label TextBoxes).
    fig.text(0.17, 0.255, "Lower L", ha="center", va="bottom", fontsize=11)
    fig.text(0.38, 0.255, "Lower R", ha="center", va="bottom", fontsize=11)
    fig.text(0.59, 0.255, "Upper L", ha="center", va="bottom", fontsize=11)
    fig.text(0.80, 0.255, "Upper R", ha="center", va="bottom", fontsize=11)

    # Row 1 boxes.
    tb_low_l = TextBox(fig.add_axes([0.12, 0.215, 0.10, 0.040]), "")
    tb_low_r = TextBox(fig.add_axes([0.33, 0.215, 0.10, 0.040]), "")
    tb_up_l = TextBox(fig.add_axes([0.54, 0.215, 0.10, 0.040]), "")
    tb_up_r = TextBox(fig.add_axes([0.75, 0.215, 0.10, 0.040]), "")

    # Row 2 labels.
    fig.text(0.35, 0.165, "x start", ha="center", va="bottom", fontsize=11)
    fig.text(0.61, 0.165, "x end", ha="center", va="bottom", fontsize=11)

    # Row 2 boxes (centered pair).
    tb_xstart = TextBox(fig.add_axes([0.27, 0.125, 0.16, 0.040]), "")
    tb_xend = TextBox(fig.add_axes([0.53, 0.125, 0.16, 0.040]), "")

    # Row 3 buttons.
    b_recompute = Button(fig.add_axes([0.07, 0.060, 0.10, 0.052]), "Recompute")
    b_remove = Button(fig.add_axes([0.19, 0.060, 0.10, 0.052]), "Remove sel")
    b_add = Button(fig.add_axes([0.31, 0.060, 0.10, 0.052]), "Add manual")
    b_save = Button(fig.add_axes([0.43, 0.060, 0.09, 0.052]), "Save")
    b_prev = Button(fig.add_axes([0.55, 0.060, 0.08, 0.052]), "Prev")
    b_next = Button(fig.add_axes([0.65, 0.060, 0.08, 0.052]), "Next")
    b_skip = Button(fig.add_axes([0.75, 0.060, 0.08, 0.052]), "Skip")

    # Row 4 status/help.
    status_text = fig.text(0.055, 0.020, "", fontsize=10, ha="left")

    def current_item() -> AFCSegmentReviewItem:
        return items[int(state["idx"])]

    def current_signal():
        item = current_item()
        seg = signal_lookup.get(int(item.segment_index), {"time": np.array([], dtype=float), "signal": np.array([], dtype=float)})
        return np.asarray(seg["time"], dtype=float), np.asarray(seg["signal"], dtype=float)

    def update_status(msg: str) -> None:
        status_text.set_text(str(msg))
        fig.canvas.draw_idle()

    def set_textboxes_from_item(item: AFCSegmentReviewItem) -> None:
        tb_low_l.set_val("" if not np.isfinite(item.afc_lower_left_value) else f"{float(item.afc_lower_left_value):.6f}")
        tb_low_r.set_val("" if not np.isfinite(item.afc_lower_right_value) else f"{float(item.afc_lower_right_value):.6f}")
        tb_up_l.set_val("" if not np.isfinite(item.afc_upper_left_value) else f"{float(item.afc_upper_left_value):.6f}")
        tb_up_r.set_val("" if not np.isfinite(item.afc_upper_right_value) else f"{float(item.afc_upper_right_value):.6f}")
        tb_xstart.set_val(f"{float(item.x_start_s):.6f}")
        tb_xend.set_val(f"{float(item.x_end_s):.6f}")

    def maybe_save_plot(decision: AFCSegmentReviewDecision) -> None:
        if not bool(afc_config.save_review_png) or not review_plots_dir:
            return
        item = current_item()
        t, s = current_signal()
        if t.size == 0 or s.size == 0:
            return
        out = Path(review_plots_dir) / f"segment_{int(item.segment_index):03d}_review.png"
        item_for_plot = replace(
            item,
            manual_afc_times_s=list(decision.manual_afc_times_s),
            manual_afc_amps=list(decision.manual_afc_amps),
        )
        save_afc_review_plot(
            str(out),
            time_array=t,
            signal_array=s,
            review_item=item_for_plot,
            manual_afc_times_s=decision.manual_afc_times_s,
        )

    def redraw() -> None:
        item = current_item()
        t, s = current_signal()
        helper_times, helper_amps = _pairs_to_times_amps(state["helper_points"])
        manual_times, manual_amps = _pairs_to_times_amps(state["manual_points"])
        item_for_plot = replace(
            item,
            helper_candidate_times_s=helper_times,
            helper_candidate_amps=helper_amps,
            manual_afc_times_s=manual_times,
            manual_afc_amps=manual_amps,
        )
        title = f"{item.segment_name} ({state['idx'] + 1}/{len(items)}) | status={item.status}"
        plot_afc_review_item(
            ax=ax,
            time_array=t,
            signal_array=s,
            review_item=item_for_plot,
            helper_candidate_times_s=helper_times,
            manual_afc_times_s=manual_times,
            selected_time_s=state["selected_time"],
            title=title,
        )
        b_add.label.set_text("Add OFF" if bool(state["add_mode"]) else "Add manual")
        fig.canvas.draw_idle()

    def apply_box_values_to_item() -> AFCSegmentReviewItem:
        item = current_item()
        updated = replace(
            item,
            afc_lower_left_value=_parse_optional_float(tb_low_l.text, np.nan),
            afc_lower_right_value=_parse_optional_float(tb_low_r.text, np.nan),
            afc_upper_left_value=_parse_optional_float(tb_up_l.text, np.nan),
            afc_upper_right_value=_parse_optional_float(tb_up_r.text, np.nan),
            x_start_s=_parse_float(tb_xstart.text, item.x_start_s),
            x_end_s=_parse_float(tb_xend.text, item.x_end_s),
        )
        t, s = current_signal()
        updated = _normalize_item_to_signal(updated, t, s, afc_config)
        items[int(state["idx"])] = updated
        return updated

    def sync_item_state_to_state(item: AFCSegmentReviewItem) -> None:
        helper_times = item.helper_candidate_times_s if item.helper_candidate_times_s else item.auto_candidate_times_s
        helper_amps = item.helper_candidate_amps if item.helper_candidate_amps else item.auto_candidate_amps
        t, s = current_signal()
        state["helper_points"] = _pairs_from_times_amps(helper_times, helper_amps, t, s)

        manual_times = item.manual_afc_times_s if item.manual_afc_times_s else item.manual_added_times_s
        manual_amps = item.manual_afc_amps if item.manual_afc_amps else item.manual_added_amps
        state["manual_points"] = _pairs_from_times_amps(manual_times, manual_amps, t, s)

    def load_item(i: int) -> None:
        state["idx"] = int(np.clip(i, 0, len(items) - 1))
        state["selected_time"] = None
        state["add_mode"] = False
        state["helper_points"] = []
        state["manual_points"] = []
        state["can_advance"] = False
        item = current_item()
        t, s = current_signal()
        item = _normalize_item_to_signal(item, t, s, afc_config)
        items[int(state["idx"])] = item

        dec = decisions_map.get(_item_key(item))
        if dec is not None:
            item = replace(
                item,
                afc_lower_left_value=float(dec.afc_lower_left_value),
                afc_lower_right_value=float(dec.afc_lower_right_value),
                afc_upper_left_value=float(dec.afc_upper_left_value),
                afc_upper_right_value=float(dec.afc_upper_right_value),
                x_start_s=float(dec.x_start_s),
                x_end_s=float(dec.x_end_s),
                manual_afc_times_s=[float(x) for x in dec.manual_afc_times_s],
                manual_afc_amps=[float(x) for x in dec.manual_afc_amps],
                manual_added_times_s=[float(x) for x in dec.manual_added_times_s],
                manual_added_amps=[float(x) for x in dec.manual_added_amps],
            )
            items[int(state["idx"])] = item
            state["can_advance"] = str(dec.status).lower() in {"completed", "skipped"}
        else:
            state["can_advance"] = str(item.status).lower() in {"completed", "skipped"}

        sync_item_state_to_state(item)
        set_textboxes_from_item(item)
        redraw()
        update_status(
            f"Loaded segment {item.segment_index}. Fixed report peaks shown: {len(item.main_peak_times_s)} main, {len(item.rescue_peak_times_s)} rescue."
        )

    def upsert_decision(status: str) -> AFCSegmentReviewDecision:
        item = apply_box_values_to_item()
        t, s = current_signal()
        manual_times_raw, manual_amps_raw = _pairs_to_times_amps(state["manual_points"])
        manual_times, manual_amps = _align_times_and_amps(manual_times_raw, manual_amps_raw, t, s)
        state["manual_points"] = _pairs_from_times_amps(manual_times, manual_amps, t, s)
        manual_times, manual_amps = _pairs_to_times_amps(state["manual_points"])
        decision = AFCSegmentReviewDecision(
            segment_name=str(item.segment_name),
            segment_index=int(item.segment_index),
            afc_lower_left_value=float(item.afc_lower_left_value),
            afc_lower_right_value=float(item.afc_lower_right_value),
            afc_upper_left_value=float(item.afc_upper_left_value),
            afc_upper_right_value=float(item.afc_upper_right_value),
            x_start_s=float(item.x_start_s),
            x_end_s=float(item.x_end_s),
            manual_afc_times_s=list(manual_times),
            manual_afc_amps=[float(x) for x in manual_amps],
            notes="",
            status=str(status),
            accepted_times_s=[],
            accepted_amps=[],
            rejected_times_s=[],
            rejected_amps=[],
            manual_added_times_s=list(manual_times),
            manual_added_amps=[float(x) for x in manual_amps],
        )
        decisions_map[_decision_key(decision)] = decision
        idx_item = int(state["idx"])
        items[idx_item] = replace(
            items[idx_item],
            afc_lower_left_value=float(decision.afc_lower_left_value),
            afc_lower_right_value=float(decision.afc_lower_right_value),
            afc_upper_left_value=float(decision.afc_upper_left_value),
            afc_upper_right_value=float(decision.afc_upper_right_value),
            x_start_s=float(decision.x_start_s),
            x_end_s=float(decision.x_end_s),
            manual_afc_times_s=list(decision.manual_afc_times_s),
            manual_afc_amps=list(decision.manual_afc_amps),
            manual_added_times_s=list(decision.manual_added_times_s),
            manual_added_amps=list(decision.manual_added_amps),
            status="skipped" if status == "skipped" else "completed",
        )
        return decision

    def sync_decisions_and_persist(force: bool = False) -> None:
        decisions.clear()
        decisions.extend(sorted(decisions_map.values(), key=lambda d: int(d.segment_index)))
        _apply_decisions_to_items(items, decisions)
        if bool(afc_config.save_partial_progress) or force:
            persist_session(force=force)

    def on_recompute(_event):
        item = apply_box_values_to_item()
        if not (np.isfinite(item.afc_lower_left_value) and np.isfinite(item.afc_lower_right_value)):
            update_status("Enter AFC lower left and AFC lower right first")
            return
        t, s = current_signal()
        if t.size == 0 or s.size == 0:
            update_status("No signal available for this segment.")
            return
        current_manual_times, current_manual_amps = _pairs_to_times_amps(state["manual_points"])
        refreshed = recompute_afc_candidates_for_segment(item, t, s, afc_config)
        refreshed = replace(
            refreshed,
            manual_afc_times_s=current_manual_times,
            manual_afc_amps=current_manual_amps,
            manual_added_times_s=current_manual_times,
            manual_added_amps=current_manual_amps,
        )
        items[int(state["idx"])] = refreshed
        helper_times, helper_amps = _align_times_and_amps(
            refreshed.helper_candidate_times_s,
            refreshed.helper_candidate_amps,
            t,
            s,
        )
        state["helper_points"] = _pairs_from_times_amps(helper_times, helper_amps, t, s)
        state["can_advance"] = True
        redraw()
        if bool(afc_config.save_partial_progress):
            persist_session(force=False)
        update_status(f"Helper candidates refreshed: {len(state['helper_points'])}. Manual AFC unchanged.")

    def _nearest_from_pool(pool: Sequence[float], target: Optional[float]) -> Optional[float]:
        vals = _dedup_sorted(pool)
        if not vals:
            return None
        if target is None or (not np.isfinite(float(target))):
            return vals[0]
        return min(vals, key=lambda x: abs(float(x) - float(target)))

    def on_remove_selected(_event):
        manual_times, _ = _pairs_to_times_amps(state["manual_points"])
        nearest = _nearest_from_pool(list(manual_times), state["selected_time"])
        if nearest is None:
            update_status("No AFC point available to remove.")
            return
        state["manual_points"] = _remove_point(state["manual_points"], nearest)
        redraw()
        update_status(f"Removed AFC point near {nearest:.4f}s")

    def on_toggle_add(_event):
        state["add_mode"] = not bool(state["add_mode"])
        redraw()
        if state["add_mode"]:
            update_status("Add mode ON: click helper crosses to select final AFC points.")
        else:
            update_status("Add mode OFF.")

    def on_save(_event):
        decision = upsert_decision(status="completed")
        maybe_save_plot(decision)
        sync_decisions_and_persist(force=True)
        state["can_advance"] = True
        redraw()
        update_status("Segment AFC review saved.")

    def on_prev(_event):
        load_item(max(0, int(state["idx"]) - 1))

    def on_next(_event):
        if not bool(state["can_advance"]):
            update_status("Click Recompute first or Skip")
            return
        upsert_decision(status="completed")
        sync_decisions_and_persist(force=True)
        if int(state["idx"]) >= len(items) - 1:
            state["review_completed"] = True
            update_status("Review completed. Building final outputs...")
            fig.canvas.draw_idle()
            plt.pause(0.05)
            plt.close(fig)
            return
        next_idx = min(len(items) - 1, int(state["idx"]) + 1)
        load_item(next_idx)

    def on_skip(_event):
        state["manual_points"] = []
        state["helper_points"] = []
        upsert_decision(status="skipped")
        sync_decisions_and_persist(force=True)
        if int(state["idx"]) >= len(items) - 1:
            state["review_completed"] = True
            update_status("Review completed. Building final outputs...")
            fig.canvas.draw_idle()
            plt.pause(0.05)
            plt.close(fig)
            return
        next_idx = min(len(items) - 1, int(state["idx"]) + 1)
        load_item(next_idx)
        update_status("Segment skipped and moved to next segment.")

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        t, s = current_signal()
        if t.size == 0 or s.size == 0:
            return
        x = float(event.xdata)
        if bool(state["add_mode"]):
            item = current_item()
            helper_times, helper_amps = _pairs_to_times_amps(state["helper_points"])
            if not helper_times:
                update_status("No helper peaks available. Click Recompute first.")
                return
            x_span = max(float(item.x_end_s) - float(item.x_start_s), 0.0)
            snap_tol = max(0.01, min(0.08, 0.02 * x_span))
            helper_arr = np.asarray(helper_times, dtype=float)
            i_near = int(np.argmin(np.abs(helper_arr - x)))
            chosen_t = float(helper_arr[i_near])
            if abs(chosen_t - x) > snap_tol:
                update_status("Click closer to a helper peak to add it.")
                return
            if not (float(item.x_start_s) <= chosen_t <= float(item.x_end_s)):
                update_status("Selected helper is outside current x-range.")
                return
            if 0 <= i_near < len(helper_amps) and np.isfinite(float(helper_amps[i_near])):
                chosen_a = float(helper_amps[i_near])
            else:
                chosen_a = _nearest_amp(t, s, chosen_t)
            state["selected_time"] = chosen_t
            state["manual_points"] = _upsert_point(state["manual_points"], chosen_t, chosen_a)
            redraw()
            update_status(f"Added final AFC at {chosen_t:.4f}s (Add mode still ON)")
            return
        idx_near = int(np.argmin(np.abs(t - x)))
        chosen_t = float(t[idx_near])
        state["selected_time"] = chosen_t
        redraw()
        update_status(f"Selected {chosen_t:.4f}s")

    b_recompute.on_clicked(on_recompute)
    b_remove.on_clicked(on_remove_selected)
    b_add.on_clicked(on_toggle_add)
    b_save.on_clicked(on_save)
    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_skip.on_clicked(on_skip)
    fig.canvas.mpl_connect("button_press_event", on_click)

    load_item(int(np.clip(start_idx, 0, len(items) - 1)))
    plt.show(block=True)

    sync_decisions_and_persist(force=True)
    return build_session()
