import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

def match_events_one_to_one(
    predicted_times: np.ndarray,
    annotated_times: np.ndarray,
    tolerance_s: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements deterministic one-to-one matching between predicted events and independently
    annotated contraction times.

    Args:
        predicted_times: Array of predicted event timestamps in seconds.
        annotated_times: Array of ground-truth annotated contraction timestamps in seconds.
        tolerance_s: Maximum allowed distance in seconds for a match.

    Returns:
        Tuple of three arrays:
        - tp_indices: Indices of predicted_times that are true positives.
        - fp_indices: Indices of predicted_times that are false positives.
        - fn_indices: Indices of annotated_times that are false negatives.
    """
    predicted_times = np.asarray(predicted_times)
    annotated_times = np.asarray(annotated_times)

    if len(predicted_times) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.arange(len(annotated_times), dtype=int)

    if len(annotated_times) == 0:
        return np.array([], dtype=int), np.arange(len(predicted_times), dtype=int), np.array([], dtype=int)

    # Matrix of absolute time differences
    diffs = np.abs(predicted_times[:, np.newaxis] - annotated_times[np.newaxis, :])

    # Greedy matching: iteratively find the global minimum difference
    # and mark the corresponding predicted and annotated indices as matched.
    tp_pred = []
    tp_anno = []

    # Track which have been matched
    pred_matched = np.zeros(len(predicted_times), dtype=bool)
    anno_matched = np.zeros(len(annotated_times), dtype=bool)

    # Set diffs above tolerance to infinity
    diffs[diffs > tolerance_s] = np.inf

    while True:
        min_idx = np.unravel_index(np.argmin(diffs), diffs.shape)
        if np.isinf(diffs[min_idx]):
            break

        p_idx, a_idx = min_idx

        tp_pred.append(p_idx)
        tp_anno.append(a_idx)

        pred_matched[p_idx] = True
        anno_matched[a_idx] = True

        # Remove matched events from future consideration
        diffs[p_idx, :] = np.inf
        diffs[:, a_idx] = np.inf

    tp_indices = np.array(tp_pred, dtype=int)
    fp_indices = np.where(~pred_matched)[0]
    fn_indices = np.where(~anno_matched)[0]

    return tp_indices, fp_indices, fn_indices
