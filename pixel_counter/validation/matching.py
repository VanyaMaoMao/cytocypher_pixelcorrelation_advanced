import numpy as np
from typing import Tuple
from scipy.optimize import linear_sum_assignment

def match_events_one_to_one(
    predicted_times: np.ndarray,
    annotated_times: np.ndarray,
    tolerance_s: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements deterministic one-to-one matching between predicted events and independently
    annotated contraction times, maximizing the total number of matches within tolerance.

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

    # Cost matrix: absolute time differences
    cost_matrix = np.abs(predicted_times[:, np.newaxis] - annotated_times[np.newaxis, :])

    # We want to forbid matches where cost > tolerance.
    # linear_sum_assignment minimizes the total cost.
    # If we add a large penalty to invalid matches, it will avoid them unless there's no valid alternative.
    # To truly enforce maximum cardinality of VALID matches, we should use a modified cost:
    # Any valid match has cost < tolerance_s. We subtract a large constant so it prefers making *more* matches.

    # Create an augmented cost matrix where valid matches have negative cost (to maximize match count)
    # and invalid matches have high positive cost.
    large_penalty = 1000.0
    augmented_cost = np.where(cost_matrix <= tolerance_s, cost_matrix - large_penalty, cost_matrix + large_penalty)

    row_ind, col_ind = linear_sum_assignment(augmented_cost)

    tp_pred = []
    tp_anno = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= tolerance_s:
            tp_pred.append(r)
            tp_anno.append(c)

    tp_indices = np.array(tp_pred, dtype=int)

    fp_indices = np.setdiff1d(np.arange(len(predicted_times)), tp_indices)
    fn_indices = np.setdiff1d(np.arange(len(annotated_times)), tp_anno)

    return tp_indices, fp_indices, fn_indices
