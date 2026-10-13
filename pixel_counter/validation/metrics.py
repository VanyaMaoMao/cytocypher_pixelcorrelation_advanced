import numpy as np
from typing import Dict, Any, Tuple

def calculate_validation_metrics(
    tp_count: int,
    fp_count: int,
    fn_count: int,
    predicted_count: int,
    annotated_count: int,
    predicted_rate_hz: float,
    annotated_rate_hz: float,
    duration_s: float
) -> Dict[str, Any]:
    """
    Calculates comprehensive validation metrics given match counts.
    """
    precision = tp_count / predicted_count if predicted_count > 0 else 0.0
    recall = tp_count / annotated_count if annotated_count > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    count_error = predicted_count - annotated_count
    abs_count_error = abs(count_error)

    # Hz/BPM error and bias
    if not np.isnan(predicted_rate_hz) and not np.isnan(annotated_rate_hz):
        hz_error = predicted_rate_hz - annotated_rate_hz
        bpm_error = hz_error * 60.0
    else:
        hz_error = np.nan
        bpm_error = np.nan

    # Check for double/half frequency errors
    is_double_freq = False
    is_half_freq = False
    if annotated_count > 0:
        ratio = predicted_count / annotated_count
        if 1.8 <= ratio <= 2.2:
            is_double_freq = True
        elif 0.4 <= ratio <= 0.6:
            is_half_freq = True

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "count_error": count_error,
        "abs_count_error": abs_count_error,
        "hz_error": hz_error,
        "bpm_error": bpm_error,
        "is_double_freq": is_double_freq,
        "is_half_freq": is_half_freq
    }
