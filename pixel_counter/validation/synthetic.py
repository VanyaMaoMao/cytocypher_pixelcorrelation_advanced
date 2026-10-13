import numpy as np
from typing import Dict, Any

def generate_synthetic_validation_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    Generates a set of deterministic synthetic recordings with known ground-truth annotations.
    """
    dataset = {}
    rng = np.random.default_rng(seed)

    # 1. Standard 1Hz clean signal
    fs = 250.0
    t1 = np.arange(2500) / fs # 10 seconds
    sig1 = np.zeros_like(t1)
    true_times_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for c in true_times_1:
        sig1 += 1.0 * np.exp(-0.5 * ((t1 - c) / 0.02) ** 2)
    sig1 += rng.normal(0, 0.01, sig1.shape)

    dataset["synthetic_1hz_clean"] = {
        "signal": sig1,
        "fs": fs,
        "true_times": np.array(true_times_1),
        "true_polarity": 1,
        "description": "Clean 1Hz positive peaks"
    }

    # 2. Inverted 2Hz signal
    t2 = np.arange(2500) / fs # 10 seconds
    sig2 = np.zeros_like(t2)
    true_times_2 = np.arange(0.5, 9.6, 0.5)
    for c in true_times_2:
        sig2 -= 1.0 * np.exp(-0.5 * ((t2 - c) / 0.02) ** 2)
    sig2 += rng.normal(0, 0.01, sig2.shape)

    dataset["synthetic_2hz_inverted"] = {
        "signal": sig2,
        "fs": fs,
        "true_times": true_times_2,
        "true_polarity": -1,
        "description": "Clean 2Hz negative peaks"
    }

    # 3. Missing data / gap signal
    t3 = np.arange(2500) / fs
    sig3 = np.zeros_like(t3)
    true_times_3 = [1.0, 2.0, 8.0, 9.0] # Gap between 3 and 7
    for c in true_times_3:
        sig3 += 1.0 * np.exp(-0.5 * ((t3 - c) / 0.02) ** 2)

    # Introduce NaN gap
    gap_start = int(3.0 * fs)
    gap_end = int(7.0 * fs)
    sig3[gap_start:gap_end] = np.nan

    dataset["synthetic_gap_1hz"] = {
        "signal": sig3,
        "fs": fs,
        "true_times": np.array(true_times_3),
        "true_polarity": 1,
        "description": "1Hz peaks with large NaN gap in middle"
    }

    # 4. Irregular rhythm
    t4 = np.arange(5000) / fs # 20 seconds
    sig4 = np.zeros_like(t4)
    true_times_4 = [1.0, 1.2, 1.8, 3.5, 4.0, 6.0, 6.1, 8.0, 15.0, 18.0]
    for c in true_times_4:
        sig4 += 1.0 * np.exp(-0.5 * ((t4 - c) / 0.02) ** 2)
    sig4 += rng.normal(0, 0.02, sig4.shape)

    dataset["synthetic_irregular"] = {
        "signal": sig4,
        "fs": fs,
        "true_times": np.array(true_times_4),
        "true_polarity": 1,
        "description": "Irregular high-quality peaks"
    }

    return dataset
