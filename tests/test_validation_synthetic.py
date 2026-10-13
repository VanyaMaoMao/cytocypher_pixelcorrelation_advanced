import numpy as np
from pixel_counter.validation.synthetic import generate_synthetic_validation_dataset

def test_synthetic_data_deterministic():
    ds1 = generate_synthetic_validation_dataset(seed=42)
    ds2 = generate_synthetic_validation_dataset(seed=42)

    for key in ds1.keys():
        sig1 = ds1[key]["signal"]
        sig2 = ds2[key]["signal"]

        # Test for equality (ignoring nan equality issues by using array_equal with equal_nan)
        np.testing.assert_array_equal(sig1, sig2)
