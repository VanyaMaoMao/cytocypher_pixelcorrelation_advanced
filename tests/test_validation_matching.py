import numpy as np
from pixel_counter.validation.matching import match_events_one_to_one

def test_perfect_matching():
    pred = np.array([1.0, 2.0, 3.0])
    anno = np.array([1.0, 2.0, 3.0])

    tp_idx, fp_idx, fn_idx = match_events_one_to_one(pred, anno, tolerance_s=0.05)

    assert len(tp_idx) == 3
    assert len(fp_idx) == 0
    assert len(fn_idx) == 0

def test_missing_and_extra_matching():
    # pred has an extra 1.5, misses 3.0
    pred = np.array([1.0, 1.5, 2.0])
    anno = np.array([1.0, 2.0, 3.0])

    tp_idx, fp_idx, fn_idx = match_events_one_to_one(pred, anno, tolerance_s=0.05)

    assert len(tp_idx) == 2
    assert len(fp_idx) == 1
    assert len(fn_idx) == 1

    assert pred[fp_idx[0]] == 1.5
    assert anno[fn_idx[0]] == 3.0

def test_within_tolerance():
    pred = np.array([1.02, 1.98])
    anno = np.array([1.0, 2.0])

    tp_idx, fp_idx, fn_idx = match_events_one_to_one(pred, anno, tolerance_s=0.05)

    assert len(tp_idx) == 2
    assert len(fp_idx) == 0
    assert len(fn_idx) == 0

def test_outside_tolerance():
    pred = np.array([1.06, 1.94])
    anno = np.array([1.0, 2.0])

    tp_idx, fp_idx, fn_idx = match_events_one_to_one(pred, anno, tolerance_s=0.05)

    assert len(tp_idx) == 0
    assert len(fp_idx) == 2
    assert len(fn_idx) == 2

def test_maximum_matching_greedy_failure():
    # A greedy approach might match pred[0] (0.04) with anno[1] (0.05) distance=0.01
    # which leaves pred[1] (0.07) and anno[0] (0.00) unmatched because distance > 0.05.
    # Total matches = 1.
    # Maximum cardinality matching should pair (0.00, 0.04) distance=0.04 and (0.05, 0.07) distance=0.02.
    # Total matches = 2.

    pred = np.array([0.04, 0.07])
    anno = np.array([0.00, 0.05])

    tp_idx, fp_idx, fn_idx = match_events_one_to_one(pred, anno, tolerance_s=0.05)

    assert len(tp_idx) == 2
    assert len(fp_idx) == 0
    assert len(fn_idx) == 0
