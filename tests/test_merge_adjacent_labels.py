"""Unit tests for vbx_native.merge_adjacent_labels.

The C++ port in vbx_lib/src/vbx.cpp is compared against the Python
reference in VBx.diarization_lib.merge_adjacent_labels across a handful
of hand-crafted scenarios that cover the branches of the algorithm:
merging same-label runs, detecting real time gaps, and midpoint-splitting
overlaps between different-label segments.
"""

import numpy as np

import vbx_native
from VBx.diarization_lib import merge_adjacent_labels as merge_py


def _call_cpp(starts, ends, labels):
    return vbx_native.merge_adjacent_labels(
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends,   dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
    )


def _call_py(starts, ends, labels):
    # merge_py mutates inputs — copy defensively.
    return merge_py(
        np.asarray(starts, dtype=np.float64).copy(),
        np.asarray(ends,   dtype=np.float64).copy(),
        np.asarray(labels, dtype=np.int32).copy(),
    )


def _assert_matches_python(starts, ends, labels):
    py_s, py_e, py_l = _call_py(starts, ends, labels)
    cpp_s, cpp_e, cpp_l = _call_cpp(starts, ends, labels)
    np.testing.assert_allclose(cpp_s, py_s, atol=1e-12)
    np.testing.assert_allclose(cpp_e, py_e, atol=1e-12)
    np.testing.assert_array_equal(cpp_l, py_l)


# ---------------------------------------------------------------------------
# Core merging behavior
# ---------------------------------------------------------------------------

def test_single_segment():
    """One segment in, one segment out — trivial trailing-run case."""
    _assert_matches_python([0.0], [1.0], [3])


def test_all_same_label_contiguous():
    """Contiguous runs of the same label collapse to a single segment."""
    _assert_matches_python(
        starts=[0.0, 1.0, 2.0, 3.0],
        ends  =[1.0, 2.0, 3.0, 4.0],
        labels=[0, 0, 0, 0],
    )


def test_alternating_labels_no_merge():
    """Contiguous but differently-labeled segments stay separate."""
    _assert_matches_python(
        starts=[0.0, 1.0, 2.0, 3.0],
        ends  =[1.0, 2.0, 3.0, 4.0],
        labels=[0, 1, 0, 1],
    )


def test_time_gap_splits_same_label():
    """A real time gap breaks a same-label run into two segments."""
    _assert_matches_python(
        starts=[0.0, 1.0, 5.0, 6.0],
        ends  =[1.0, 2.0, 6.0, 7.0],
        labels=[0, 0, 0, 0],
    )


def test_isclose_gap_is_merged():
    """Float-level noise well within numpy.isclose tolerance still merges."""
    _assert_matches_python(
        starts=[0.0, 1.0 + 1e-12, 2.0 - 1e-12],
        ends  =[1.0, 2.0,         3.0],
        labels=[7, 7, 7],
    )


# ---------------------------------------------------------------------------
# Overlap handling
# ---------------------------------------------------------------------------

def test_overlap_same_label_merged():
    """Overlapping same-label segments merge; no midpoint split needed."""
    _assert_matches_python(
        starts=[0.0, 0.5, 1.8],
        ends  =[1.0, 2.0, 3.0],
        labels=[2, 2, 2],
    )


def test_overlap_different_labels_midpoint_split():
    """Overlap between two different-label segments is resolved at midpoint."""
    _assert_matches_python(
        starts=[0.0, 0.8],
        ends  =[1.0, 2.0],
        labels=[0, 1],
    )


def test_three_way_mix():
    """Mixed run that exercises merge, gap split, and overlap split together."""
    _assert_matches_python(
        starts=[0.0, 1.0, 2.0, 5.0, 5.9, 7.1, 7.5],
        ends  =[1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0],
        labels=[0,   0,   0,   1,   1,   2,   2],
    )


# ---------------------------------------------------------------------------
# Randomized cross-check
# ---------------------------------------------------------------------------

def test_random_contiguous_runs():
    """Randomly generated contiguous runs across a few seeds."""
    for seed in (1, 7, 42, 2026):
        rng = np.random.default_rng(seed)
        N = int(rng.integers(5, 40))
        starts = np.cumsum(rng.uniform(0.1, 0.5, size=N))
        ends   = starts + rng.uniform(0.1, 0.3, size=N)
        labels = rng.integers(0, 4, size=N).astype(np.int32)
        _assert_matches_python(starts, ends, labels)
