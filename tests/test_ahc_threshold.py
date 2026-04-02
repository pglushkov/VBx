import numpy as np

from VBx.diarization_lib import twoGMMcalib_lin
import vbx_native


def test_ahc_threshold_matches_python():
    """C++ ahc_threshold must match Python twoGMMcalib_lin threshold."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(200)

    py_threshold, _ = twoGMMcalib_lin(scores)
    cpp_threshold = vbx_native.ahc_threshold(scores)

    np.testing.assert_allclose(cpp_threshold, py_threshold, atol=1e-12)


def test_ahc_threshold_bimodal():
    """Two well-separated clusters: threshold should land between modes."""
    rng = np.random.default_rng(7)
    low = rng.normal(-3.0, 0.5, 500)
    high = rng.normal(3.0, 0.5, 500)
    scores = np.concatenate([low, high])

    threshold = vbx_native.ahc_threshold(scores)

    assert -1.0 < threshold < 1.0, f"threshold {threshold} not between modes"


def test_ahc_threshold_float32():
    """Float32 path should match Python within fp32 tolerance."""
    rng = np.random.default_rng(99)
    scores = rng.standard_normal(150).astype(np.float32)

    py_threshold, _ = twoGMMcalib_lin(scores.astype(np.float64))
    cpp_threshold = vbx_native.ahc_threshold(scores)

    np.testing.assert_allclose(cpp_threshold, py_threshold, atol=1e-4)


def test_ahc_threshold_custom_niters():
    """Verify niters parameter affects the result."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(200)

    t1 = vbx_native.ahc_threshold(scores, niters=1)
    t20 = vbx_native.ahc_threshold(scores, niters=20)

    assert t1 != t20
