import numpy as np
from scipy.spatial.distance import squareform

from VBx.diarization_lib import cos_similarity
import vbx_native


def test_cosine_similarity_vs_python():
    """Compare C++ condensed cosine similarity against Python reference."""
    rng = np.random.default_rng(42)
    # N x-vectors of dimension D
    N, D = 20, 64
    xvecs = rng.standard_normal((N, D))

    # Python reference: full N x N matrix -> condensed upper triangle
    full_matrix = cos_similarity(xvecs)
    expected = squareform(full_matrix, checks=False)

    # C++ implementation: returns condensed directly
    result = vbx_native.cosine_similarity(xvecs)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_cosine_similarity_float32():
    """Verify float32 path works and matches within fp32 tolerance."""
    rng = np.random.default_rng(123)
    N, D = 15, 32
    xvecs = rng.standard_normal((N, D)).astype(np.float32)

    full_matrix = cos_similarity(xvecs.astype(np.float64))
    expected = squareform(full_matrix, checks=False).astype(np.float32)

    result = vbx_native.cosine_similarity(xvecs)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have zero similarity."""
    xvecs = np.eye(5)  # 5 orthogonal unit vectors
    result = vbx_native.cosine_similarity(xvecs)
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_cosine_similarity_identical():
    """Identical vectors should have similarity 1.0."""
    xvecs = np.array([[1.0, 2.0, 3.0]] * 4)
    result = vbx_native.cosine_similarity(xvecs)
    np.testing.assert_allclose(result, 1.0, atol=1e-15)
