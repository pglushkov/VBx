import fastcluster
import numpy as np
import vbx_native
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from VBx.diarization_lib import cos_similarity


def test_average_linkage_heights():
    """Heights from C++ and Python fastcluster should match."""
    rng = np.random.default_rng(42)
    N, D = 20, 64
    xvecs = rng.standard_normal((N, D))

    # Python reference
    sim = cos_similarity(xvecs)
    dist_condensed = squareform(-sim, checks=False)
    Z_py = fastcluster.linkage(dist_condensed, method="average")

    # C++ path
    sim_cpp = vbx_native.cosine_similarity(xvecs)
    dist_cpp = -sim_cpp
    Z_cpp = vbx_native.average_linkage(np.ascontiguousarray(dist_cpp), N)

    # Heights must match (sorted, since merge order may differ at ties)
    np.testing.assert_allclose(np.sort(Z_cpp[:, 2]), np.sort(Z_py[:, 2]), atol=1e-12)


def test_average_linkage_end_to_end_labels():
    """Full C++ path vs full Python path should produce same cluster partitions."""
    rng = np.random.default_rng(42)
    N, D = 20, 64
    xvecs = rng.standard_normal((N, D))

    # Python path
    sim_py = cos_similarity(xvecs)
    dist_py = squareform(-sim_py, checks=False)
    Z_py = fastcluster.linkage(dist_py, method="average")

    # C++ path
    sim_cpp = vbx_native.cosine_similarity(xvecs)
    Z_cpp = vbx_native.average_linkage(-sim_cpp, N)

    # Shift heights to non-negative for fcluster
    Z_py[:, 2] += abs(Z_py[:, 2].min())
    Z_cpp[:, 2] += abs(Z_cpp[:, 2].min())

    assert np.allclose(Z_py, Z_cpp)


def test_average_linkage_small():
    """Sanity check with a trivial 3-point case."""
    # 3 points: d(0,1)=1, d(0,2)=5, d(1,2)=3 — no ties
    dist = np.array([1.0, 5.0, 3.0])
    Z_py = fastcluster.linkage(dist.copy(), method="average")
    Z_cpp = vbx_native.average_linkage(dist, 3)

    # Heights and sizes should match exactly
    np.testing.assert_allclose(Z_cpp[:, 2], Z_py[:, 2], atol=1e-15)
    np.testing.assert_allclose(Z_cpp[:, 3], Z_py[:, 3], atol=1e-15)


# def test_clusterization():
#     # TODO: work on this test once final clusterization is done
#     def _same_partition(a, b):
#         """Check if two label arrays define the same partition (ignoring label values)."""
#         mapping = {}
#         for la, lb in zip(a, b):
#             if la in mapping:
#                 if mapping[la] != lb:
#                     return False
#             else:
#                 mapping[la] = lb
#         # Also check reverse mapping is consistent
#         rev = {}
#         for la, lb in zip(a, b):
#             if lb in rev:
#                 if rev[lb] != la:
#                     return False
#             else:
#                 rev[lb] = la
#         return True

#     for t in [0.5, 1.0, 1.5, 2.0]:
#         labels_py = fcluster(Z_py, t, criterion="distance")
#         labels_cpp = fcluster(Z_cpp, t, criterion="distance")
#         assert _same_partition(labels_py, labels_cpp), (
#             f"Partitions differ at threshold {t}"
#         )
