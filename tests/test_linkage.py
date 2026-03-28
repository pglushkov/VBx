import fastcluster
import numpy as np
import vbx_native
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from VBx.diarization_lib import cos_similarity


def _normalize_labels(labels):
    """Remap labels so first occurrence of each label gets 1, next new label gets 2, etc."""
    mapping = {}
    out = []
    for l in labels:
        if l not in mapping:
            mapping[l] = len(mapping) + 1
        out.append(mapping[l])
    return np.array(out)


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


def test_average_linkage_end_to_end():
    """Full C++ path vs full Python path should produce identical linkage matrices."""
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

    assert np.allclose(Z_py, Z_cpp)


def test_average_linkage_small():
    """Sanity check with a trivial 3-point case."""
    # 3 points: d(0,1)=1, d(0,2)=5, d(1,2)=3 — no ties
    dist = np.array([1.0, 5.0, 3.0])
    Z_py = fastcluster.linkage(dist.copy(), method="average")
    Z_cpp = vbx_native.average_linkage(dist, 3)

    np.testing.assert_allclose(Z_cpp[:, 2], Z_py[:, 2], atol=1e-15)
    np.testing.assert_allclose(Z_cpp[:, 3], Z_py[:, 3], atol=1e-15)


def test_fcluster_distance_vs_scipy():
    """C++ fcluster_distance should match scipy fcluster for various thresholds."""
    rng = np.random.default_rng(123)
    N, D = 30, 64
    xvecs = rng.standard_normal((N, D))

    sim = vbx_native.cosine_similarity(xvecs)
    Z = vbx_native.average_linkage(-sim, N)

    # Shift to non-negative heights (required by scipy fcluster)
    Z[:, 2] += abs(Z[:, 2].min())

    for t in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        labels_scipy = _normalize_labels(fcluster(Z.copy(), t, criterion="distance"))
        labels_cpp = _normalize_labels(vbx_native.fcluster_distance(Z, t))
        assert np.array_equal(labels_scipy, labels_cpp), \
            f"Partitions differ at threshold {t}: scipy={labels_scipy}, cpp={labels_cpp}"


def test_fcluster_distance_label_order_differs():
    """Demonstrate that C++ and scipy assign different label numbers to the same partition."""
    # 4 points: d(2,3)=1, d(0,1)=2, all others=10
    # Merge order: {2,3} first, then {0,1}, then everything.
    # Cut at t=5 → two clusters: {0,1} and {2,3}
    dist = np.array([2.0, 10.0, 10.0, 10.0, 10.0, 1.0])
    Z = vbx_native.average_linkage(dist, 4)

    labels_scipy = fcluster(Z, 5.0, criterion="distance")
    labels_cpp = np.array(vbx_native.fcluster_distance(Z, 5.0))

    # Raw labels differ: scipy visits smaller cluster index first (top-down),
    # C++ scans leaves left-to-right.
    assert not np.array_equal(labels_scipy, labels_cpp), \
        "Expected different raw labels (if this fails, the test premise is wrong)"

    # But normalized labels match — same partition.
    assert np.array_equal(_normalize_labels(labels_scipy), _normalize_labels(labels_cpp))


def test_fcluster_distance_all_one_cluster():
    """Threshold above max height should put everything in one cluster."""
    dist = np.array([1.0, 2.0, 3.0])
    Z = vbx_native.average_linkage(dist, 3)
    labels = np.array(vbx_native.fcluster_distance(Z, 999.0))
    assert np.all(labels == labels[0])


def test_fcluster_distance_all_singletons():
    """Threshold below min height should give each point its own cluster."""
    dist = np.array([1.0, 2.0, 3.0])
    Z = vbx_native.average_linkage(dist, 3)
    labels = np.array(vbx_native.fcluster_distance(Z, Z[0, 2] - 0.01))
    assert len(set(labels)) == 3
