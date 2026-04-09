import numpy as np
from scipy.special import softmax

import vbx_native
from VBx.VBx import VBx_plda
from VBx.vbhmm import get_clusters_from_xvectors
from VBx.diarization_lib import merge_adjacent_labels
from VBx.kaldi_utils import PLDAParams

from utils import normalize_labels


# ---------------------------------------------------------------------------
# Synthetic inputs: K well-separated speaker centroids, time-contiguous runs.
# ---------------------------------------------------------------------------

def _make_inputs(T, D, num_speakers, rng, seg_len=0.25):
    """Build (xvecs, seg_starts, seg_ends) that AHC will cleanly separate."""
    centroids = rng.standard_normal((num_speakers, D)) * 5.0
    # Sorted assignments create time-contiguous speaker runs — closer to what
    # a real diarization pipeline sees, and keeps merge_adjacent_labels
    # actually exercised.
    assignments = np.sort(rng.integers(0, num_speakers, size=T))
    xvecs = centroids[assignments] + rng.standard_normal((T, D)) * 0.05
    seg_starts = np.arange(T, dtype=np.float64) * seg_len
    seg_ends = seg_starts + seg_len
    return xvecs, seg_starts, seg_ends


def _make_plda(D, rng):
    plda_mean = rng.standard_normal(D) * 0.1
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    plda_psi = np.abs(rng.standard_normal(D)) + 0.5
    return PLDAParams(dim=D, mean=plda_mean, transform=q, psi=plda_psi)


# ---------------------------------------------------------------------------
# Python reference — stitches get_clusters_from_xvectors + VBx_plda +
# merge_adjacent_labels, matching run_vbhmm() but fed arrays in-process.
#
# Only the PLDA path is covered: vbhmm()'s cosine branch is not implemented
# in C++ yet, so diarize(plda=None) is not end-to-end testable.
#
# Label numbering between scipy AHC and C++ AHC intentionally differs
# (documented in test_clusterization.test_fcluster_distance_label_order_differs),
# so downstream posteriors / speaker_priors end up with permuted speaker
# columns. We compare the things that are well-defined under that
# permutation:
#   - segment start/end times   (depend on partition, not label numbering)
#   - segment speaker labels    (normalize before comparing)
#   - per-iteration ELBO        (symmetric under speaker relabeling)
# Column-wise comparison of posteriors / priors is already covered by
# test_vbhmm.py where qinit is shared.
# ---------------------------------------------------------------------------

def _python_reference(
    xvecs, seg_starts, seg_ends, plda, *,
    ahc_threshold, loop_prob, Fa, Fb, max_iters, epsilon, init_smoothing,
):
    labels1st = get_clusters_from_xvectors(xvecs, ahc_threshold, use_cpp=False)
    K = int(np.max(labels1st)) + 1
    qinit = np.zeros((len(labels1st), K))
    qinit[np.arange(len(labels1st)), labels1st] = 1.0
    qinit = softmax(qinit * init_smoothing, axis=1)

    q, sp, L = VBx_plda(
        xvecs, plda,
        pi=qinit.shape[1], gamma=qinit,
        maxIters=max_iters, epsilon=epsilon,
        loopProb=loop_prob, Fa=Fa, Fb=Fb,
    )
    elbo = np.array([item[0] for item in L])
    labels = np.argsort(-q, axis=1)[:, 0]

    # merge_adjacent_labels mutates inputs — hand it copies.
    starts, ends, out_labels = merge_adjacent_labels(
        seg_starts.copy(), seg_ends.copy(), labels,
    )
    return starts, ends, out_labels, elbo


_COMMON_KW = dict(
    ahc_threshold=-0.015,
    loop_prob=0.9,
    Fa=0.3,
    Fb=17.0,
    max_iters=20,
    epsilon=1e-6,
    init_smoothing=5.0,
)


def test_diarize_with_plda_vs_python():
    """Full diarize() pipeline (PLDA path) must match Python reference
    up to speaker label permutation."""
    rng = np.random.default_rng(2027)
    xvecs, seg_starts, seg_ends = _make_inputs(T=60, D=16, num_speakers=3, rng=rng)
    plda = _make_plda(D=16, rng=rng)

    py_starts, py_ends, py_spk, py_elbo = _python_reference(
        xvecs, seg_starts, seg_ends, plda=plda, **_COMMON_KW,
    )

    cpp_starts, cpp_ends, cpp_spk, _cpp_q, _cpp_sp, cpp_elbo = vbx_native.diarize(
        xvecs, seg_starts, seg_ends,
        plda_mean=plda.mean,
        plda_transform=plda.transform,
        plda_psi=plda.psi,
        run_vbhmm=True,
        **_COMMON_KW,
    )

    # Same partition -> same run boundaries.
    np.testing.assert_allclose(cpp_starts, py_starts, atol=1e-12)
    np.testing.assert_allclose(cpp_ends,   py_ends,   atol=1e-12)

    # Same partition after canonicalizing speaker numbers.
    np.testing.assert_array_equal(
        normalize_labels(cpp_spk), normalize_labels(py_spk),
    )

    # ELBO is invariant under speaker relabeling and should also signal
    # the same number of VB iterations to convergence.
    assert cpp_elbo.shape == py_elbo.shape
    np.testing.assert_allclose(cpp_elbo, py_elbo, atol=1e-10)


def test_diarize_plda_mismatch_raises():
    """Passing only some of the PLDA fields must be rejected."""
    rng = np.random.default_rng(2029)
    xvecs, seg_starts, seg_ends = _make_inputs(T=20, D=8, num_speakers=2, rng=rng)
    plda = _make_plda(D=8, rng=rng)

    try:
        vbx_native.diarize(
            xvecs, seg_starts, seg_ends,
            plda_mean=plda.mean,
            # plda_transform omitted on purpose
            plda_psi=plda.psi,
        )
    except Exception as exc:
        msg = str(exc)
        assert "plda_mean" in msg or "plda_transform" in msg or "plda_psi" in msg
    else:
        raise AssertionError("Expected an exception for partial PLDA args")
