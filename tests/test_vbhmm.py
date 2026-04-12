import numpy as np
import pytest
import vbx_native

from VBx.kaldi_utils import PLDAParams
from VBx.VBx import VBx_plda
from VBx.VBx import forward_backward as fb_py


def _random_hmm_inputs(T, S, rng):
    """Build random HMM inputs: log-likelihoods, row-stochastic transition, uniform-ish ip."""
    lls = rng.standard_normal((T, S))
    tr = rng.random((S, S))
    tr /= tr.sum(axis=1, keepdims=True)
    ip = rng.random(S)
    ip /= ip.sum()
    return lls, tr, ip


def test_forward_backward_vs_python():
    """Random (T=50, S=4) HMM — all four outputs match Python reference."""
    rng = np.random.default_rng(42)
    lls, tr, ip = _random_hmm_inputs(T=50, S=4, rng=rng)

    py_post, py_tll, py_lfw, py_lbw = fb_py(lls, tr, ip)
    cpp_post, cpp_tll, cpp_lfw, cpp_lbw = vbx_native.forward_backward(lls, tr, ip)

    np.testing.assert_allclose(cpp_lfw, py_lfw, atol=1e-10)
    np.testing.assert_allclose(cpp_lbw, py_lbw, atol=1e-10)
    np.testing.assert_allclose(cpp_tll, py_tll, atol=1e-10)
    np.testing.assert_allclose(cpp_post, py_post, atol=1e-10)


def test_forward_backward_single_state():
    """S=1: all mass on single state, posterior is identically 1.0."""
    rng = np.random.default_rng(0)
    T = 10
    lls = rng.standard_normal((T, 1))
    tr = np.ones((1, 1))
    ip = np.ones(1)

    py_post, py_tll, _, _ = fb_py(lls, tr, ip)
    cpp_post, cpp_tll, _, _ = vbx_native.forward_backward(lls, tr, ip)

    np.testing.assert_allclose(cpp_post, py_post, atol=1e-10)
    np.testing.assert_allclose(cpp_tll, py_tll, atol=1e-10)


def test_forward_backward_single_frame():
    """T=1: degenerate (no transitions), posterior is softmax over lls[0] + log(ip+eps)."""
    rng = np.random.default_rng(1)
    lls, tr, ip = _random_hmm_inputs(T=1, S=5, rng=rng)

    py_post, py_tll, py_lfw, py_lbw = fb_py(lls, tr, ip)
    cpp_post, cpp_tll, cpp_lfw, cpp_lbw = vbx_native.forward_backward(lls, tr, ip)

    np.testing.assert_allclose(cpp_post, py_post, atol=1e-10)
    np.testing.assert_allclose(cpp_tll, py_tll, atol=1e-10)
    np.testing.assert_allclose(cpp_lfw, py_lfw, atol=1e-10)
    np.testing.assert_allclose(cpp_lbw, py_lbw, atol=1e-10)


def test_forward_backward_two_frames():
    """T=2: first real forward/backward step exercised."""
    rng = np.random.default_rng(2)
    lls, tr, ip = _random_hmm_inputs(T=2, S=3, rng=rng)

    py_post, py_tll, _, _ = fb_py(lls, tr, ip)
    cpp_post, cpp_tll, _, _ = vbx_native.forward_backward(lls, tr, ip)

    np.testing.assert_allclose(cpp_post, py_post, atol=1e-10)
    np.testing.assert_allclose(cpp_tll, py_tll, atol=1e-10)


def test_forward_backward_posteriors_normalized():
    """Per-frame posteriors must sum to 1."""
    rng = np.random.default_rng(3)
    lls, tr, ip = _random_hmm_inputs(T=25, S=6, rng=rng)

    cpp_post, _, _, _ = vbx_native.forward_backward(lls, tr, ip)
    np.testing.assert_allclose(cpp_post.sum(axis=1), 1.0, atol=1e-10)


def test_forward_backward_float32():
    """float32 path matches Python within fp32 tolerance."""
    rng = np.random.default_rng(7)
    lls, tr, ip = _random_hmm_inputs(T=30, S=3, rng=rng)

    py_post, py_tll, _, _ = fb_py(lls, tr, ip)
    cpp_post, cpp_tll, _, _ = vbx_native.forward_backward(
        lls.astype(np.float32), tr.astype(np.float32), ip.astype(np.float32)
    )

    np.testing.assert_allclose(cpp_post, py_post, atol=1e-5)
    np.testing.assert_allclose(cpp_tll, py_tll, rtol=1e-5)


# ---------------------------------------------------------------------------
# vbhmm() — full VBx_plda iteration loop
# ---------------------------------------------------------------------------


def _make_dummy_plda_inputs(T, D, S, rng):
    """Build dummy x-vectors + PLDA params with D_raw == lda_dim (no slicing).

    The transform is a random orthogonal matrix so the PLDA projection is
    well-conditioned. psi is positive (drawn from |N(0,1)| + 0.5).
    """
    xvecs = rng.standard_normal((T, D))
    plda_mean = rng.standard_normal(D) * 0.1
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    plda_transform = q
    plda_psi = np.abs(rng.standard_normal(D)) + 0.5

    plda_params = PLDAParams(
        dim=D,
        mean=plda_mean,
        transform=plda_transform,
        psi=plda_psi,
    )

    # Initial responsibilities: random row-stochastic (T, S).
    gamma_init = rng.random((T, S))
    gamma_init = gamma_init / gamma_init.sum(axis=1, keepdims=True)
    return xvecs, gamma_init, plda_params


def test_vbhmm_plda_vs_python():
    """C++ vbhmm() must match VBx.VBx.VBx_plda on dummy PLDA inputs."""
    rng = np.random.default_rng(2024)
    T, D, S = 40, 8, 3
    xvecs, gamma_init, plda = _make_dummy_plda_inputs(T, D, S, rng)

    loop_prob = 0.9
    Fa, Fb = 0.3, 17.0
    max_iters = 20
    epsilon = 1e-6

    py_gamma, py_pi, py_Li = VBx_plda(
        xvecs,
        plda,
        loopProb=loop_prob,
        Fa=Fa,
        Fb=Fb,
        pi=S,  # int -> uniform 1/S initial speaker priors
        gamma=gamma_init,
        maxIters=max_iters,
        epsilon=epsilon,
    )
    py_elbo = np.array([item[0] for item in py_Li])

    cpp_gamma, cpp_pi, cpp_elbo = vbx_native.vbhmm(
        xvecs,
        gamma_init,
        plda.mean,
        plda.transform,
        plda.psi,
        loop_prob=loop_prob,
        Fa=Fa,
        Fb=Fb,
        max_iters=max_iters,
        epsilon=epsilon,
    )

    # Same number of iterations (convergence hits at the same step).
    assert cpp_elbo.shape == py_elbo.shape

    np.testing.assert_allclose(cpp_elbo, py_elbo, atol=1e-10)
    np.testing.assert_allclose(cpp_pi, py_pi, atol=1e-10)
    np.testing.assert_allclose(cpp_gamma, py_gamma, atol=1e-10)


# ---------------------------------------------------------------------------
# Tougher vbhmm tests — exercise paths the simple square-D test misses
# ---------------------------------------------------------------------------


def _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng):
    """Build PLDA inputs where D_raw > lda_dim, matching real Kaldi usage.

    The transform has shape (D_raw, D_raw) — full eigenvector matrix — just
    like read_plda_params_from_kaldi_format produces (wccn.T[::-1]).
    Python's VBx_plda slices the projection to [:, :lda_dim]; C++ relies on
    plda.lda_dim to limit the inner loop.  psi has length lda_dim (pre-sliced).
    """
    xvecs = rng.standard_normal((T, D_raw))
    plda_mean = rng.standard_normal(D_raw) * 0.3

    # Full-rank transform (D_raw x D_raw) — not pre-sliced to lda_dim rows.
    q, _ = np.linalg.qr(rng.standard_normal((D_raw, D_raw)))
    plda_transform = q

    # psi only has lda_dim entries (the top eigenvalues after reversal+slice).
    plda_psi = np.abs(rng.standard_normal(lda_dim)) + 0.5

    plda_params = PLDAParams(
        dim=lda_dim,
        mean=plda_mean,
        transform=plda_transform,
        psi=plda_psi,
    )

    gamma_init = rng.random((T, S))
    gamma_init = gamma_init / gamma_init.sum(axis=1, keepdims=True)
    return xvecs, gamma_init, plda_params


def _run_cpp_vs_python(
    xvecs, gamma_init, plda, S, loop_prob, Fa, Fb, max_iters, epsilon, atol=1e-10
):
    """Run both backends and assert all outputs match."""
    py_gamma, py_pi, py_Li = VBx_plda(
        xvecs,
        plda,
        loopProb=loop_prob,
        Fa=Fa,
        Fb=Fb,
        pi=S,
        gamma=gamma_init.copy(),
        maxIters=max_iters,
        epsilon=epsilon,
    )
    py_elbo = np.array([item[0] for item in py_Li])

    cpp_gamma, cpp_pi, cpp_elbo = vbx_native.vbhmm(
        np.ascontiguousarray(xvecs),
        np.ascontiguousarray(gamma_init.copy()),
        np.ascontiguousarray(plda.mean),
        np.ascontiguousarray(plda.transform),
        np.ascontiguousarray(plda.psi),
        loop_prob=loop_prob,
        Fa=Fa,
        Fb=Fb,
        max_iters=max_iters,
        epsilon=epsilon,
    )

    # Debug: print ELBO histories before asserting
    print(f"\nPython ELBO ({len(py_elbo)} iters): {py_elbo}")
    print(f"\nC++ ELBO ({len(cpp_elbo)} iters): {cpp_elbo}")
    if len(py_elbo) > 0 and len(cpp_elbo) > 0:
        n_common = min(len(py_elbo), len(cpp_elbo))
        print(
            f"\nFirst {n_common} ELBO diffs: {cpp_elbo[:n_common] - py_elbo[:n_common]}"
        )

    assert cpp_elbo.shape == py_elbo.shape, (
        f"Iteration count mismatch: C++ {cpp_elbo.shape[0]} vs Python {py_elbo.shape[0]}"
    )
    np.testing.assert_allclose(
        cpp_elbo, py_elbo, atol=atol, err_msg="ELBO history diverged"
    )
    np.testing.assert_allclose(
        cpp_pi, py_pi, atol=atol, err_msg="Speaker priors diverged"
    )
    np.testing.assert_allclose(
        cpp_gamma, py_gamma, atol=atol, err_msg="Posteriors (gamma) diverged"
    )


def test_vbhmm_nonsquare_transform():
    """D_raw=32 > lda_dim=8: exercises the real PLDA projection/slicing path."""
    rng = np.random.default_rng(7777)
    T, D_raw, lda_dim, S = 60, 32, 8, 4
    xvecs, gamma_init, plda = _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng)

    _run_cpp_vs_python(
        xvecs,
        gamma_init,
        plda,
        S,
        loop_prob=0.9,
        Fa=0.3,
        Fb=17.0,
        max_iters=30,
        epsilon=1e-6,
    )


def test_vbhmm_many_speakers():
    """S=10 speakers, some should get pruned (pi -> 0) by VB."""
    rng = np.random.default_rng(4242)
    T, D_raw, lda_dim, S = 80, 16, 8, 10
    xvecs, gamma_init, plda = _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng)

    _run_cpp_vs_python(
        xvecs,
        gamma_init,
        plda,
        S,
        loop_prob=0.99,
        Fa=0.3,
        Fb=17.0,
        max_iters=40,
        epsilon=1e-6,
    )


def test_vbhmm_high_Fa():
    """High Fa stresses the log-likelihood magnitudes — catches overflow bugs."""
    rng = np.random.default_rng(9999)
    T, D_raw, lda_dim, S = 50, 24, 12, 3
    xvecs, gamma_init, plda = _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng)

    _run_cpp_vs_python(
        xvecs,
        gamma_init,
        plda,
        S,
        loop_prob=0.9,
        Fa=1.0,
        Fb=1.0,
        max_iters=30,
        epsilon=1e-6,
    )


def test_vbhmm_two_frames():
    """T=2: minimal sequence — forward-backward has exactly one step."""
    rng = np.random.default_rng(1111)
    T, D_raw, lda_dim, S = 2, 16, 8, 3
    xvecs, gamma_init, plda = _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng)

    _run_cpp_vs_python(
        xvecs,
        gamma_init,
        plda,
        S,
        loop_prob=0.9,
        Fa=0.3,
        Fb=17.0,
        max_iters=20,
        epsilon=1e-6,
    )


def test_vbhmm_single_speaker():
    """S=1: degenerate — gamma must be all 1s, no speaker competition."""
    rng = np.random.default_rng(5555)
    T, D_raw, lda_dim, S = 30, 16, 8, 1
    xvecs, gamma_init, plda = _make_realistic_plda_inputs(T, D_raw, lda_dim, S, rng)

    _run_cpp_vs_python(
        xvecs,
        gamma_init,
        plda,
        S,
        loop_prob=0.9,
        Fa=0.3,
        Fb=17.0,
        max_iters=20,
        epsilon=1e-6,
    )
