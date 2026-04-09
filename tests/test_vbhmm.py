import numpy as np

from VBx.VBx import VBx_plda, forward_backward as fb_py
from VBx.kaldi_utils import PLDAParams
import vbx_native


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
