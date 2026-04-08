import numpy as np

from VBx.VBx import forward_backward as fb_py
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
