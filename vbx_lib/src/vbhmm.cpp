#include "vbx/vbhmm.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vbx/linalg.h"

namespace vbx {

// Forward-backward over a left-to-right HMM in log space.
// Mirrors VBx/VBx.py::forward_backward exactly, including the `eps = 1e-8`
// smoothing applied to the transition matrix and initial probability vector
// before taking logs (guards against log(0) when a state is fully forbidden).
//
// Shapes: log_likelihoods (T x S), transition (S x S), initial (S,).
template <typename Scalar>
ForwardBackwardResultT<Scalar> forward_backward(MatrixViewT<Scalar> log_likelihoods,
                                                 MatrixViewT<Scalar> transition,
                                                 const Scalar* initial)
{
    const int T = log_likelihoods.rows;
    const int S = log_likelihoods.cols;
    assert(T > 0 && S > 0);
    assert(transition.rows == S && transition.cols == S);

    constexpr Scalar kEps = static_cast<Scalar>(1e-8);

    ForwardBackwardResultT<Scalar> result;
    result.posteriors = MatrixT<Scalar>(T, S);
    result.log_fw     = MatrixT<Scalar>(T, S);
    result.log_bw     = MatrixT<Scalar>(T, S);

    // ltr = log(transition + eps), shape (S x S), row-major.
    std::vector<Scalar> ltr(S * S);
    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            ltr[i * S + j] = std::log(transition(i, j) + kEps);
        }
    }

    // lfw[0] = lls[0] + log(initial + eps)
    for (int s = 0; s < S; ++s) {
        result.log_fw(0, s) = log_likelihoods(0, s) + std::log(initial[s] + kEps);
    }

    // lbw[T-1] = 0
    for (int s = 0; s < S; ++s) {
        result.log_bw(T - 1, s) = Scalar{0};
    }

    // Forward recursion:
    //   lfw[t][i] = lls[t][i] + logsumexp_k( lfw[t-1][k] + ltr[k][i] )
    std::vector<Scalar> scratch(S);
    for (int t = 1; t < T; ++t) {
        for (int i = 0; i < S; ++i) {
            for (int k = 0; k < S; ++k) {
                scratch[k] = result.log_fw(t - 1, k) + ltr[k * S + i];
            }
            result.log_fw(t, i) = log_likelihoods(t, i) +
                                   linalg::logsumexp<Scalar>(scratch.data(), S);
        }
    }

    // Backward recursion:
    //   lbw[t][i] = logsumexp_j( ltr[i][j] + lls[t+1][j] + lbw[t+1][j] )
    for (int t = T - 2; t >= 0; --t) {
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                scratch[j] = ltr[i * S + j]
                             + log_likelihoods(t + 1, j)
                             + result.log_bw(t + 1, j);
            }
            result.log_bw(t, i) = linalg::logsumexp<Scalar>(scratch.data(), S);
        }
    }

    // Total log-likelihood from the last forward row.
    result.total_log_lik = linalg::logsumexp<Scalar>(
        &result.log_fw.storage[(T - 1) * S], S);

    // Posteriors: exp(lfw + lbw - tll)
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < S; ++s) {
            result.posteriors(t, s) = std::exp(
                result.log_fw(t, s) + result.log_bw(t, s) - result.total_log_lik);
        }
    }

    return result;
}

// Run the full VBx iteration loop on x-vectors. Port of VBx/VBx.py::VBx_plda.
// Only the PLDA path is implemented at this stage — the PLDA-less path
// (uniform pseudo_phi = 1/D from VBx/VBx.py::VBx) will follow later.
//
// Assumes `plda.transform` has shape (lda_dim, D_raw), i.e. already sliced
// to the LDA output dimensionality. `load_plda()` is still stubbed out, so
// this convention can be formalised there when it lands.
template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(MatrixViewT<Scalar> xvecs,
                            MatrixViewT<Scalar> gamma_init,
                            const VbhmmParams& params,
                            const std::optional<PldaModelT<Scalar>>& plda)
{
    if (!plda.has_value()) {
        throw std::invalid_argument(
            "vbx::vbhmm: PLDA-less path is not implemented yet");
    }
    const PldaModelT<Scalar>& M = *plda;

    // log(2*pi) — avoids relying on the non-standard M_PI macro.
    constexpr double kLog2Pi = 1.8378770664093454835606594728112;

    // -----------------------------------------------------------------------
    // Shape extraction + sanity checks.
    // -----------------------------------------------------------------------
    const int T    = xvecs.rows;
    const int Draw = xvecs.cols;
    const int D    = M.lda_dim;            // projected feature dim
    const int S    = gamma_init.cols;      // number of speakers
    assert(T > 0 && S > 0 && D > 0 && Draw > 0);
    assert(gamma_init.rows == T);
    assert(static_cast<int>(M.mean.size()) == Draw);
    assert(M.transform.rows == D && M.transform.cols == Draw);
    assert(static_cast<int>(M.diag_ac.size()) == D);

    // -----------------------------------------------------------------------
    // Scalar-cast of (double) config once at entry.
    // -----------------------------------------------------------------------
    const Scalar Fa        = static_cast<Scalar>(params.Fa);
    const Scalar Fb        = static_cast<Scalar>(params.Fb);
    const Scalar loopProb  = static_cast<Scalar>(params.loop_prob);
    const Scalar eps_conv  = static_cast<Scalar>(params.epsilon);
    const int    max_iters = params.max_iters;
    const Scalar FaFb      = Fa / Fb;
    const Scalar one_minus_loop = Scalar{1} - loopProb;

    // -----------------------------------------------------------------------
    // 1) PLDA projection: X = (xvecs - mean) @ transform^T, shape (T, D).
    //    Python:  X = (x_vecs - plda_mu).dot(plda_tr.T)[:, :lda_dim]
    // -----------------------------------------------------------------------
    MatrixT<Scalar> X(T, D);
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
            Scalar acc = Scalar{0};
            for (int k = 0; k < Draw; ++k) {
                acc += (xvecs(t, k) - M.mean[k]) * M.transform(d, k);
            }
            X(t, d) = acc;
        }
    }

    // -----------------------------------------------------------------------
    // 2) G[t] = -0.5 * (||X[t]||^2 + D * log(2*pi))   — per-frame constant
    // -----------------------------------------------------------------------
    const Scalar D_log_2pi = static_cast<Scalar>(static_cast<double>(D) * kLog2Pi);
    std::vector<Scalar> G(T);
    for (int t = 0; t < T; ++t) {
        Scalar sq = Scalar{0};
        for (int d = 0; d < D; ++d) {
            sq += X(t, d) * X(t, d);
        }
        G[t] = Scalar{-0.5} * (sq + D_log_2pi);
    }

    // -----------------------------------------------------------------------
    // 3) V = sqrt(Phi), rho = X * V (row-wise scale)
    // -----------------------------------------------------------------------
    const Scalar* Phi = M.diag_ac.data();
    std::vector<Scalar> V(D);
    for (int d = 0; d < D; ++d) V[d] = std::sqrt(Phi[d]);

    MatrixT<Scalar> rho(T, D);
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
            rho(t, d) = X(t, d) * V[d];
        }
    }

    // -----------------------------------------------------------------------
    // 4) State: gamma <- gamma_init (copy), pi <- uniform(S).
    //    Python's run_vbhmm always passes `pi=qinit.shape[1]` (int -> uniform).
    // -----------------------------------------------------------------------
    MatrixT<Scalar> gamma(T, S);
    for (int t = 0; t < T; ++t) {
        for (int s = 0; s < S; ++s) {
            gamma(t, s) = gamma_init(t, s);
        }
    }
    std::vector<Scalar> pi(S, Scalar{1} / static_cast<Scalar>(S));

    // -----------------------------------------------------------------------
    // Scratch buffers reused across iterations.
    // -----------------------------------------------------------------------
    MatrixT<Scalar> invL(S, D);
    MatrixT<Scalar> alpha(S, D);
    MatrixT<Scalar> log_p(T, S);
    MatrixT<Scalar> tr_mat(S, S);
    std::vector<Scalar> gamma_sum(S);
    std::vector<Scalar> half_ip(S);   // 0.5 * (invL + alpha^2) @ Phi

    std::vector<Scalar> elbo_history;
    elbo_history.reserve(max_iters);

    // -----------------------------------------------------------------------
    // Main VB iteration loop.
    // -----------------------------------------------------------------------
    for (int it = 0; it < max_iters; ++it) {
        // --- gamma_sum[s] = sum_t gamma[t, s]
        for (int s = 0; s < S; ++s) gamma_sum[s] = Scalar{0};
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < S; ++s) {
                gamma_sum[s] += gamma(t, s);
            }
        }

        // --- invL[s, d] = 1 / (1 + (Fa/Fb) * gamma_sum[s] * Phi[d])   (eq 17)
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) {
                invL(s, d) = Scalar{1} /
                              (Scalar{1} + FaFb * gamma_sum[s] * Phi[d]);
            }
        }

        // --- alpha[s, d] = (Fa/Fb) * invL[s, d] * (gamma^T @ rho)[s, d]  (eq 16)
        //     First compute gamma^T @ rho into alpha, then scale in place.
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) alpha(s, d) = Scalar{0};
        }
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < S; ++s) {
                const Scalar g = gamma(t, s);
                for (int d = 0; d < D; ++d) {
                    alpha(s, d) += g * rho(t, d);
                }
            }
        }
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) {
                alpha(s, d) *= FaFb * invL(s, d);
            }
        }

        // --- half_ip[s] = 0.5 * sum_d (invL[s, d] + alpha[s, d]^2) * Phi[d]
        //     This is the per-speaker constant subtracted in log_p (eq 23).
        for (int s = 0; s < S; ++s) {
            Scalar acc = Scalar{0};
            for (int d = 0; d < D; ++d) {
                acc += (invL(s, d) + alpha(s, d) * alpha(s, d)) * Phi[d];
            }
            half_ip[s] = Scalar{0.5} * acc;
        }

        // --- log_p[t, s] = Fa * (<rho[t], alpha[s]> - half_ip[s] + G[t])   (eq 23)
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < S; ++s) {
                Scalar dot = Scalar{0};
                for (int d = 0; d < D; ++d) {
                    dot += rho(t, d) * alpha(s, d);
                }
                log_p(t, s) = Fa * (dot - half_ip[s] + G[t]);
            }
        }

        // --- tr[i, j] = loopProb * (i==j) + (1-loopProb) * pi[j]          (eq 1)
        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                tr_mat(i, j) = one_minus_loop * pi[j];
            }
            tr_mat(i, i) += loopProb;
        }

        // --- Forward-backward:
        //       (gamma, log_pX, lfw, lbw) = forward_backward(log_p, tr, pi)
        auto fb = forward_backward<Scalar>(log_p, tr_mat, pi.data());

        // Update gamma in place from posteriors.
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < S; ++s) {
                gamma(t, s) = fb.posteriors(t, s);
            }
        }

        // --- ELBO = log_pX + (Fb/2) * sum(log(invL) - invL - alpha^2 + 1)  (eq 25)
        Scalar elbo_corr = Scalar{0};
        for (int s = 0; s < S; ++s) {
            for (int d = 0; d < D; ++d) {
                const Scalar il = invL(s, d);
                const Scalar a  = alpha(s, d);
                elbo_corr += std::log(il) - il - a * a + Scalar{1};
            }
        }
        const Scalar ELBO = fb.total_log_lik + Fb * Scalar{0.5} * elbo_corr;
        elbo_history.push_back(ELBO);

        // --- pi update (eq 24):
        //   pi_new[s] = gamma[0, s]
        //             + (1 - loopProb) * pi[s] *
        //               sum_{t=1..T-1} exp( logsumexp(lfw[t-1])
        //                                    + log_p[t, s]
        //                                    + lbw[t, s]
        //                                    - log_pX )
        std::vector<Scalar> pi_flow(S, Scalar{0});
        for (int t = 1; t < T; ++t) {
            const Scalar lse_prev =
                linalg::logsumexp<Scalar>(&fb.log_fw(t - 1, 0), S);
            for (int s = 0; s < S; ++s) {
                const Scalar lv = lse_prev + log_p(t, s) + fb.log_bw(t, s)
                                  - fb.total_log_lik;
                pi_flow[s] += std::exp(lv);
            }
        }
        Scalar pi_sum = Scalar{0};
        std::vector<Scalar> pi_new(S);
        for (int s = 0; s < S; ++s) {
            pi_new[s] = fb.posteriors(0, s)
                        + one_minus_loop * pi[s] * pi_flow[s];
            pi_sum += pi_new[s];
        }
        for (int s = 0; s < S; ++s) {
            pi[s] = pi_new[s] / pi_sum;
        }

        // --- Convergence: stop as soon as ELBO gain < epsilon (post first iter).
        if (it > 0) {
            const Scalar delta = ELBO - elbo_history[it - 1];
            if (delta < eps_conv) {
                // Note: a negative delta means the bound went down — Python
                // prints a warning here; we silently break for now.
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Package result.
    // -----------------------------------------------------------------------
    VbhmmResultT<Scalar> result;
    result.posteriors     = std::move(gamma);
    result.speaker_priors = std::move(pi);
    result.elbo_history   = std::move(elbo_history);
    return result;
}

template ForwardBackwardResultF forward_backward<float>(MatrixViewF, MatrixViewF, const float*);
template ForwardBackwardResultD forward_backward<double>(MatrixViewD, MatrixViewD, const double*);
template VbhmmResultF vbhmm<float>(MatrixViewF, MatrixViewF, const VbhmmParams&, const std::optional<PldaModelF>&);
template VbhmmResultD vbhmm<double>(MatrixViewD, MatrixViewD, const VbhmmParams&, const std::optional<PldaModelD>&);

}  // namespace vbx
