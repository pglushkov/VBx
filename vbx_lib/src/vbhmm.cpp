#include "vbx/vbhmm.h"

#include <cassert>
#include <cmath>
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

template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(MatrixViewT<Scalar>, MatrixViewT<Scalar>, const VbhmmParams&) { return {}; }

template ForwardBackwardResultF forward_backward<float>(MatrixViewF, MatrixViewF, const float*);
template ForwardBackwardResultD forward_backward<double>(MatrixViewD, MatrixViewD, const double*);
template VbhmmResultF vbhmm<float>(MatrixViewF, MatrixViewF, const VbhmmParams&);
template VbhmmResultD vbhmm<double>(MatrixViewD, MatrixViewD, const VbhmmParams&);

}  // namespace vbx
