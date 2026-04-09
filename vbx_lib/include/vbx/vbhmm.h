#ifndef VBX_VBHMM_H_
#define VBX_VBHMM_H_

#include <optional>
#include <vector>
#include "vbx/types.h"
#include "vbx/scoring.h"

namespace vbx {

struct VbhmmParams {
    double loop_prob      = 0.99;
    double Fa             = 0.3;
    double Fb             = 17.0;
    int    max_speakers   = 10;
    int    max_iters      = 40;
    double epsilon        = 1e-6;
    double alpha_q_init   = 1.0;
    double init_smoothing = 5.0;
};

template <typename Scalar>
struct ForwardBackwardResultT {
    MatrixT<Scalar> posteriors;
    Scalar          total_log_lik;
    MatrixT<Scalar> log_fw;
    MatrixT<Scalar> log_bw;
};

using ForwardBackwardResultF = ForwardBackwardResultT<float>;
using ForwardBackwardResultD = ForwardBackwardResultT<double>;
using ForwardBackwardResult  = ForwardBackwardResultD;

template <typename Scalar>
struct VbhmmResultT {
    MatrixT<Scalar>     posteriors;
    std::vector<Scalar> speaker_priors;
    std::vector<Scalar> elbo_history;
};

using VbhmmResultF = VbhmmResultT<float>;
using VbhmmResultD = VbhmmResultT<double>;
using VbhmmResult  = VbhmmResultD;

template <typename Scalar>
ForwardBackwardResultT<Scalar> forward_backward(MatrixViewT<Scalar> log_likelihoods,
                                                 MatrixViewT<Scalar> transition,
                                                 const Scalar* initial);

// Run the full VBx iteration loop. If `plda` is provided, x-vectors are
// projected via the PLDA transform and Phi=psi is used as per VBx.VBx_plda;
// otherwise the PLDA-less path (uniform pseudo_phi = 1/D) from VBx.VBx is used.
template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(MatrixViewT<Scalar> xvecs,
                            MatrixViewT<Scalar> gamma_init,
                            const VbhmmParams& params = {},
                            const std::optional<PldaModelT<Scalar>>& plda = std::nullopt);

extern template ForwardBackwardResultF forward_backward<float>(MatrixViewF, MatrixViewF, const float*);
extern template ForwardBackwardResultD forward_backward<double>(MatrixViewD, MatrixViewD, const double*);
extern template VbhmmResultF vbhmm<float>(MatrixViewF, MatrixViewF, const VbhmmParams&, const std::optional<PldaModelF>&);
extern template VbhmmResultD vbhmm<double>(MatrixViewD, MatrixViewD, const VbhmmParams&, const std::optional<PldaModelD>&);

}  // namespace vbx

#endif  // VBX_VBHMM_H_
