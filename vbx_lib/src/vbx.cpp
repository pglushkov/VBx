#include "vbx/vbx.h"

#include <cassert>
#include <cmath>

namespace vbx {

std::string get_version() { return "0.1.0"; }

template <typename Scalar>
DiarResultT<Scalar> diarize(MatrixViewT<Scalar> xvecs,
                             const Segment* /*segments*/,
                             int num_segments,
                             const VbxParams& params,
                             const std::optional<PldaModelT<Scalar>>& /*plda*/)
{
    assert(xvecs.rows == num_segments &&
           "xvecs row count must match num_segments");

    // -----------------------------------------------------------------------
    // First half: AHC clustering + qinit construction (input for VB-HMM).
    // -----------------------------------------------------------------------

    // 1) Initial cluster labels via AHC on x-vectors (0-based).
    const std::vector<int> labels = ahc_cluster<Scalar>(xvecs, params.ahc);

    // 2) Number of initial speakers = max(labels) + 1.
    const int N = static_cast<int>(labels.size());
    int num_speakers = 0;
    for (int i = 0; i < N; ++i) {
        if (labels[i] + 1 > num_speakers) num_speakers = labels[i] + 1;
    }

    // 3) Build qinit: one_hot(labels) * init_smoothing, then row-wise softmax.
    //    Mirrors Python:  qinit = softmax(one_hot * init_smoothing, axis=1)
    //    With a one-hot input this collapses to:
    //        on-label  -> exp(s) / (exp(s) + K - 1)
    //        off-label -> 1      / (exp(s) + K - 1)
    const Scalar smoothing = static_cast<Scalar>(params.vbhmm.init_smoothing);
    const Scalar exp_s     = std::exp(smoothing);
    const Scalar denom     = exp_s + static_cast<Scalar>(num_speakers - 1);
    const Scalar p_on      = exp_s / denom;
    const Scalar p_off     = Scalar{1} / denom;

    MatrixT<Scalar> qinit(N, num_speakers, p_off);
    for (int i = 0; i < N; ++i) {
        qinit(i, labels[i]) = p_on;
    }

    // TODO(second-half): run VB-HMM (cosine or PLDA per params.scoring and
    // params.run_vbhmm) seeded with `qinit`, extract final labels from
    // posteriors, merge adjacent segments, and populate DiarResult.
    (void)qinit;
    return {};
}

// Explicit template instantiations.
template DiarResultF diarize<float>(MatrixViewF,
                                     const Segment*,
                                     int,
                                     const VbxParams&,
                                     const std::optional<PldaModelF>&);
template DiarResultD diarize<double>(MatrixViewD,
                                      const Segment*,
                                      int,
                                      const VbxParams&,
                                      const std::optional<PldaModelD>&);

}  // namespace vbx
