#ifndef VBX_SCORING_H_
#define VBX_SCORING_H_

#include <vector>
#include "vbx/types.h"

namespace vbx {

enum class ScoringMode { kCosine, kPlda };

template <typename Scalar>
struct PldaModelT {
    MatrixT<Scalar>     transform;
    std::vector<Scalar> mean;
    std::vector<Scalar> diag_ac;
    int                 lda_dim = 0;
};

using PldaModelF = PldaModelT<float>;
using PldaModelD = PldaModelT<double>;
using PldaModel  = PldaModelD;

template <typename Scalar>
PldaModelT<Scalar> load_plda(const char* path);

template <typename Scalar>
MatrixT<Scalar> score_cosine(MatrixViewT<Scalar> xvecs);

template <typename Scalar>
MatrixT<Scalar> score_plda(MatrixViewT<Scalar> xvecs, const PldaModelT<Scalar>& plda);

template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_cosine(MatrixViewT<Scalar> xvecs, Scalar Fa);

template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_plda(MatrixViewT<Scalar> xvecs,
                                              const PldaModelT<Scalar>& plda,
                                              Scalar Fa);

}  // namespace vbx

#endif  // VBX_SCORING_H_
