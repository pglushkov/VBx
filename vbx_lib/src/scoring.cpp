#include "vbx/scoring.h"

namespace vbx {

template <typename Scalar>
PldaModelT<Scalar> load_plda(const char*) { return {}; }

template <typename Scalar>
MatrixT<Scalar> score_cosine(MatrixViewT<Scalar>) { return {}; }

template <typename Scalar>
MatrixT<Scalar> score_plda(MatrixViewT<Scalar>, const PldaModelT<Scalar>&) { return {}; }

template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_cosine(MatrixViewT<Scalar>, Scalar) { return {}; }

template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_plda(MatrixViewT<Scalar>, const PldaModelT<Scalar>&, Scalar) { return {}; }

}  // namespace vbx
