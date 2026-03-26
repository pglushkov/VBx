#include "vbx/clustering.h"

namespace vbx {

template <typename Scalar>
std::vector<Scalar> cosine_similarity(MatrixViewT<Scalar>) { return {}; }

template <typename Scalar>
CalibrationResultT<Scalar> two_gmm_calibrate(const Scalar*, int, int) { return {}; }

template <typename Scalar>
std::vector<int> ahc_cluster(CondensedMatrixViewT<Scalar>, const AhcParams&) { return {}; }

template std::vector<float> cosine_similarity<float>(MatrixViewF);
template std::vector<double> cosine_similarity<double>(MatrixViewD);
template CalibrationResultF two_gmm_calibrate<float>(const float*, int, int);
template CalibrationResultD two_gmm_calibrate<double>(const double*, int, int);
template std::vector<int> ahc_cluster<float>(CondensedMatrixViewF, const AhcParams&);
template std::vector<int> ahc_cluster<double>(CondensedMatrixViewD, const AhcParams&);

}  // namespace vbx
