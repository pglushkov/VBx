#include <cmath>
#include <vector>

#include "vbx/clustering.h"
#include "fastcluster.h"

namespace vbx {

template <typename Scalar>
std::vector<Scalar> cosine_similarity(MatrixViewT<Scalar> xvecs) {
    const int n = xvecs.rows;
    const int d = xvecs.cols;

    // Compute L2 norms
    std::vector<Scalar> norms(n);
    for (int i = 0; i < n; ++i) {
        Scalar sum = 0;
        for (int k = 0; k < d; ++k) {
            Scalar v = xvecs(i, k);
            sum += v * v;
        }
        norms[i] = std::sqrt(sum);
    }

    // Compute condensed upper triangle: for each (i, j) where i < j
    std::vector<Scalar> result(condensed_size(n));
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Scalar dot = 0;
            for (int k = 0; k < d; ++k) {
                dot += xvecs(i, k) * xvecs(j, k);
            }
            Scalar denom = norms[i] * norms[j];
            result[idx++] = (denom > Scalar{1e-32}) ? dot / denom : Scalar{0};
        }
    }
    return result;
}

template <typename Scalar>
CalibrationResultT<Scalar> two_gmm_calibrate(const Scalar*, int, int) { return {}; }

LinkageResult average_linkage_inplace(double* distmat, int n) {
    const int steps = n - 1;
    std::vector<int> merge(steps * 2);
    std::vector<double> height(steps);

    hclust_fast(n, distmat, HCLUST_METHOD_AVERAGE,
                merge.data(), height.data());

    return {std::move(merge), std::move(height)};
}

LinkageResult average_linkage(const double* distmat, int n) {
    std::vector<double> dist_copy(distmat, distmat + condensed_size(n));
    return average_linkage_inplace(dist_copy.data(), n);
}

template <typename Scalar>
std::vector<int> ahc_cluster(CondensedMatrixViewT<Scalar>, const AhcParams&) { return {}; }

template std::vector<float> cosine_similarity<float>(MatrixViewF);
template std::vector<double> cosine_similarity<double>(MatrixViewD);
template CalibrationResultF two_gmm_calibrate<float>(const float*, int, int);
template CalibrationResultD two_gmm_calibrate<double>(const double*, int, int);
template std::vector<int> ahc_cluster<float>(CondensedMatrixViewF, const AhcParams&);
template std::vector<int> ahc_cluster<double>(CondensedMatrixViewD, const AhcParams&);

}  // namespace vbx
