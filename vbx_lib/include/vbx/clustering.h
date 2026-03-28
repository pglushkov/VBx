#ifndef VBX_CLUSTERING_H_
#define VBX_CLUSTERING_H_

#include <vector>
#include "vbx/types.h"

namespace vbx {

// ---------------------------------------------------------------------------
// Two-Gaussian calibration result
// ---------------------------------------------------------------------------

template <typename Scalar>
struct CalibrationResultT {
    Scalar threshold;
    std::vector<Scalar> calibrated;
};

using CalibrationResultF = CalibrationResultT<float>;
using CalibrationResultD = CalibrationResultT<double>;

// ---------------------------------------------------------------------------
// AHC params
// ---------------------------------------------------------------------------

struct AhcParams {
    double threshold = -0.015;
    // Linkage is always average (UPGMA), matching reference implementation.
};

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

// Compute condensed upper-triangular cosine similarities (no diagonal).
// Returns N*(N-1)/2 elements in row-major upper triangle order
// (same layout as scipy.spatial.distance.squareform).
template <typename Scalar>
std::vector<Scalar> cosine_similarity(MatrixViewT<Scalar> xvecs);

// Fit 2-component GMM with shared variance on scores,
// return intersection threshold and calibrated log-odds scores.
template <typename Scalar>
CalibrationResultT<Scalar> two_gmm_calibrate(const Scalar* scores, int n,
                                              int niters = 20);

// ---------------------------------------------------------------------------
// Linkage result — scipy convention, (n-1) x 4 row-major
// ---------------------------------------------------------------------------

struct LinkageResult {
    // (n-1) x 4 row-major: [idx1, idx2, height, size] per step.
    // Indices: leaves 0..n-1, merged clusters n, n+1, ...
    // idx1 < idx2 in each row.
    std::vector<double> data;
    int steps = 0;  // n - 1

    double* row(int i) { return data.data() + i * 4; }
    const double* row(int i) const { return data.data() + i * 4; }
};

// Average linkage on condensed distance matrix (N*(N-1)/2 elements).
// Copies distmat internally — safe when caller still needs the data.
LinkageResult average_linkage(const double* distmat, int n);

// In-place variant — destroys distmat. Zero-copy for internal pipelines.
LinkageResult average_linkage_inplace(double* distmat, int n);

// AHC on condensed cosine similarity vector.
// Returns N cluster labels (0-based).
template <typename Scalar>
std::vector<int> ahc_cluster(CondensedMatrixViewT<Scalar> sim,
                             const AhcParams& params = {});

// ---------------------------------------------------------------------------
// Explicit instantiations (defined in clustering.cpp)
// ---------------------------------------------------------------------------

extern template std::vector<float> cosine_similarity<float>(MatrixViewF);
extern template std::vector<double> cosine_similarity<double>(MatrixViewD);
extern template CalibrationResultF two_gmm_calibrate<float>(const float*, int, int);
extern template CalibrationResultD two_gmm_calibrate<double>(const double*, int, int);
extern template std::vector<int> ahc_cluster<float>(CondensedMatrixViewF, const AhcParams&);
extern template std::vector<int> ahc_cluster<double>(CondensedMatrixViewD, const AhcParams&);

}  // namespace vbx

#endif  // VBX_CLUSTERING_H_
