#ifndef VBX_CLUSTERING_H_
#define VBX_CLUSTERING_H_

#include <vector>
#include "vbx/types.h"

namespace vbx {

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

// Estimate Agglomerative Hierachical Clustering distance threshold
// by fitting a 2-component GMM with shared variance to pairwise scores via EM.
// Returns the equal-posterior crossing point.
template <typename Scalar>
Scalar ahc_threshold(const Scalar* scores, int n, int niters = 20);

// ---------------------------------------------------------------------------
// Linkage result — scipy convention, (n-1) x 4 row-major
// ---------------------------------------------------------------------------

class LinkageResult {
    // (n-1) x 4 row-major: [idx1, idx2, height, size] per step.
    // Indices: leaves 0..n-1, merged clusters n, n+1, ...
    // idx1 < idx2 in each row.
    std::vector<double> data_;

public:
    LinkageResult(int steps):data_(steps * 4, 0.0) {}

    int n_steps() const { return static_cast<int>(data_.size()/4 ); }

    double* row(int i) { return data_.data() + i * 4; }
    const double* row(int i) const { return data_.data() + i * 4; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
};

// Average linkage on condensed distance matrix (N*(N-1)/2 elements).
// Copies distmat internally — safe when caller still needs the data.
LinkageResult average_linkage(const double* distmat, int n);

// In-place variant — destroys distmat. Zero-copy for internal pipelines.
LinkageResult average_linkage_inplace(double* distmat, int n);

// Cut a linkage matrix at distance threshold t (scipy "distance" criterion).
// Returns N labels (1-based, matching scipy convention).
std::vector<int> fcluster_distance(const LinkageResult& Z, double t);

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
extern template float ahc_threshold<float>(const float*, int, int);
extern template double ahc_threshold<double>(const double*, int, int);
extern template std::vector<int> ahc_cluster<float>(CondensedMatrixViewF, const AhcParams&);
extern template std::vector<int> ahc_cluster<double>(CondensedMatrixViewD, const AhcParams&);

}  // namespace vbx

#endif  // VBX_CLUSTERING_H_
