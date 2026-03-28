#include <algorithm>
#include <cmath>
#include <numeric>
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

// Convert R hclust merge/height to scipy (n-1)x4 linkage matrix.
// R convention: negative = leaf (1-based), positive = cluster step (1-based).
// Scipy convention: leaves 0..n-1, clusters n, n+1, ...; row = [idx1, idx2, height, size].
static LinkageResult r_to_scipy(const int* merge, const double* height,
                                int n) {
    const int steps = n - 1;
    LinkageResult lr;
    lr.steps = steps;
    lr.data.resize(steps * 4);

    // Track cluster sizes: leaves at [0..n-1], merged clusters at [n..]
    std::vector<double> sizes(n, 1.0);

    for (int i = 0; i < steps; ++i) {
        // R merge layout is column-major: left children at [0..steps-1],
        // right children at [steps..2*steps-1].
        int r1 = merge[i];
        int r2 = merge[i + steps];

        double s1 = (r1 < 0) ? double(-r1 - 1) : double(n + r1 - 1);
        double s2 = (r2 < 0) ? double(-r2 - 1) : double(n + r2 - 1);

        double sz1 = (r1 < 0) ? 1.0 : sizes[static_cast<int>(n + r1 - 1)];
        double sz2 = (r2 < 0) ? 1.0 : sizes[static_cast<int>(n + r2 - 1)];
        double sz = sz1 + sz2;
        sizes.push_back(sz);

        // scipy convention: idx1 < idx2
        if (s1 > s2) std::swap(s1, s2);

        lr.data[i * 4 + 0] = s1;
        lr.data[i * 4 + 1] = s2;
        lr.data[i * 4 + 2] = height[i];
        lr.data[i * 4 + 3] = sz;
    }
    return lr;
}

LinkageResult average_linkage_inplace(double* distmat, int n) {
    const int steps = n - 1;
    std::vector<int> merge(steps * 2);
    std::vector<double> height(steps);

    hclust_fast(n, distmat, HCLUST_METHOD_AVERAGE,
                merge.data(), height.data());

    return r_to_scipy(merge.data(), height.data(), n);
}

LinkageResult average_linkage(const double* distmat, int n) {
    std::vector<double> dist_copy(distmat, distmat + condensed_size(n));
    return average_linkage_inplace(dist_copy.data(), n);
}

std::vector<int> fcluster_distance(const LinkageResult& Z, double t) {
    const int n = Z.steps + 1;
    // Union-find: parent[i] = parent of node i (leaves 0..n-1, clusters n..2n-2)
    std::vector<int> parent(2 * n - 1);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    // Union all merge steps with height <= t
    for (int i = 0; i < Z.steps; ++i) {
        const double* row = Z.row(i);
        if (row[2] > t) break;  // monotonic heights — all remaining are > t
        int a = find(static_cast<int>(row[0]));
        int b = find(static_cast<int>(row[1]));
        int cluster_id = n + i;  // scipy convention
        parent[a] = cluster_id;
        parent[b] = cluster_id;
    }

    // Assign labels (1-based) to each leaf
    std::vector<int> labels(n);
    int next_label = 1;
    std::vector<int> root_to_label(2 * n - 1, 0);
    for (int i = 0; i < n; ++i) {
        int root = find(i);
        if (root_to_label[root] == 0) {
            root_to_label[root] = next_label++;
        }
        labels[i] = root_to_label[root];
    }
    return labels;
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
