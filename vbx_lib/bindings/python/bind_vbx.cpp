#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vbx/vbx.h"

namespace nb = nanobind;

// Wrap cosine_similarity: numpy 2D array -> numpy 1D condensed array
template <typename Scalar>
nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>
cosine_similarity_py(nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> xvecs) {
    const int n = static_cast<int>(xvecs.shape(0));
    const int d = static_cast<int>(xvecs.shape(1));
    vbx::MatrixViewT<Scalar> view{xvecs.data(), n, d, d};

    auto result = new std::vector<Scalar>(vbx::cosine_similarity(view));
    size_t shape[1] = {result->size()};

    nb::capsule owner(result, [](void* p) noexcept {
        delete static_cast<std::vector<Scalar>*>(p);
    });

    return nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
        result->data(), 1, shape, owner);
}

// Wrap average_linkage: condensed distances -> scipy-convention (n-1)x4 linkage matrix
// Converts from R hclust convention (1-based, negative leaves) to scipy convention
// (0-based, leaves 0..n-1, clusters n, n+1, ...; columns: idx1, idx2, height, size).
nb::ndarray<nb::numpy, double, nb::ndim<2>>
average_linkage_py(
        nb::ndarray<nb::numpy, const double, nb::ndim<1>> distmat,
        int n) {
    auto lr = vbx::average_linkage(distmat.data(), n);
    const int steps = n - 1;

    // Build scipy-style (n-1) x 4 linkage matrix
    auto* Z = new std::vector<double>(steps * 4);

    // Track cluster sizes: leaves have size 1, merged clusters accumulate
    std::vector<double> sizes(n, 1.0);  // leaves
    // sizes for new clusters (indexed steps - 1)

    for (int i = 0; i < steps; ++i) {
        // R convention: negative = leaf (1-based), positive = cluster (1-based)
        int r1 = lr.merge[i * 2];
        int r2 = lr.merge[i * 2 + 1];

        // Convert to scipy: leaves -> 0-based index, clusters -> n + step_index
        double s1 = (r1 < 0) ? (-r1 - 1) : (n + (r1 - 1));
        double s2 = (r2 < 0) ? (-r2 - 1) : (n + (r2 - 1));

        // Cluster sizes: leaves at [0..n-1], merged clusters at [n..n+i-1]
        double sz1 = (r1 < 0) ? 1.0 : sizes[n + r1 - 1];
        double sz2 = (r2 < 0) ? 1.0 : sizes[n + r2 - 1];
        double sz = sz1 + sz2;
        sizes.push_back(sz);  // size for cluster (n + i)

        (*Z)[i * 4 + 0] = s1;
        (*Z)[i * 4 + 1] = s2;
        (*Z)[i * 4 + 2] = lr.height[i];
        (*Z)[i * 4 + 3] = sz;
    }

    size_t shape[2] = {static_cast<size_t>(steps), 4};
    nb::capsule owner(Z, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        Z->data(), 2, shape, owner);
}

NB_MODULE(vbx_native, m) {
    m.def("get_version", &vbx::get_version);
    m.def("cosine_similarity", &cosine_similarity_py<double>, nb::arg("xvecs"));
    m.def("cosine_similarity", &cosine_similarity_py<float>, nb::arg("xvecs"));
    m.def("average_linkage", &average_linkage_py,
          nb::arg("distmat"), nb::arg("n"));
}
