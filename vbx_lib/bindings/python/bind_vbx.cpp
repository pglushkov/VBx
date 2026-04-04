#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "vbx/vbx.h"
#include "vbx/clustering.h"

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

// Wrap average_linkage: condensed distances -> scipy-convention (n-1)x4 matrix
nb::ndarray<nb::numpy, double, nb::ndim<2>>
average_linkage_py(
        nb::ndarray<nb::numpy, const double, nb::ndim<1>> distmat,
        int n) {
    // C++ does R-to-scipy conversion internally
    auto* lr = new vbx::LinkageResult(vbx::average_linkage(distmat.data(), n));

    size_t shape[2] = {static_cast<size_t>(lr->n_steps()), 4};
    nb::capsule owner(lr, [](void* p) noexcept {
        delete static_cast<vbx::LinkageResult*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        lr->data(), 2, shape, owner);
}

// Wrap fcluster_distance: linkage (n-1)x4 matrix + threshold -> 1D label array
nb::ndarray<nb::numpy, int, nb::ndim<1>>
fcluster_distance_py(
        nb::ndarray<nb::numpy, const double, nb::ndim<2>> Z_arr,
        double t) {
    const int steps = static_cast<int>(Z_arr.shape(0));
    const int n = steps + 1;

    // Build LinkageResult from numpy array
    vbx::LinkageResult lr(steps);
    std::copy(Z_arr.data(), Z_arr.data() + steps * 4, lr.data());

    auto* labels = new std::vector<int>(vbx::fcluster_distance(lr, t));
    size_t shape[1] = {static_cast<size_t>(n)};

    nb::capsule owner(labels, [](void* p) noexcept {
        delete static_cast<std::vector<int>*>(p);
    });

    return nb::ndarray<nb::numpy, int, nb::ndim<1>>(
        labels->data(), 1, shape, owner);
}

template <typename Scalar>
Scalar ahc_threshold_py(nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>> scores,
                        int niters) {
    return vbx::ahc_threshold<Scalar>(scores.data(),
                                       static_cast<int>(scores.shape(0)),
                                       niters);
}

NB_MODULE(vbx_native, m) {
    m.def("get_version", &vbx::get_version);
    m.def("cosine_similarity", &cosine_similarity_py<double>, nb::arg("xvecs"));
    m.def("cosine_similarity", &cosine_similarity_py<float>, nb::arg("xvecs"));
    m.def("average_linkage", &average_linkage_py,
          nb::arg("distmat"), nb::arg("n"));
    m.def("fcluster_distance", &fcluster_distance_py,
          nb::arg("Z"), nb::arg("t"));
    m.def("ahc_threshold", &ahc_threshold_py<double>,
          nb::arg("scores"), nb::arg("niters") = 20);
    m.def("ahc_threshold", &ahc_threshold_py<float>,
          nb::arg("scores"), nb::arg("niters") = 20);
}
