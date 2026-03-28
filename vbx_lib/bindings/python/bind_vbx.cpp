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

// Wrap average_linkage: condensed distances -> scipy-convention (n-1)x4 matrix
nb::ndarray<nb::numpy, double, nb::ndim<2>>
average_linkage_py(
        nb::ndarray<nb::numpy, const double, nb::ndim<1>> distmat,
        int n) {
    // C++ does R-to-scipy conversion internally
    auto* lr = new vbx::LinkageResult(vbx::average_linkage(distmat.data(), n));

    size_t shape[2] = {static_cast<size_t>(lr->steps), 4};
    nb::capsule owner(lr, [](void* p) noexcept {
        delete static_cast<vbx::LinkageResult*>(p);
    });

    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
        lr->data.data(), 2, shape, owner);
}

NB_MODULE(vbx_native, m) {
    m.def("get_version", &vbx::get_version);
    m.def("cosine_similarity", &cosine_similarity_py<double>, nb::arg("xvecs"));
    m.def("cosine_similarity", &cosine_similarity_py<float>, nb::arg("xvecs"));
    m.def("average_linkage", &average_linkage_py,
          nb::arg("distmat"), nb::arg("n"));
}
