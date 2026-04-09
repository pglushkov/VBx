// NOTE: unforunately failed to find a clean way to use scikit_build_core
// and have code of nanobind package being hosted in the project.
// scikit_build_core will download appropriate version of nanobind and
// compile/link against it during Python package building or when
// cmake with VBX_BUILD_PYTHON_BINDINGS=ON is being called

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "vbx/vbx.h"
#include "vbx/clustering.h"
#include "vbx/vbhmm.h"

namespace nb = nanobind;

// Wrap cosine_similarity: numpy 2D array -> numpy 1D condensed or 2D full matrix
template <typename Scalar>
nb::object
cosine_similarity_py(nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> xvecs,
                     bool condense_result) {
    const int n = static_cast<int>(xvecs.shape(0));
    const int d = static_cast<int>(xvecs.shape(1));
    vbx::MatrixViewT<Scalar> view{xvecs.data(), n, d, d};

    auto result = new std::vector<Scalar>(vbx::cosine_similarity(view, condense_result));

    nb::capsule owner(result, [](void* p) noexcept {
        delete static_cast<std::vector<Scalar>*>(p);
    });

    if (condense_result) {
        size_t shape[1] = {result->size()};
        return nb::cast(nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
            result->data(), 1, shape, owner));
    } else {
        size_t shape[2] = {static_cast<size_t>(n), static_cast<size_t>(n)};
        return nb::cast(nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
            result->data(), 2, shape, owner));
    }
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

template <typename Scalar>
nb::ndarray<nb::numpy, int, nb::ndim<1>>
ahc_cluster_py(nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> xvecs,
               double threshold) {
    const int n = static_cast<int>(xvecs.shape(0));
    const int d = static_cast<int>(xvecs.shape(1));
    vbx::MatrixViewT<Scalar> view{xvecs.data(), n, d, d};
    vbx::AhcParams params;
    params.threshold = threshold;

    auto* labels = new std::vector<int>(vbx::ahc_cluster(view, params));
    size_t shape[1] = {static_cast<size_t>(n)};

    nb::capsule owner(labels, [](void* p) noexcept {
        delete static_cast<std::vector<int>*>(p);
    });

    return nb::ndarray<nb::numpy, int, nb::ndim<1>>(
        labels->data(), 1, shape, owner);
}

// Wrap vbhmm: runs the full VBx iteration loop with a PLDA model.
// PLDA is passed as loose numpy arrays (mean, transform, psi) to avoid
// plumbing a PldaModelT Python class for now — when diarize() needs a bound
// PLDA class, this should be refactored.
// Returns (posteriors, speaker_priors, elbo_history) as a 3-tuple.
template <typename Scalar>
nb::object vbhmm_py(
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> xvecs,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> gamma_init,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>> plda_mean,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> plda_transform,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>> plda_psi,
    double loop_prob,
    double Fa,
    double Fb,
    int max_iters,
    double epsilon) {

    const int T    = static_cast<int>(xvecs.shape(0));
    const int Draw = static_cast<int>(xvecs.shape(1));
    const int S    = static_cast<int>(gamma_init.shape(1));
    const int D    = static_cast<int>(plda_psi.shape(0));

    // Build PldaModelT from the loose numpy arrays.
    vbx::PldaModelT<Scalar> plda;
    plda.lda_dim = D;
    plda.mean.assign(plda_mean.data(), plda_mean.data() + Draw);
    plda.diag_ac.assign(plda_psi.data(), plda_psi.data() + D);
    plda.transform = vbx::MatrixT<Scalar>(D, Draw);
    std::copy(plda_transform.data(),
              plda_transform.data() + static_cast<size_t>(D) * Draw,
              plda.transform.storage.data());

    vbx::VbhmmParams params;
    params.loop_prob = loop_prob;
    params.Fa        = Fa;
    params.Fb        = Fb;
    params.max_iters = max_iters;
    params.epsilon   = epsilon;

    vbx::MatrixViewT<Scalar> xvecs_view{xvecs.data(), T, Draw, Draw};
    vbx::MatrixViewT<Scalar> gamma_view{gamma_init.data(), T, S, S};

    std::optional<vbx::PldaModelT<Scalar>> plda_opt;
    plda_opt.emplace(std::move(plda));

    auto* result = new vbx::VbhmmResultT<Scalar>(
        vbx::vbhmm<Scalar>(xvecs_view, gamma_view, params, plda_opt));

    nb::capsule owner(result, [](void* p) noexcept {
        delete static_cast<vbx::VbhmmResultT<Scalar>*>(p);
    });

    size_t post_shape[2]  = {static_cast<size_t>(T), static_cast<size_t>(S)};
    size_t prior_shape[1] = {static_cast<size_t>(S)};
    size_t elbo_shape[1]  = {result->elbo_history.size()};

    auto posteriors = nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
        result->posteriors.storage.data(), 2, post_shape, owner);
    auto priors = nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
        result->speaker_priors.data(), 1, prior_shape, owner);
    auto elbo = nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
        result->elbo_history.data(), 1, elbo_shape, owner);

    return nb::make_tuple(posteriors, priors, elbo);
}

// Wrap forward_backward: (lls, tr, ip) -> (posteriors, total_log_lik, log_fw, log_bw)
// Mirrors the Python reference in VBx/VBx.py::forward_backward — returns a 4-tuple
// so the test can unpack it the same way as the Python implementation.
template <typename Scalar>
nb::object forward_backward_py(
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> lls,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> tr,
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>> ip) {

    const int T = static_cast<int>(lls.shape(0));
    const int S = static_cast<int>(lls.shape(1));

    vbx::MatrixViewT<Scalar> lls_view{lls.data(), T, S, S};
    vbx::MatrixViewT<Scalar> tr_view {tr.data(),  S, S, S};

    auto* result = new vbx::ForwardBackwardResultT<Scalar>(
        vbx::forward_backward<Scalar>(lls_view, tr_view, ip.data()));

    nb::capsule owner(result, [](void* p) noexcept {
        delete static_cast<vbx::ForwardBackwardResultT<Scalar>*>(p);
    });

    size_t shape[2] = {static_cast<size_t>(T), static_cast<size_t>(S)};
    auto posteriors = nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
        result->posteriors.storage.data(), 2, shape, owner);
    auto log_fw = nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
        result->log_fw.storage.data(), 2, shape, owner);
    auto log_bw = nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
        result->log_bw.storage.data(), 2, shape, owner);

    return nb::make_tuple(posteriors, result->total_log_lik, log_fw, log_bw);
}

NB_MODULE(vbx_native, m) {
    m.def("get_version", &vbx::get_version);
    m.def("cosine_similarity", &cosine_similarity_py<double>,
          nb::arg("xvecs"), nb::arg("condense_result") = true);
    m.def("cosine_similarity", &cosine_similarity_py<float>,
          nb::arg("xvecs"), nb::arg("condense_result") = true);
    m.def("average_linkage", &average_linkage_py,
          nb::arg("distmat"), nb::arg("n"));
    m.def("fcluster_distance", &fcluster_distance_py,
          nb::arg("Z"), nb::arg("t"));
    m.def("ahc_threshold", &ahc_threshold_py<double>,
          nb::arg("scores"), nb::arg("niters") = 20);
    m.def("ahc_threshold", &ahc_threshold_py<float>,
          nb::arg("scores"), nb::arg("niters") = 20);
    m.def("ahc_cluster", &ahc_cluster_py<double>,
          nb::arg("xvecs"), nb::arg("threshold") = -0.015);
    m.def("ahc_cluster", &ahc_cluster_py<float>,
          nb::arg("xvecs"), nb::arg("threshold") = -0.015);
    m.def("forward_backward", &forward_backward_py<double>,
          nb::arg("log_likelihoods"), nb::arg("transition"), nb::arg("initial"));
    m.def("forward_backward", &forward_backward_py<float>,
          nb::arg("log_likelihoods"), nb::arg("transition"), nb::arg("initial"));
    m.def("vbhmm", &vbhmm_py<double>,
          nb::arg("xvecs"), nb::arg("gamma_init"),
          nb::arg("plda_mean"), nb::arg("plda_transform"), nb::arg("plda_psi"),
          nb::arg("loop_prob") = 0.99,
          nb::arg("Fa")        = 0.3,
          nb::arg("Fb")        = 17.0,
          nb::arg("max_iters") = 40,
          nb::arg("epsilon")   = 1e-6);
    m.def("vbhmm", &vbhmm_py<float>,
          nb::arg("xvecs"), nb::arg("gamma_init"),
          nb::arg("plda_mean"), nb::arg("plda_transform"), nb::arg("plda_psi"),
          nb::arg("loop_prob") = 0.99,
          nb::arg("Fa")        = 0.3,
          nb::arg("Fb")        = 17.0,
          nb::arg("max_iters") = 40,
          nb::arg("epsilon")   = 1e-6);
}
