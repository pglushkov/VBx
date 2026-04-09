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
//
// NOTE on ownership: all three returned ndarrays share a single nb::capsule
// owning the `VbhmmResultT`. This is suboptimal — the three logically
// independent arrays now have a coupled lifetime (the last one alive keeps
// all backing storage alive). Acceptable tradeoff for the first dirty
// version; revisit if the binding starts getting used in contexts where
// callers want to drop e.g. posteriors early while holding on to elbo.
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

// Wrapper that owns the full DiarResult plus unpacked segment arrays so all
// six returned ndarrays can share one capsule.
//
// NOTE on ownership: this is the same suboptimal pattern as vbhmm_py above —
// six logically independent ndarrays (seg starts / ends / speakers /
// posteriors / priors / elbo) all share a single nb::capsule that owns this
// struct, so their lifetimes become coupled. Dropping one on the Python
// side does not release its backing storage until the last sibling is also
// gone. Acceptable tradeoff for the first dirty version; revisit if callers
// start needing independent lifetimes (e.g. keep segments, drop posteriors).
template <typename Scalar>
struct DiarBindingResultT {
    vbx::DiarResultT<Scalar> core;
    std::vector<double>      seg_starts;
    std::vector<double>      seg_ends;
    std::vector<int>         seg_speakers;
};

// Wrap diarize(): end-to-end AHC + VB-HMM + segment merge.
// PLDA is optional — pass all three of (plda_mean, plda_transform, plda_psi)
// to take the PLDA path, or leave all three as None for the cosine path.
// Returns (seg_starts, seg_ends, seg_speakers, posteriors, priors, elbo).
template <typename Scalar>
nb::object diarize_py(
    nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>> xvecs,
    nb::ndarray<nb::numpy, const double, nb::ndim<1>> seg_starts_in,
    nb::ndarray<nb::numpy, const double, nb::ndim<1>> seg_ends_in,
    nb::object plda_mean_obj,
    nb::object plda_transform_obj,
    nb::object plda_psi_obj,
    double ahc_threshold,
    double loop_prob,
    double Fa,
    double Fb,
    int    max_iters,
    double epsilon,
    double init_smoothing,
    bool   run_vbhmm) {

    const int T    = static_cast<int>(xvecs.shape(0));
    const int Draw = static_cast<int>(xvecs.shape(1));

    if (static_cast<int>(seg_starts_in.shape(0)) != T ||
        static_cast<int>(seg_ends_in.shape(0))   != T) {
        throw std::invalid_argument(
            "seg_starts/seg_ends must have length equal to xvecs rows");
    }

    // Build Segment array from the two parallel float64 inputs.
    std::vector<vbx::Segment> segments(T);
    for (int i = 0; i < T; ++i) {
        segments[i].start_sec = seg_starts_in.data()[i];
        segments[i].end_sec   = seg_ends_in.data()[i];
    }

    // Optional PLDA: all three fields must be set together, or all None.
    std::optional<vbx::PldaModelT<Scalar>> plda_opt;
    const bool mean_none      = plda_mean_obj.is_none();
    const bool transform_none = plda_transform_obj.is_none();
    const bool psi_none       = plda_psi_obj.is_none();
    if (mean_none != transform_none || mean_none != psi_none) {
        throw std::invalid_argument(
            "plda_mean/plda_transform/plda_psi must all be None or all set");
    }
    if (!mean_none) {
        auto plda_mean_arr = nb::cast<
            nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>>>(plda_mean_obj);
        auto plda_transform_arr = nb::cast<
            nb::ndarray<nb::numpy, const Scalar, nb::ndim<2>>>(plda_transform_obj);
        auto plda_psi_arr = nb::cast<
            nb::ndarray<nb::numpy, const Scalar, nb::ndim<1>>>(plda_psi_obj);

        const int D = static_cast<int>(plda_psi_arr.shape(0));

        vbx::PldaModelT<Scalar> plda;
        plda.lda_dim = D;
        plda.mean.assign(plda_mean_arr.data(), plda_mean_arr.data() + Draw);
        plda.diag_ac.assign(plda_psi_arr.data(), plda_psi_arr.data() + D);
        plda.transform = vbx::MatrixT<Scalar>(D, Draw);
        std::copy(plda_transform_arr.data(),
                  plda_transform_arr.data() + static_cast<size_t>(D) * Draw,
                  plda.transform.storage.data());
        plda_opt.emplace(std::move(plda));
    }

    vbx::VbxParams params;
    params.ahc.threshold        = ahc_threshold;
    params.vbhmm.loop_prob      = loop_prob;
    params.vbhmm.Fa             = Fa;
    params.vbhmm.Fb             = Fb;
    params.vbhmm.max_iters      = max_iters;
    params.vbhmm.epsilon        = epsilon;
    params.vbhmm.init_smoothing = init_smoothing;
    params.run_vbhmm            = run_vbhmm;

    vbx::MatrixViewT<Scalar> xvecs_view{xvecs.data(), T, Draw, Draw};

    auto* wrap = new DiarBindingResultT<Scalar>();
    wrap->core = vbx::diarize<Scalar>(
        xvecs_view, segments.data(), T, params, plda_opt);

    // Unpack merged segments into three parallel arrays the capsule owns.
    // Resize first, then take .data() — later growth would reallocate and
    // invalidate the pointers the returned ndarrays hold.
    const size_t M = wrap->core.segments.size();
    wrap->seg_starts.resize(M);
    wrap->seg_ends.resize(M);
    wrap->seg_speakers.resize(M);
    for (size_t i = 0; i < M; ++i) {
        wrap->seg_starts[i]   = wrap->core.segments[i].start_sec;
        wrap->seg_ends[i]     = wrap->core.segments[i].end_sec;
        wrap->seg_speakers[i] = wrap->core.segments[i].speaker_id;
    }

    nb::capsule owner(wrap, [](void* p) noexcept {
        delete static_cast<DiarBindingResultT<Scalar>*>(p);
    });

    const int T_out = wrap->core.posteriors.rows;
    const int S_out = wrap->core.posteriors.cols;

    size_t m_shape[1]     = {M};
    size_t post_shape[2]  = {static_cast<size_t>(T_out),
                             static_cast<size_t>(S_out)};
    size_t prior_shape[1] = {wrap->core.speaker_priors.size()};
    size_t elbo_shape[1]  = {wrap->core.elbo_history.size()};

    auto starts_out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        wrap->seg_starts.data(), 1, m_shape, owner);
    auto ends_out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        wrap->seg_ends.data(), 1, m_shape, owner);
    auto spk_out = nb::ndarray<nb::numpy, int, nb::ndim<1>>(
        wrap->seg_speakers.data(), 1, m_shape, owner);
    auto posteriors = nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
        wrap->core.posteriors.storage.data(), 2, post_shape, owner);
    auto priors = nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
        wrap->core.speaker_priors.data(), 1, prior_shape, owner);
    auto elbo = nb::ndarray<nb::numpy, Scalar, nb::ndim<1>>(
        wrap->core.elbo_history.data(), 1, elbo_shape, owner);

    return nb::make_tuple(starts_out, ends_out, spk_out,
                          posteriors, priors, elbo);
}

// Wrap merge_adjacent_labels: (seg_starts, seg_ends, labels) ->
// (out_starts, out_ends, out_speakers). Flat-array shape matches the
// Python reference in VBx.diarization_lib.merge_adjacent_labels.
//
// Ownership note: three returned ndarrays share a single capsule. Same
// first-version tradeoff as vbhmm_py / diarize_py — acceptable, revisit
// if independent lifetimes are ever needed.
struct MergeBindingResult {
    std::vector<double> seg_starts;
    std::vector<double> seg_ends;
    std::vector<int>    seg_speakers;
};

nb::object merge_adjacent_labels_py(
    nb::ndarray<nb::numpy, const double, nb::ndim<1>> seg_starts_in,
    nb::ndarray<nb::numpy, const double, nb::ndim<1>> seg_ends_in,
    nb::ndarray<nb::numpy, const int,    nb::ndim<1>> labels_in) {

    const int N = static_cast<int>(seg_starts_in.shape(0));
    if (static_cast<int>(seg_ends_in.shape(0)) != N ||
        static_cast<int>(labels_in.shape(0))   != N) {
        throw std::invalid_argument(
            "seg_starts, seg_ends, and labels must have the same length");
    }

    std::vector<vbx::Segment> segments(N);
    for (int i = 0; i < N; ++i) {
        segments[i].start_sec = seg_starts_in.data()[i];
        segments[i].end_sec   = seg_ends_in.data()[i];
    }
    std::vector<int> labels(labels_in.data(), labels_in.data() + N);

    auto merged = vbx::merge_adjacent_labels(segments.data(), N, labels);

    auto* wrap = new MergeBindingResult();
    const size_t M = merged.size();
    wrap->seg_starts.resize(M);
    wrap->seg_ends.resize(M);
    wrap->seg_speakers.resize(M);
    for (size_t i = 0; i < M; ++i) {
        wrap->seg_starts[i]   = merged[i].start_sec;
        wrap->seg_ends[i]     = merged[i].end_sec;
        wrap->seg_speakers[i] = merged[i].speaker_id;
    }

    nb::capsule owner(wrap, [](void* p) noexcept {
        delete static_cast<MergeBindingResult*>(p);
    });

    size_t shape[1] = {M};
    auto starts_out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        wrap->seg_starts.data(), 1, shape, owner);
    auto ends_out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
        wrap->seg_ends.data(), 1, shape, owner);
    auto spk_out = nb::ndarray<nb::numpy, int, nb::ndim<1>>(
        wrap->seg_speakers.data(), 1, shape, owner);

    return nb::make_tuple(starts_out, ends_out, spk_out);
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
    m.def("diarize", &diarize_py<double>,
          nb::arg("xvecs"), nb::arg("seg_starts"), nb::arg("seg_ends"),
          nb::arg("plda_mean")      = nb::none(),
          nb::arg("plda_transform") = nb::none(),
          nb::arg("plda_psi")       = nb::none(),
          nb::arg("ahc_threshold")  = -0.015,
          nb::arg("loop_prob")      = 0.99,
          nb::arg("Fa")             = 0.3,
          nb::arg("Fb")             = 17.0,
          nb::arg("max_iters")      = 40,
          nb::arg("epsilon")        = 1e-6,
          nb::arg("init_smoothing") = 5.0,
          nb::arg("run_vbhmm")      = true);
    m.def("merge_adjacent_labels", &merge_adjacent_labels_py,
          nb::arg("seg_starts"), nb::arg("seg_ends"), nb::arg("labels"));
    m.def("diarize", &diarize_py<float>,
          nb::arg("xvecs"), nb::arg("seg_starts"), nb::arg("seg_ends"),
          nb::arg("plda_mean")      = nb::none(),
          nb::arg("plda_transform") = nb::none(),
          nb::arg("plda_psi")       = nb::none(),
          nb::arg("ahc_threshold")  = -0.015,
          nb::arg("loop_prob")      = 0.99,
          nb::arg("Fa")             = 0.3,
          nb::arg("Fb")             = 17.0,
          nb::arg("max_iters")      = 40,
          nb::arg("epsilon")        = 1e-6,
          nb::arg("init_smoothing") = 5.0,
          nb::arg("run_vbhmm")      = true);
}
