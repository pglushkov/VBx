#include "vbx/vbx.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

namespace vbx {

std::string get_version() { return "0.1.0"; }

// Port of VBx.diarization_lib.merge_adjacent_labels: merges neighbouring or
// overlapping segments with the same label, then splits residual overlaps
// between different-label segments at the midpoint.
std::vector<DiarSegment> merge_adjacent_labels(
    const Segment* segs, int N, const std::vector<int>& labels)
{
    std::vector<DiarSegment> out;
    if (N == 0) return out;
    assert(static_cast<int>(labels.size()) == N);

    // Tolerance matches numpy.isclose defaults (rtol=1e-5, atol=1e-8).
    constexpr double kAtol = 1e-8;
    constexpr double kRtol = 1e-5;
    auto adjacent_or_overlap = [&](int i) {
        const double a = segs[i].end_sec;
        const double b = segs[i + 1].start_sec;
        const bool close = std::abs(a - b) <= kAtol + kRtol * std::abs(b);
        return close || a > b;
    };

    // Sweep segments left-to-right, cutting a run whenever the label changes
    // or a real time-gap opens up.
    out.reserve(N);
    int run_start = 0;
    for (int i = 0; i + 1 < N; ++i) {
        if (labels[i + 1] != labels[i] || !adjacent_or_overlap(i)) {
            DiarSegment ds;
            ds.start_sec  = segs[run_start].start_sec;
            ds.end_sec    = segs[i].end_sec;
            ds.speaker_id = labels[run_start];
            out.push_back(ds);
            run_start = i + 1;
        }
    }
    // Trailing run (always present when N >= 1).
    DiarSegment tail;
    tail.start_sec  = segs[run_start].start_sec;
    tail.end_sec    = segs[N - 1].end_sec;
    tail.speaker_id = labels[run_start];
    out.push_back(tail);

    // Resolve residual overlap between consecutive merged segments.
    for (size_t i = 0; i + 1 < out.size(); ++i) {
        if (out[i + 1].start_sec < out[i].end_sec) {
            const double mid = (out[i].end_sec + out[i + 1].start_sec) / 2.0;
            out[i].end_sec       = mid;
            out[i + 1].start_sec = mid;
        }
    }
    return out;
}

template <typename Scalar>
DiarResultT<Scalar> diarize(MatrixViewT<Scalar> xvecs,
                             const Segment* segments,
                             int num_segments,
                             const VbxParams& params,
                             const std::optional<PldaModelT<Scalar>>& plda)
{
    assert(xvecs.rows == num_segments &&
           "xvecs row count must match num_segments");
    assert(num_segments > 0);

    // -----------------------------------------------------------------------
    // Stage 1 — AHC clustering: initial per-frame labels + qinit seed.
    // -----------------------------------------------------------------------
    const std::vector<int> ahc_labels = ahc_cluster<Scalar>(xvecs, params.ahc);
    const int n_steps = static_cast<int>(ahc_labels.size());
    const int num_speakers =
        *std::max_element(ahc_labels.begin(), ahc_labels.end()) + 1;

    // qinit = softmax(one_hot(ahc_labels) * init_smoothing, axis=1).
    // Closed form for the one-hot case:
    //     on-label  -> exp(s) / (exp(s) + K - 1)
    //     off-label -> 1      / (exp(s) + K - 1)
    const Scalar smoothing = static_cast<Scalar>(params.vbhmm.init_smoothing);
    const Scalar exp_s     = std::exp(smoothing);
    const Scalar denom     = exp_s + static_cast<Scalar>(num_speakers - 1);
    const Scalar p_on      = exp_s / denom;
    const Scalar p_off     = Scalar{1} / denom;

    MatrixT<Scalar> qinit(n_steps, num_speakers, p_off);
    for (int i = 0; i < n_steps; ++i) {
        qinit(i, ahc_labels[i]) = p_on;
    }

    // -----------------------------------------------------------------------
    // Stage 2 — VB-HMM refinement (optional).
    //   - params.run_vbhmm == true:  call vbhmm(), take argmax of posteriors
    //   - params.run_vbhmm == false: keep raw AHC labels, use qinit as posteriors
    // -----------------------------------------------------------------------
    MatrixT<Scalar>     posteriors;
    std::vector<Scalar> speaker_priors;
    std::vector<Scalar> elbo_history;
    std::vector<int>    final_labels;

    if (params.run_vbhmm) {
        VbhmmResultT<Scalar> vb = vbhmm<Scalar>(xvecs, qinit, params.vbhmm, plda);

        // Final labels = argmax over the per-frame speaker posteriors. Ties
        // break to the lowest index, matching numpy.argsort(-q, axis=1)[:, 0].
        const int K = vb.posteriors.cols;
        final_labels.resize(n_steps);
        for (int t = 0; t < n_steps; ++t) {
            const Scalar* row = &vb.posteriors.storage[t * K];
            int    best     = 0;
            Scalar best_val = row[0];
            for (int s = 1; s < K; ++s) {
                if (row[s] > best_val) {
                    best_val = row[s];
                    best     = s;
                }
            }
            final_labels[t] = best;
        }
        posteriors     = std::move(vb.posteriors);
        speaker_priors = std::move(vb.speaker_priors);
        elbo_history   = std::move(vb.elbo_history);
    } else {
        final_labels = ahc_labels;
        posteriors   = std::move(qinit);
        // speaker_priors / elbo_history stay empty — nothing to report.
    }

    // -----------------------------------------------------------------------
    // Stage 3 — Merge adjacent same-label segments into DiarResult.
    // -----------------------------------------------------------------------
    DiarResultT<Scalar> result;
    result.segments       = merge_adjacent_labels(segments, num_segments, final_labels);
    result.posteriors     = std::move(posteriors);
    result.speaker_priors = std::move(speaker_priors);
    result.elbo_history   = std::move(elbo_history);
    return result;
}

// Explicit template instantiations.
template DiarResultF diarize<float>(MatrixViewF,
                                     const Segment*,
                                     int,
                                     const VbxParams&,
                                     const std::optional<PldaModelF>&);
template DiarResultD diarize<double>(MatrixViewD,
                                      const Segment*,
                                      int,
                                      const VbxParams&,
                                      const std::optional<PldaModelD>&);

}  // namespace vbx
