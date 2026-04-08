#ifndef VBX_VBX_H_
#define VBX_VBX_H_

#include <optional>
#include <string>
#include <vector>
#include "vbx/types.h"
#include "vbx/clustering.h"
#include "vbx/scoring.h"
#include "vbx/vbhmm.h"

namespace vbx {

std::string get_version();

struct VbxParams {
    AhcParams     ahc     = {};
    VbhmmParams   vbhmm   = {};
    ScoringMode   scoring = ScoringMode::kCosine;
    double        target_energy = 1.0;
    bool          run_vbhmm     = true;
};

template <typename Scalar>
DiarResultT<Scalar> diarize(MatrixViewT<Scalar> xvecs,
                             const Segment* segments,
                             int num_segments,
                             const VbxParams& params = {},
                             const std::optional<PldaModelT<Scalar>>& plda = std::nullopt);

template <typename Scalar>
std::vector<int> cluster(MatrixViewT<Scalar> xvecs,
                          const AhcParams& params = {});

// template <typename Scalar>
// VbhmmResultT<Scalar> refine_vbhmm(MatrixViewT<Scalar> log_likelihoods,
//                                     const std::vector<int>& initial_labels,
//                                     const VbhmmParams& params = {});

}  // namespace vbx

#endif  // VBX_VBX_H_
