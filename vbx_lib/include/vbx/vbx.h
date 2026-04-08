#ifndef VBX_VBX_H_
#define VBX_VBX_H_

#include <optional>
#include <string>
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

extern template DiarResultF diarize<float>(MatrixViewF,
                                            const Segment*,
                                            int,
                                            const VbxParams&,
                                            const std::optional<PldaModelF>&);
extern template DiarResultD diarize<double>(MatrixViewD,
                                             const Segment*,
                                             int,
                                             const VbxParams&,
                                             const std::optional<PldaModelD>&);

}  // namespace vbx

#endif  // VBX_VBX_H_
