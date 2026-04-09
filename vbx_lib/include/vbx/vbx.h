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

// Merge neighbouring / overlapping segments that share a label, then split
// any residual overlap between different-label segments at the midpoint.
// Port of VBx.diarization_lib.merge_adjacent_labels. Label-agnostic and
// scalar-free — both Segment and DiarSegment times are always double.
std::vector<DiarSegment> merge_adjacent_labels(
    const Segment* segments, int num_segments,
    const std::vector<int>& labels);

struct VbxParams {
    AhcParams     ahc     = {};
    VbhmmParams   vbhmm   = {};
    // PLDA vs cosine scoring is selected solely by whether a PldaModelT is
    // passed to diarize() — an empty std::optional means the no-PLDA path.
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
