#include "vbx/vbx.h"

namespace vbx {

std::string get_version() { return "0.1.0"; }

template <typename Scalar>
DiarResultT<Scalar> diarize(MatrixViewT<Scalar>, const Segment*, int,
                             const std::optional<PldaModelT<Scalar>>&,
                             const VbxParams&) { return {}; }

template <typename Scalar>
std::vector<int> cluster(MatrixViewT<Scalar>, const AhcParams&) { return {}; }

template <typename Scalar>
VbhmmResultT<Scalar> refine_vbhmm(MatrixViewT<Scalar>, const std::vector<int>&,
                                    const VbhmmParams&) { return {}; }

}  // namespace vbx
