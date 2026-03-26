#include "vbx/vbhmm.h"

namespace vbx {

template <typename Scalar>
ForwardBackwardResultT<Scalar> forward_backward(MatrixViewT<Scalar>, MatrixViewT<Scalar>,
                                                 const Scalar*, int) { return {}; }

template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(MatrixViewT<Scalar>, MatrixViewT<Scalar>, const VbhmmParams&) { return {}; }

}  // namespace vbx
