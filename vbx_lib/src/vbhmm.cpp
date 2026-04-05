#include "vbx/vbhmm.h"

namespace vbx {

template <typename Scalar>
ForwardBackwardResultT<Scalar> forward_backward(MatrixViewT<Scalar>, MatrixViewT<Scalar>,
                                                 const Scalar*, int) { return {}; }

template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(MatrixViewT<Scalar>, MatrixViewT<Scalar>, const VbhmmParams&) { return {}; }

template ForwardBackwardResultF forward_backward<float>(MatrixViewF, MatrixViewF, const float*, int);
template ForwardBackwardResultD forward_backward<double>(MatrixViewD, MatrixViewD, const double*, int);
template VbhmmResultF vbhmm<float>(MatrixViewF, MatrixViewF, const VbhmmParams&);
template VbhmmResultD vbhmm<double>(MatrixViewD, MatrixViewD, const VbhmmParams&);

}  // namespace vbx
