#include "vbx/linalg.h"

namespace vbx {
namespace linalg {

template <typename S> void matmul(MatrixViewT<S>, MatrixViewT<S>, MutableMatrixViewT<S>) {}
template <typename S> void matmul_atb(MatrixViewT<S>, MatrixViewT<S>, MutableMatrixViewT<S>) {}
template <typename S> void exp(const S*, S*, int) {}
template <typename S> void log(const S*, S*, int) {}
template <typename S> S logsumexp(const S*, int) { return S{0}; }
template <typename S> void logsumexp_rows(MatrixViewT<S>, S*) {}
template <typename S> void l2_normalize_rows(MutableMatrixViewT<S>) {}
template <typename S> void col_means(MatrixViewT<S>, S*) {}
template <typename S> void subtract_row(MutableMatrixViewT<S>, const S*) {}

template void matmul<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
template void matmul<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
template void matmul_atb<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
template void matmul_atb<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
template void l2_normalize_rows<float>(MutableMatrixViewF);
template void l2_normalize_rows<double>(MutableMatrixViewD);

}  // namespace linalg
}  // namespace vbx
