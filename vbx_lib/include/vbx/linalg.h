#ifndef VBX_LINALG_H_
#define VBX_LINALG_H_

#include "vbx/types.h"

namespace vbx {
namespace linalg {

template <typename S> void matmul(MatrixViewT<S> A, MatrixViewT<S> B, MutableMatrixViewT<S> C);
template <typename S> void matmul_atb(MatrixViewT<S> A, MatrixViewT<S> B, MutableMatrixViewT<S> C);
template <typename S> void exp(const S* a, S* out, int n);
template <typename S> void log(const S* a, S* out, int n);
template <typename S> S logsumexp(const S* a, int n);
template <typename S> void logsumexp_rows(MatrixViewT<S> A, S* out);
template <typename S> void l2_normalize_rows(MutableMatrixViewT<S> A);
template <typename S> void col_means(MatrixViewT<S> A, S* out);
template <typename S> void subtract_row(MutableMatrixViewT<S> A, const S* v);

extern template void matmul<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
extern template void matmul<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
extern template void matmul_atb<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
extern template void matmul_atb<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
extern template void l2_normalize_rows<float>(MutableMatrixViewF);
extern template void l2_normalize_rows<double>(MutableMatrixViewD);

}  // namespace linalg
}  // namespace vbx

#endif  // VBX_LINALG_H_
