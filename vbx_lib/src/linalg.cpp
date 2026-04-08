#include "vbx/linalg.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vbx {
namespace linalg {

template <typename S> void matmul(MatrixViewT<S>, MatrixViewT<S>, MutableMatrixViewT<S>) {}
template <typename S> void matmul_atb(MatrixViewT<S>, MatrixViewT<S>, MutableMatrixViewT<S>) {}
template <typename S> void exp(const S*, S*, int) {}
template <typename S> void log(const S*, S*, int) {}

// Numerically stable log-sum-exp over a contiguous vector of length n.
// Handles the all-(-inf) case by returning -inf directly, matching scipy.
template <typename S>
S logsumexp(const S* a, int n) {
    if (n <= 0) return -std::numeric_limits<S>::infinity();
    const S m = *std::max_element(a, a + n);
    if (m == -std::numeric_limits<S>::infinity()) return m;
    S sum = S{0};
    for (int i = 0; i < n; ++i) {
        sum += std::exp(a[i] - m);
    }
    return m + std::log(sum);
}

// Row-wise logsumexp: out[i] = logsumexp(A[i, :]).
// A.stride may exceed A.cols (non-contiguous rows supported).
template <typename S>
void logsumexp_rows(MatrixViewT<S> A, S* out) {
    for (int i = 0; i < A.rows; ++i) {
        const S* row = A.data + i * A.stride;
        out[i] = logsumexp<S>(row, A.cols);
    }
}

template <typename S> void l2_normalize_rows(MutableMatrixViewT<S>) {}
template <typename S> void col_means(MatrixViewT<S>, S*) {}
template <typename S> void subtract_row(MutableMatrixViewT<S>, const S*) {}

template void matmul<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
template void matmul<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
template void matmul_atb<float>(MatrixViewF, MatrixViewF, MutableMatrixViewF);
template void matmul_atb<double>(MatrixViewD, MatrixViewD, MutableMatrixViewD);
template void l2_normalize_rows<float>(MutableMatrixViewF);
template void l2_normalize_rows<double>(MutableMatrixViewD);
template float  logsumexp<float>(const float*, int);
template double logsumexp<double>(const double*, int);
template void logsumexp_rows<float>(MatrixViewF, float*);
template void logsumexp_rows<double>(MatrixViewD, double*);

}  // namespace linalg
}  // namespace vbx
