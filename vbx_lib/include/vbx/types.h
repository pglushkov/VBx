#ifndef VBX_TYPES_H_
#define VBX_TYPES_H_

#include <vector>

namespace vbx {

// ---------------------------------------------------------------------------
// MatrixViewT / MutableMatrixViewT — non-owning, row-major
// ---------------------------------------------------------------------------

template <typename Scalar>
struct MatrixViewT {
    const Scalar* data = nullptr;
    int rows = 0;
    int cols = 0;
    int stride = 0;  // elements between consecutive rows (>= cols)

    Scalar operator()(int r, int c) const { return data[r * stride + c]; }
};

template <typename Scalar>
struct MutableMatrixViewT {
    Scalar* data = nullptr;
    int rows = 0;
    int cols = 0;
    int stride = 0;

    Scalar& operator()(int r, int c) { return data[r * stride + c]; }
    Scalar  operator()(int r, int c) const { return data[r * stride + c]; }

    operator MatrixViewT<Scalar>() const {
        return {data, rows, cols, stride};
    }
};

using MatrixViewF        = MatrixViewT<float>;
using MatrixViewD        = MatrixViewT<double>;
using MutableMatrixViewF = MutableMatrixViewT<float>;
using MutableMatrixViewD = MutableMatrixViewT<double>;
using MatrixView         = MatrixViewD;
using MutableMatrixView  = MutableMatrixViewD;

// ---------------------------------------------------------------------------
// MatrixT — owning, row-major
// ---------------------------------------------------------------------------

template <typename Scalar>
struct MatrixT {
    std::vector<Scalar> storage;
    int rows = 0;
    int cols = 0;

    MatrixT() = default;

    MatrixT(int rows, int cols)
        : storage(rows * cols, Scalar{0}), rows(rows), cols(cols) {}

    MatrixT(int rows, int cols, Scalar fill)
        : storage(rows * cols, fill), rows(rows), cols(cols) {}

    Scalar& operator()(int r, int c) { return storage[r * cols + c]; }
    Scalar  operator()(int r, int c) const { return storage[r * cols + c]; }

    operator MatrixViewT<Scalar>() const {
        return {storage.data(), rows, cols, cols};
    }
    operator MutableMatrixViewT<Scalar>() {
        return {storage.data(), rows, cols, cols};
    }

    Scalar*       data()       { return storage.data(); }
    const Scalar* data() const { return storage.data(); }
};

using MatrixF = MatrixT<float>;
using MatrixD = MatrixT<double>;
using Matrix  = MatrixD;

// ---------------------------------------------------------------------------
// CondensedMatrixViewT — non-owning view over upper triangle (no diagonal)
// ---------------------------------------------------------------------------

template <typename Scalar>
struct CondensedMatrixViewT {
    const Scalar* data = nullptr;
    int n = 0;  // number of items (logical n x n symmetric matrix)

    Scalar operator()(int i, int j) const {
        if (i > j) std::swap(i, j);
        return data[i * n - i * (i + 1) / 2 + j - i - 1];
    }
};

inline int condensed_size(int n) { return n * (n - 1) / 2; }

using CondensedMatrixViewF = CondensedMatrixViewT<float>;
using CondensedMatrixViewD = CondensedMatrixViewT<double>;
using CondensedMatrixView  = CondensedMatrixViewD;

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

struct Segment {
    double start = 0.0;  // seconds
    double end   = 0.0;  // seconds
};

// ---------------------------------------------------------------------------
// DiarResult
// ---------------------------------------------------------------------------

struct DiarSegment {
    double start      = 0.0;
    double end        = 0.0;
    int    speaker_id = 0;
};

template <typename Scalar>
struct DiarResultT {
    std::vector<DiarSegment> segments;
    MatrixT<Scalar>          posteriors;
    std::vector<Scalar>      speaker_priors;
    std::vector<Scalar>      elbo_history;
};

using DiarResultF = DiarResultT<float>;
using DiarResultD = DiarResultT<double>;
using DiarResult  = DiarResultD;

}  // namespace vbx

#endif  // VBX_TYPES_H_
