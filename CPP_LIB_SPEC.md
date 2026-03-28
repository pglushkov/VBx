# Overview
Main goal of this doc is to make a technical spec of C++ library that will replace VBx Python code while maintaining same (ideally bit-identical) functional performance.

# C++ library high level requirements
1. Using C++17 standard
2. Using CMake as build facility
3. Using Google coding standards with reasonable exceptions
4. C++ library with clean C++ API and Python bindings
5. Original library contains X-vectors extraction model, but we will assume X-vectors will be our input, along with segment markup. Basically we're interested only in run_vbhmm() part of the full algorithm.
6. **WARNING: No thread safety guarantees are provided. All APIs assume single-threaded use.**

---

# Precision policy

The library supports both `float` (fp32) and `double` (fp64) compute precision.

- **Internal math code** is templated on `typename Scalar` — single implementation,
  explicitly instantiated for `float` and `double` in `.cpp` files.
- **Public API** uses concrete type aliases (`F` / `D` suffixed) so headers stay
  template-free at the call site and nanobind/CLI consumers see plain types.
  Unqualified aliases (e.g. `Matrix`, `MatrixView`) default to `double`.
- **PLDA model** is stored on disk as **fp32** (Kaldi binary format). The loader
  converts to the caller's chosen precision at load time — the model struct is
  `PldaModelT<Scalar>` like everything else. No mixed-precision math anywhere
  in the pipeline.
- **Config structs** (`VbhmmParams`, `AhcParams`, `VbxParams`) use `double` —
  small human-readable values, cast to `Scalar` once at entry.
- **Python bindings** expose both precisions; numpy dtype selects the path
  (`float32` -> `F` variants, `float64` -> `D` variants).

```cpp
// In every public header, after the template definition:
extern template struct MatrixViewT<float>;
extern template struct MatrixViewT<double>;
// etc.
```

---

# Project structure

```
vbx_lib/
├── CMakeLists.txt
├── include/
│   └── vbx/
│       ├── types.h            // Common types and data structures
│       ├── linalg.h           // Linear algebra abstraction layer
│       ├── clustering.h       // AHC clustering API
│       ├── scoring.h          // PLDA and cosine scoring
│       ├── vbhmm.h            // VB-HMM API
│       └── vbx.h              // End-to-end API + convenience header
├── src/
│   ├── linalg.cpp             // explicit instantiations for float/double
│   ├── clustering.cpp
│   ├── scoring.cpp
│   ├── vbhmm.cpp
│   └── vbx.cpp
├── bindings/
│   └── python/
│       ├── CMakeLists.txt
│       └── bind_vbx.cpp       // nanobind Python module
├── tools/
│   └── vbx_cli.cpp            // Console utility
└── third_party/
    └── fastcluster/           // hclust-cpp vendored
```

---

# Module 1 — Common types (`vbx/types.h`)

## 1.1 Matrix view — `MatrixViewT<Scalar>`

Zero-copy, non-owning view over a contiguous row-major buffer.
Used at all API boundaries so callers can pass raw pointers (C), `std::vector`,
or numpy arrays (via nanobind) without copies.

```cpp
namespace vbx {

template <typename Scalar>
struct MatrixViewT {
    const Scalar* data;   // pointer to first element
    int rows;             // number of rows (e.g. number of x-vectors)
    int cols;             // number of columns (e.g. x-vector dimension)
    int stride;           // elements between consecutive rows (>= cols, allows sub-views)

    Scalar operator()(int r, int c) const;
};

template <typename Scalar>
struct MutableMatrixViewT {
    Scalar* data;
    int rows;
    int cols;
    int stride;

    Scalar& operator()(int r, int c);
    Scalar  operator()(int r, int c) const;

    operator MatrixViewT<Scalar>() const;
};

// Concrete aliases
using MatrixViewF        = MatrixViewT<float>;
using MatrixViewD        = MatrixViewT<double>;
using MutableMatrixViewF = MutableMatrixViewT<float>;
using MutableMatrixViewD = MutableMatrixViewT<double>;

// Unqualified = double (matches Python reference float64)
using MatrixView        = MatrixViewD;
using MutableMatrixView = MutableMatrixViewD;

}  // namespace vbx
```

**Design notes:**
- Row-major by default. All internal code and documentation assumes row-major.
- `stride` enables sub-matrix views without copying.
- Unqualified `MatrixView` = `double` for backward compat and conciseness.

## 1.2 Owning matrix — `MatrixT<Scalar>`

Internal owning type backed by `std::vector<Scalar>`. Returned by library functions.

```cpp
namespace vbx {

template <typename Scalar>
struct MatrixT {
    std::vector<Scalar> storage;
    int rows;
    int cols;

    MatrixT();
    MatrixT(int rows, int cols);                   // zero-initialized
    MatrixT(int rows, int cols, Scalar fill_val);

    Scalar& operator()(int r, int c);
    Scalar  operator()(int r, int c) const;

    operator MatrixViewT<Scalar>() const;
    operator MutableMatrixViewT<Scalar>();

    Scalar*       data();
    const Scalar* data() const;
};

using MatrixF = MatrixT<float>;
using MatrixD = MatrixT<double>;
using Matrix  = MatrixD;   // default

}  // namespace vbx
```

## 1.3 Segment — `Segment`

Numerical representation of a timed segment (matches `out_seg_fn` format from
`extract_xvectors()`). Always `double` — time values don't benefit from fp32.

```cpp
namespace vbx {

struct Segment {
    double start;    // seconds
    double end;      // seconds
};

}  // namespace vbx
```

## 1.4 Diarization result — `DiarResultT<Scalar>`

```cpp
namespace vbx {

struct DiarSegment {
    double start;       // seconds (always double)
    double end;         // seconds
    int    speaker_id;  // 0-based
};

template <typename Scalar>
struct DiarResultT {
    std::vector<DiarSegment> segments;        // merged speaker-homogeneous segments
    MatrixT<Scalar>          posteriors;      // T x S frame-level posteriors (may be empty)
    std::vector<Scalar>      speaker_priors;  // S-dim converged speaker priors
    std::vector<Scalar>      elbo_history;    // ELBO per iteration (empty if AHC-only)
};

using DiarResultF = DiarResultT<float>;
using DiarResultD = DiarResultT<double>;
using DiarResult  = DiarResultD;

}  // namespace vbx
```

---

# Module 2 — Linear algebra abstraction (`vbx/linalg.h`)

Thin wrapper so the linalg backend can be swapped later (e.g. Eigen, MKL, OpenBLAS)
without touching algorithm code. All functions templated on `Scalar`.

```cpp
namespace vbx {
namespace linalg {

template <typename S>
void matmul(MatrixViewT<S> A, MatrixViewT<S> B, MutableMatrixViewT<S> C);

template <typename S>
void matmul_atb(MatrixViewT<S> A, MatrixViewT<S> B, MutableMatrixViewT<S> C);

template <typename S>
void exp(const S* a, S* out, int n);

template <typename S>
void log(const S* a, S* out, int n);

template <typename S>
S logsumexp(const S* a, int n);

template <typename S>
void logsumexp_rows(MatrixViewT<S> A, S* out);

template <typename S>
void l2_normalize_rows(MutableMatrixViewT<S> A);

template <typename S>
void col_means(MatrixViewT<S> A, S* out);

template <typename S>
void subtract_row(MutableMatrixViewT<S> A, const S* v);

}  // namespace linalg
}  // namespace vbx
```

**Default implementation:** plain C++ loops over raw pointers. No external deps.
Later can `#ifdef` or link-time swap to Eigen/BLAS.

---

# Module 3 — Clustering (`vbx/clustering.h`)

## 3.1 Cosine similarity matrix

```cpp
namespace vbx {

// Compute N x N cosine similarity matrix from N x-vectors of dimension D.
// xvecs: N x D matrix (rows = x-vectors).
// Returns: N x N symmetric matrix, values in [-1, 1].
template <typename Scalar>
MatrixT<Scalar> cosine_similarity(MatrixViewT<Scalar> xvecs);

}  // namespace vbx
```

Reference: `diarization_lib.py:cos_similarity()` — L2-normalizes rows, then
computes `X @ X^T` in chunks for memory efficiency.

## 3.2 Two-Gaussian calibration

```cpp
namespace vbx {

template <typename Scalar>
struct CalibrationResultT {
    Scalar threshold;                // score where the two Gaussians intersect
    std::vector<Scalar> calibrated;  // log-odds-ratio calibrated scores (same length as input)
};

using CalibrationResultF = CalibrationResultT<float>;
using CalibrationResultD = CalibrationResultT<double>;

// Fit 2-component GMM with shared variance on flattened scores,
// return the intersection threshold and calibrated scores.
// niters: EM iterations (default 20).
template <typename Scalar>
CalibrationResultT<Scalar> two_gmm_calibrate(const Scalar* scores, int n, int niters = 20);

}  // namespace vbx
```

Reference: `diarization_lib.py:twoGMMcalib_lin()`

## 3.3 AHC clustering

```cpp
namespace vbx {

struct AhcParams {
    double threshold  = -0.015;  // bias added to calibrated threshold for fcluster cutoff
    // Linkage is always "average" (UPGMA), matching reference implementation.
};

// Perform Agglomerative Hierarchical Clustering on an N x N cosine similarity matrix.
//
// Steps (matching reference pipeline):
//   1. Run two_gmm_calibrate on flattened similarity matrix to get threshold
//   2. Convert similarity to condensed distance (negate)
//   3. fastcluster::linkage with average method
//   4. Cut dendrogram at -(calibrated_thr + params.threshold) + adjust
//
// sim_matrix: N x N cosine similarity matrix (from cosine_similarity()).
// Returns: vector of N cluster labels (0-based).
template <typename Scalar>
std::vector<int> ahc_cluster(MatrixViewT<Scalar> sim_matrix, const AhcParams& params = {});

}  // namespace vbx
```

Reference: `vbhmm.py` lines 120-133.

---

# Module 4 — Scoring (`vbx/scoring.h`)

Two scoring backends: **cosine** (for models trained so PLDA is redundant) and
**PLDA** (classic x-vector pipeline). Both produce the same output shape so the
rest of the pipeline is scoring-agnostic.

## 4.1 Scoring mode

```cpp
namespace vbx {

enum class ScoringMode {
    kCosine,  // cosine similarity as score (no model needed)
    kPlda     // PLDA log-likelihood-ratio scoring in LDA space
};

}  // namespace vbx
```

## 4.2 PLDA model

Templated on `Scalar` like everything else. On-disk format is Kaldi binary fp32;
the loader (see 4.3) converts to target precision at load time.

```cpp
namespace vbx {

template <typename Scalar>
struct PldaModelT {
    MatrixT<Scalar>     transform;  // LDA transformation matrix (D_in x lda_dim)
    std::vector<Scalar> mean;       // global mean (D_in)
    std::vector<Scalar> diag_ac;    // diagonal across-class covariance in LDA space (lda_dim)
    int                 lda_dim;
};

using PldaModelF = PldaModelT<float>;
using PldaModelD = PldaModelT<double>;
using PldaModel  = PldaModelD;

}  // namespace vbx
```

## 4.3 PLDA loader

```cpp
namespace vbx {

// Load PLDA model from Kaldi-format file, converting from on-disk fp32
// to the requested Scalar precision.
template <typename Scalar>
PldaModelT<Scalar> load_plda(const char* path);

}  // namespace vbx
```

## 4.4 Pairwise scoring

```cpp
namespace vbx {

// Cosine pairwise scores: N x N matrix of cosine similarities.
template <typename Scalar>
MatrixT<Scalar> score_cosine(MatrixViewT<Scalar> xvecs);

// PLDA pairwise scores: N x N matrix of log-likelihood-ratios.
// xvecs: N x D raw x-vectors (before transform).
template <typename Scalar>
MatrixT<Scalar> score_plda(MatrixViewT<Scalar> xvecs, const PldaModelT<Scalar>& plda);

}  // namespace vbx
```

Reference: `diarization_lib.py:PLDA_scoring_in_LDA_space()`

## 4.5 VB-HMM log-likelihoods

These produce the T x S matrix that feeds into the VB-HMM iteration.

```cpp
namespace vbx {

// Cosine-based log-likelihoods for VB-HMM.
template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_cosine(MatrixViewT<Scalar> xvecs, Scalar Fa);

// PLDA-based log-likelihoods for VB-HMM (equation 23 from VBx paper).
template <typename Scalar>
MatrixT<Scalar> compute_log_likelihoods_plda(
    MatrixViewT<Scalar> xvecs,
    const PldaModelT<Scalar>& plda,
    Scalar Fa);

}  // namespace vbx
```

---

# Module 5 — VB-HMM (`vbx/vbhmm.h`)

## 5.1 Forward-backward

```cpp
namespace vbx {

template <typename Scalar>
struct ForwardBackwardResultT {
    MatrixT<Scalar> posteriors;     // T x S per-frame state posteriors
    Scalar          total_log_lik;  // total (forward) log-likelihood
    MatrixT<Scalar> log_fw;         // T x S log forward probabilities
    MatrixT<Scalar> log_bw;         // T x S log backward probabilities
};

using ForwardBackwardResultF = ForwardBackwardResultT<float>;
using ForwardBackwardResultD = ForwardBackwardResultT<double>;
using ForwardBackwardResult  = ForwardBackwardResultD;

template <typename Scalar>
ForwardBackwardResultT<Scalar> forward_backward(
    MatrixViewT<Scalar> log_likelihoods,
    MatrixViewT<Scalar> transition,
    const Scalar* initial,
    int num_states);

}  // namespace vbx
```

Reference: `VBx.py:forward_backward()`

## 5.2 VB-HMM iteration

```cpp
namespace vbx {

// Params struct stays double — small config with human-readable values.
// Internal math casts to Scalar once at entry.
struct VbhmmParams {
    double loop_prob      = 0.99;   // P(staying in same speaker state)
    double Fa             = 0.3;    // sufficient statistics scale
    double Fb             = 17.0;   // speaker regularization (higher = fewer speakers)
    int    max_speakers   = 10;     // upper bound on number of speakers
    int    max_iters      = 40;     // max VB iterations
    double epsilon        = 1e-6;   // ELBO convergence threshold
    double alpha_q_init   = 1.0;    // Dirichlet concentration for init
    double init_smoothing = 5.0;    // softmax temperature for AHC->VB init
};

template <typename Scalar>
struct VbhmmResultT {
    MatrixT<Scalar>     posteriors;     // T x S per-frame speaker posteriors (gamma)
    std::vector<Scalar> speaker_priors; // S-dim converged speaker priors (pi)
    std::vector<Scalar> elbo_history;   // ELBO value per iteration
};

using VbhmmResultF = VbhmmResultT<float>;
using VbhmmResultD = VbhmmResultT<double>;
using VbhmmResult  = VbhmmResultD;

template <typename Scalar>
VbhmmResultT<Scalar> vbhmm(
    MatrixViewT<Scalar> log_likelihoods,
    MatrixViewT<Scalar> gamma_init,     // rows==0 for random init
    const VbhmmParams& params = {});

}  // namespace vbx
```

Reference: `VBx.py:VBx()`

---

# Module 6 — End-to-end API (`vbx/vbx.h`)

## 6.1 Full pipeline

```cpp
#include <optional>

namespace vbx {

struct VbxParams {
    AhcParams     ahc     = {};
    VbhmmParams   vbhmm   = {};
    ScoringMode   scoring = ScoringMode::kCosine;
    int           lda_dim       = 128;   // only used when scoring == kPlda
    double        target_energy = 1.0;   // only used when scoring == kPlda
    bool          run_vbhmm     = true;  // false = AHC-only
};

// Full diarization pipeline: AHC init -> (optional) VB-HMM refinement -> merged segments.
//
// plda: required when params.scoring == kPlda, must be std::nullopt otherwise.
//       Throws std::invalid_argument on mismatch.
template <typename Scalar>
DiarResultT<Scalar> diarize(
    MatrixViewT<Scalar> xvecs,
    const Segment* segments,
    int num_segments,
    const std::optional<PldaModelT<Scalar>>& plda,
    const VbxParams& params = {});

}  // namespace vbx
```

## 6.2 Convenience overloads (vector-of-vectors style)

```cpp
namespace vbx {

template <typename Scalar>
DiarResultT<Scalar> diarize(
    const std::vector<std::vector<Scalar>>& xvecs,
    const std::vector<Segment>& segments,
    const std::optional<PldaModelT<Scalar>>& plda,
    const VbxParams& params = {});

}  // namespace vbx
```

## 6.3 Subset APIs

```cpp
namespace vbx {

template <typename Scalar>
std::vector<int> cluster(
    MatrixViewT<Scalar> xvecs,
    const AhcParams& params = {});

template <typename Scalar>
VbhmmResultT<Scalar> refine_vbhmm(
    MatrixViewT<Scalar> log_likelihoods,
    const std::vector<int>& initial_labels,
    const VbhmmParams& params = {});

}  // namespace vbx
```

---

# Module 7 — Python bindings (`bindings/python/bind_vbx.cpp`)

nanobind module exposing all APIs for both precisions. numpy dtype of the input
array selects the code path automatically (`float32` -> fp32, `float64` -> fp64).

```python
import vbx_native as vbx

# numpy dtype selects precision. Returned arrays match input precision.

# --- PLDA loading (precision selected explicitly) ---
plda_f32 = vbx.load_plda_f("model/plda")    # -> PldaModelF
plda_f64 = vbx.load_plda_d("model/plda")    # -> PldaModelD

# --- Clustering ---
sim = vbx.cosine_similarity(xvecs)                       # ndarray -> ndarray
cal = vbx.two_gmm_calibrate(scores)                      # ndarray -> (threshold, calibrated)
labels = vbx.ahc_cluster(sim_matrix, threshold=-0.015)   # ndarray -> ndarray[int]

# --- Scoring ---
cosine_scores = vbx.score_cosine(xvecs)                  # ndarray -> ndarray
plda_scores   = vbx.score_plda(xvecs, plda_model)        # ndarray -> ndarray
lls_cos  = vbx.compute_log_likelihoods_cosine(xvecs, Fa=0.3)
lls_plda = vbx.compute_log_likelihoods_plda(xvecs, plda_model, Fa=0.3)

# --- VB-HMM ---
fb  = vbx.forward_backward(lls, transition, initial)     # -> ForwardBackwardResult
res = vbx.vbhmm(lls, gamma_init, params)                 # -> VbhmmResult

# --- End-to-end ---
# Cosine mode (no PLDA model needed)
result = vbx.diarize(xvecs, segments, plda=None)

# PLDA mode
result = vbx.diarize(xvecs, segments, plda=plda_model,
                     params=vbx.VbxParams(scoring=vbx.ScoringMode.Plda))

# Subset APIs
labels  = vbx.cluster(xvecs, threshold=-0.015)           # -> ndarray[int]
refined = vbx.refine_vbhmm(lls, initial_labels, params)  # -> VbhmmResult
```

Exposed for **testing purposes** — later releases may reduce to high-level only.

---

# Module 8 — CLI utility (`tools/vbx_cli.cpp`)

```
Usage: vbx_cli --xvecs <ark_or_npy> --segments <seg_file> [options]

Options:
  --xvecs       Path to x-vectors (kaldi ark or .npy)
  --segments    Path to segments file (space-separated: start end per line)
  --precision   "f32" or "f64" (default: "f64")
  --scoring     Scoring mode: "cosine" (default) or "plda"
  --plda        Path to PLDA model file (required when --scoring=plda)
  --output      Path to output CSV (default: stdout)
  --threshold   AHC threshold bias (default: -0.015)
  --Fa          VB-HMM Fa (default: 0.3)
  --Fb          VB-HMM Fb (default: 17)
  --loop-prob   VB-HMM loop probability (default: 0.99)
  --ahc-only    Skip VB-HMM refinement, output AHC labels only
  --help        Show this help
```

Output CSV columns: `start,end,speaker_id`

---

# Functional requirements
1. Python interface runs around numpy ndarrays (use nanobind)
2. Clusterization-only, VBHMM-only and full end-to-end APIs are present
3. Balance between performance and readability, optimization might come later if required
4. Stateless implementation in functional style, no heavy initialization (unless there is some fat performance win)
5. Two scoring modes: cosine (default, no external model) and PLDA (requires model files)
6. Both fp32 and fp64 compute precision: public API uses explicit type aliases (`F`/`D`), internal math is templated on `Scalar`; PLDA model converted from on-disk fp32 to target precision at load time — no mixed-precision math in the pipeline

---

# Data flow diagram

```
                                                        Kaldi file (fp32 on disk)
                                                              |
                                                        load_plda<S>()
                                                              |
x-vectors (N x D)          segments (N x Segment)      optional<PldaModelT<S>>
 MatrixViewT<S>                                          (same S throughout)
       |                           |                           |
       v                           |                           |
 cosine_similarity<S>()            |                           |
       |                           |                           |
       v                           |                           |
 two_gmm_calibrate<S>()            |                           |
       |                           |                           |
       v                           |                           |
 ahc_cluster<S>()                  |                           |
       |                           |                           |
       +-- [AHC-only path] --------.----------> merge -----> DiarResultT<S>
       |                           |
       v                           |
 compute_log_likelihoods_XXX<S>() <.---------------------------+
       |                           |       (cosine or plda)
       v                           |
 vbhmm<S>() <- init from AHC      |
       |                           |
       v                           |
 posteriors -> argmax -------------+----------> merge -----> DiarResultT<S>

 S = float | double (uniform throughout, chosen by caller at entry)
```

---

# Dependencies

| Dependency  | Purpose                     | Integration            |
|-------------|-----------------------------|------------------------|
| fastcluster | Average-linkage AHC         | Vendored (header+cpp)  |
| nanobind    | Python bindings             | CMake FetchContent     |
| (none)      | Linear algebra (plain impl) | Built-in, swappable later |
