# VBx — Speaker Diarization with VB-HMM over x-vectors

Speaker diarization pipeline based on x-vector extraction followed by Agglomerative Hierarchical Clustering (AHC) and Variational Bayes HMM (VB-HMM) clustering.

Based on: F. Landini, J. Profant, M. Diez, L. Burget — *Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization* (Computer Speech & Language, 2022).

Original project GitHub page: https://github.com/BUTSpeechFIT/VBx

## Installation

### Requirements

- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A C++17 compiler (clang, gcc, MSVC)
- CMake >= 3.16

nanobind is fetched automatically during the build — no manual install needed.

### Python package + native library

```bash
uv sync
```

This triggers the full build chain: **scikit-build-core** (the Python build backend) invokes **CMake** to compile `libvbx` (the C++17 diarization library), then **nanobind** generates `vbx_native` — a Python extension module that exposes the C++ API to Python. The pure Python package (`VBx/`) is included alongside it.

To verify the native module was compiled and is importable:

```bash
uv run python -c "import vbx_native; print(vbx_native.get_version())"
```

Hint: during development cycles to re-build C++ library and test it with Python fire

```bash
uv sync --reinstall-package vbx
```

### Standalone C++ build (no Python)

To build only the C++ library and CLI tool without any Python dependencies:

```bash
cd vbx_lib
cmake -B build -S .
cmake --build build
./build/vbx_cli
```

This builds `libvbx.a` and the `vbx_cli` executable. The Python bindings are skipped (controlled by the `VBX_BUILD_PYTHON_BINDINGS` CMake option, which defaults to OFF).
To enable it use:
```bash
cmake -B build -S . -DVBX_BUILD_PYTHON_BINDINGS=ON
```

### Benchmarks

To build and run the C++ benchmarks (using [Google Benchmark](https://github.com/google/benchmark)):

```bash
cd vbx_lib
cmake -B build -S . -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/benchmarks/run_benchmarks
```

Google Benchmark is fetched automatically via CMake FetchContent if not found on the system.

## Quick start

Run diarization on the included example audio:

```bash
uv run python run_example.py --wav-dir example/audios/16k --lab-dir example/vad
```

This will:
1. Extract x-vectors from the audio using the bundled ResNet101 ONNX model
2. Run AHC + VB-HMM clustering to produce speaker labels
3. Write results to a temp directory (printed in the log output)

To specify an output directory:

```bash
uv run python run_example.py --wav-dir example/audios/16k --lab-dir example/vad --out-dir results/
```

To run withoud PLDA and x-vector transformation fire:

```bash
uv run python run_example.py --wav-dir example/audios/16k --lab-dir example/vad --out-dir results/ --plda-file none --xvec-transform none
```

To run without PLDA using model that extracts x-vectors from PCM data:

```bash
 uv run python run_example.py --wav-dir example/audios/16k --lab-dir example/vad --out-dir results/pcm_no_plda --plda-file none --xvec-transform none --use-pcm --weights /path/to/model.onnx --backend onnx
```

## Running in VS Code / Zed

Project contatins default.launch.json file that contains pre-configured tasks to run. Make a copy
of it in .vscode directory and run the tasks from VS Code or Zed as ususal. Make sure you specify
correct virtual environment (e.g. .venv created by uv).

## Output

The output directory will contain, for each input wav file:

| File | Description |
|------|-------------|
| `<name>.ark` | Kaldi archive with extracted x-vectors (binary) |
| `<name>.seg` | Segment timing info for each x-vector |
| `<name>.rttm` | Diarization result in RTTM format |

The RTTM file contains one line per speech segment:
```
SPEAKER ES2005a 1 0.000000 7.560000 <NA> <NA> 23 <NA> <NA>
SPEAKER ES2005a 1 7.560000 6.000000 <NA> <NA> 27 <NA> <NA>
...
```

Fields: `SPEAKER <file> 1 <start_sec> <duration_sec> <NA> <NA> <speaker_id> <NA> <NA>`

## All options

```
python run_example.py \
    --wav-dir DIR          # directory with input .wav files (required)
    --lab-dir DIR          # directory with VAD .lab files (required)
    --out-dir DIR          # output directory (default: temp dir)
    --weights PATH         # model weights (default: bundled ONNX)
    --backend {onnx,pytorch}
    --xvec-transform PATH  # x-vector transform h5 file
    --plda-file PATH       # PLDA model file
    --init {AHC,AHC+VB}   # clustering method (default: AHC+VB)
    --threshold FLOAT      # AHC threshold (default: -0.015)
    --lda-dim INT          # LDA dimensionality (default: 128)
    --Fa FLOAT             # VB-HMM Fa param (default: 0.3)
    --Fb FLOAT             # VB-HMM Fb param (default: 17)
    --loopP FLOAT          # VB-HMM loop probability (default: 0.99)
```

## Programmatic usage

```python
from VBx.predict import load_model, extract_xvectors
from VBx.vbhmm import run_vbhmm

# or use the high-level wrapper:
from run_example import run_diarization

out_dir = run_diarization(
    wav_dir="path/to/wavs",
    lab_dir="path/to/vad_labels",
    out_dir="path/to/output",
)
```

## Input requirements

- **Audio**: `.wav` files, mono, 8kHz or 16kHz sample rate
- **VAD labels**: `.lab` files (one per wav, same base name) with two columns: `start_time end_time` in seconds

## Third-party dependencies (C++ library)

See `vbx_lib/THIRD_PARTY_NOTICES` for full attribution details.

## License

Apache License 2.0 — see the original repository for details.
