# VBx — Speaker Diarization with VB-HMM over x-vectors

Speaker diarization pipeline based on x-vector extraction followed by Agglomerative Hierarchical Clustering (AHC) and Variational Bayes HMM (VB-HMM) clustering.

Based on: F. Landini, J. Profant, M. Diez, L. Burget — *Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization* (Computer Speech & Language, 2022).

## Installation

```bash
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

## Quick start

Run diarization on the included example audio:

```bash
python run_example.py --wav-dir example/audios/16k --lab-dir example/vad
```

This will:
1. Extract x-vectors from the audio using the bundled ResNet101 ONNX model
2. Run AHC + VB-HMM clustering to produce speaker labels
3. Write results to a temp directory (printed in the log output)

To specify an output directory:

```bash
python run_example.py --wav-dir example/audios/16k --lab-dir example/vad --out-dir results/
```

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

## License

Apache License 2.0 — see the original repository for details.
