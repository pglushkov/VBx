#!/usr/bin/env python
"""VBx speaker diarization pipeline.

Replaces run_example.sh — runs x-vector extraction followed by VB-HMM clustering.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

from VBx.predict import extract_xvectors_melbank, extract_xvectors_pcm, load_model
from VBx.vbhmm import run_vbhmm

log_level = os.environ.get("LOG_LEVEL", "INFO")
print(f" === using log-level: {log_level}")
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Default model paths (relative to this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_HERE, "VBx/models/ResNet101_16kHz/nnet/final.onnx")
_DEFAULT_TRANSFORM = os.path.join(_HERE, "VBx/models/ResNet101_16kHz/transform.h5")
_DEFAULT_PLDA = os.path.join(_HERE, "VBx/models/ResNet101_16kHz/plda")
_PCM_HOP_LEN_SEC = 0.24  # 240 msec from original model


def run_diarization(
    wav_dir: str | Path,
    lab_dir: str | Path,
    out_dir: str | Path | None = None,
    weights: str | Path = _DEFAULT_WEIGHTS,
    backend: str = "onnx",
    xvec_transform: str | Path | None = _DEFAULT_TRANSFORM,
    plda_file: str | Path | None = _DEFAULT_PLDA,
    threshold: float = -0.015,
    lda_dim: int = 128,
    Fa: float = 0.3,
    Fb: float = 17.0,
    loopP: float = 0.99,
    score: bool = False,
    ref_rttm_dir: str | Path | None = None,
    use_pcm: bool = False,
    input_samplerate: int = 16000,
):
    """Run the full diarization pipeline on all wav files in a directory.

    Args:
        wav_dir: directory with input wav files
        lab_dir: directory with VAD label files (.lab)
        out_dir: output directory for results (default: temp directory)
        weights: path to model weights (ONNX or PyTorch)
        backend: 'onnx' or 'pytorch'
        xvec_transform: path to x-vector transform h5 file
        plda_file: path to PLDA model
        init: 'AHC' or 'AHC+VB'
        threshold: AHC threshold
        lda_dim: LDA dimensionality
        Fa: VB-HMM Fa parameter
        Fb: VB-HMM Fb parameter
        loopP: VB-HMM loop probability
        score: whether to score against reference (not yet implemented)
        ref_rttm_dir: directory with reference RTTM files (for scoring)

    Returns:
        str: path to the output directory containing RTTM files
    """
    if score:
        raise NotImplementedError(
            "Scoring is not yet supported in the Python pipeline. "
            "Use dscore/score.py directly if needed."
        )

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="vbx_")
        logger.info(f"Using temp output directory: {out_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Discover audio files
    wav_files = sorted(
        os.path.splitext(f)[0] for f in os.listdir(wav_dir) if f.endswith(".wav")
    )
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {wav_dir}")

    logger.info(f"Found {len(wav_files)} audio file(s)")

    # Load model once
    model, label_name, input_name, device, input_shape = load_model(
        weights, backend=backend
    )

    for filename in wav_files:
        logger.info(f"Processing: {filename}")

        ark_file_name = os.path.join(out_dir, f"{filename}.ark")
        seg_file_name = os.path.join(out_dir, f"{filename}.seg")

        # Step 1: Extract x-vectors
        if use_pcm:
            assert isinstance(input_shape, list) and len(input_shape) == 2
            segment_len_samples = input_shape[1]
            extract_xvectors_pcm(
                file_names=[filename],
                wav_dir=wav_dir,
                lab_dir=lab_dir,
                out_ark_fn=ark_file_name,
                out_seg_fn=seg_file_name,
                model=model,
                label_name=label_name,
                input_name=input_name,
                seg_len_samples=segment_len_samples,
                seg_jump_samples=int(_PCM_HOP_LEN_SEC * input_samplerate),
                device=device,
                backend=backend,
            )
        else:
            extract_xvectors_melbank(
                file_names=[filename],
                wav_dir=wav_dir,
                lab_dir=lab_dir,
                out_ark_fn=ark_file_name,
                out_seg_fn=seg_file_name,
                model=model,
                label_name=label_name,
                input_name=input_name,
                device=device,
                backend=backend,
            )

        # Step 2: VB-HMM clustering
        run_vbhmm(
            xvec_ark_file=ark_file_name,
            segments_file=seg_file_name,
            xvec_transform=xvec_transform,
            plda_file=plda_file,
            out_rttm_dir=out_dir,
            threshold=threshold,
            lda_dim=lda_dim,
            Fa=Fa,
            Fb=Fb,
            loopP=loopP,
        )

    logger.info(f"Results written to: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="VBx speaker diarization pipeline")
    parser.add_argument(
        "--wav-dir", required=True, help="Directory with input wav files"
    )
    parser.add_argument(
        "--lab-dir", required=True, help="Directory with VAD label files"
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default: temp dir)"
    )
    parser.add_argument(
        "--weights", default=_DEFAULT_WEIGHTS, help="Path to model weights"
    )
    parser.add_argument("--backend", default="onnx", choices=["onnx", "pytorch"])
    parser.add_argument("--xvec-transform", default=_DEFAULT_TRANSFORM)
    parser.add_argument("--plda-file", default=_DEFAULT_PLDA)
    parser.add_argument("--threshold", type=float, default=-0.015)
    parser.add_argument("--lda-dim", type=int, default=128)
    parser.add_argument("--Fa", type=float, default=0.3)
    parser.add_argument("--Fb", type=float, default=17.0)
    parser.add_argument("--loopP", type=float, default=0.99)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score against reference (not yet implemented)",
    )
    parser.add_argument(
        "--ref-rttm-dir", default=None, help="Reference RTTM directory (for scoring)"
    )
    parser.add_argument(
        "--use-pcm",
        required=False,
        default=False,
        action="store_true",
        help="set this flag to forward raw audio to the model",
    )
    parser.add_argument(
        "--in-samplerate",
        required=False,
        default=16000,
        type=int,
        help="Samplerate of audio expected by the model when using raw PCMs",
    )

    args = parser.parse_args()

    xvec_transform_file = (
        None if args.xvec_transform.lower() == "none" else args.xvec_transform
    )
    plda_file = None if args.plda_file.lower() == "none" else args.plda_file

    run_diarization(
        wav_dir=args.wav_dir,
        lab_dir=args.lab_dir,
        out_dir=args.out_dir,
        weights=args.weights,
        backend=args.backend,
        xvec_transform=xvec_transform_file,
        plda_file=plda_file,
        threshold=args.threshold,
        lda_dim=args.lda_dim,
        Fa=args.Fa,
        Fb=args.Fb,
        loopP=args.loopP,
        score=args.score,
        ref_rttm_dir=args.ref_rttm_dir,
        use_pcm=args.use_pcm,
        input_samplerate=args.in_samplerate,
    )


if __name__ == "__main__":
    main()
