"""Tests for VBx.vad — silero VAD wrapper that writes .lab files.

Exercises both the single-file and whole-directory entrypoints against
the bundled example wav.  Lab files are written to a temp dir created
per-test and cleaned up on teardown.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from VBx.vad import extract_vad_for_wav_file, extract_vad_labs_into_temp_dir


EXAMPLE_WAV_DIR = Path(__file__).resolve().parent.parent / "example" / "audios" / "16k"
EXAMPLE_WAV = EXAMPLE_WAV_DIR / "ES2005a.wav"


def _assert_lab_file_is_valid(lab_path: Path):
    """Each line must be: '<start> <end> sp' with start < end and monotonic."""
    assert lab_path.is_file(), f"Missing lab file: {lab_path}"
    content = lab_path.read_text().strip()
    assert content, f"Lab file is empty: {lab_path}"

    prev_end = -1.0
    for line in content.splitlines():
        parts = line.split()
        assert len(parts) == 3, f"Bad line in {lab_path}: {line!r}"
        start, end, tag = float(parts[0]), float(parts[1]), parts[2]
        assert tag == "sp", f"Expected trailing 'sp', got {tag!r}"
        assert start < end, f"Non-positive segment: {line!r}"
        assert start >= prev_end, f"Segments not monotonic: {line!r}"
        prev_end = end


def test_extract_vad_for_wav_file():
    """Single-file processing writes a valid .lab next to its stem."""
    assert EXAMPLE_WAV.is_file(), f"Example wav missing: {EXAMPLE_WAV}"

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        lab_path_str = extract_vad_for_wav_file(EXAMPLE_WAV, tmp_dir)
        lab_path = Path(lab_path_str)

        assert lab_path.parent == tmp_dir
        assert lab_path.name == f"{EXAMPLE_WAV.stem}.lab"
        _assert_lab_file_is_valid(lab_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_extract_vad_labs_into_temp_dir():
    """Directory processing produces one .lab per wav in the input dir."""
    assert EXAMPLE_WAV_DIR.is_dir(), f"Example wav dir missing: {EXAMPLE_WAV_DIR}"

    out_dir = extract_vad_labs_into_temp_dir(EXAMPLE_WAV_DIR)
    try:
        out_path = Path(out_dir)
        wav_stems = {p.stem for p in EXAMPLE_WAV_DIR.glob("*.wav")}
        lab_stems = {p.stem for p in out_path.glob("*.lab")}
        assert wav_stems == lab_stems, (
            f"Lab files don't match wavs: wavs={wav_stems}, labs={lab_stems}"
        )
        for lab in out_path.glob("*.lab"):
            _assert_lab_file_is_valid(lab)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
