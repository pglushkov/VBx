import tempfile
from functools import lru_cache
from pathlib import Path

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def extract_vad_labs_into_temp_dir(wav_dir: Path | str) -> str:
    wav_dir = Path(wav_dir)
    outdir = tempfile.mkdtemp()
    for wav_file in sorted(wav_dir.glob("*.wav")):
        lab_path = extract_vad_for_wav_file(wav_file, Path(outdir))
        assert Path(lab_path).is_file(), f"VAD failed to produce {lab_path}"
    return outdir


@lru_cache(maxsize=1)
def load_silero_model():
    return load_silero_vad()


def extract_vad_for_wav_file(wav_file: Path, lab_dir: Path) -> str:
    """
    Function writes results of VAD into specified dir using name of input
    WAV file and changing extension to '.lab'. This lab-file needs to have
    contents expected by the rest of the VBX package, something like:

    spech_segment1_start_sec spech_segment1_end_sec sp
    ...
    spech_segmentN_start_sec spech_segmentN_end_sec sp

    No ida WTF we need 'sp' string constant in the end, ask authors of VBX.
    """
    model = load_silero_model()
    wav = read_audio(str(wav_file))

    # speech_timestamps = [{'start':float, 'end':float}, ...]
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    lab_file = Path(lab_dir) / f"{wav_file.stem}.lab"
    with open(lab_file, "w") as f:
        for ts in speech_timestamps:
            f.write(f"{ts['start']:.3f} {ts['end']:.3f} sp\n")
    return str(lab_file)
