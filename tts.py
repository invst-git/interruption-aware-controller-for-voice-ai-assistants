"""tts.py -- text-to-speech via the piper-tts Python package (in-process,
no subprocess, no manually-managed binary)."""
import io
import wave
import numpy as np
from piper import PiperVoice
from config import PIPER_VOICE_MODEL

_voice = PiperVoice.load(PIPER_VOICE_MODEL)


def synthesize(text: str):
    """Returns (pcm_float32, samplerate) ready for AudioPlayer.play()."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _voice.synthesize_wav(text, wav_file)
    buffer.seek(0)
    with wave.open(buffer, "rb") as wav_file:
        samplerate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm, samplerate