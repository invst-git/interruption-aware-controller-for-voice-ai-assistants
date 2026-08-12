"""test_capture_quality.py -- saves exactly what VADListener captures,
so you can listen to it and compare against what you actually said.
Isolates bad capture from bad transcription."""
import sys
sys.path.append("..")
import wave
import numpy as np
from vad_listener import VADListener, SAMPLE_RATE

listener = VADListener(min_silence_duration_ms=500)
listener.start()

print("Say one sentence, then pause.\n")
audio = listener.wait_for_utterance()
listener.stop()

pcm16 = (audio * 32767).astype(np.int16)
with wave.open("captured.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes(pcm16.tobytes())

peak = np.abs(audio).max()
rms = np.sqrt(np.mean(audio**2))
print(f"Saved {len(audio)/SAMPLE_RATE:.2f}s to captured.wav")
print(f"Peak amplitude: {peak:.3f}  (near 1.0 repeatedly = clipping)")
print(f"RMS level: {rms:.4f}  (very low = mic too quiet / wrong device)")