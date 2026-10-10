"""
test_aec_drift_check.py -- checks if the delay between play and mic
changes over time (drift). Reuses the WAV files from test_aec_offline.py.
Splits them into small time chunks and checks the true delay in each
chunk on its own, instead of using one number for the whole clip.
"""
import wave
import numpy as np

TARGET_SR = 16000
WINDOW_S = 1.0  # size of each time chunk, in seconds


def load_wav(path):
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        raw = f.readframes(f.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


ref, ref_sr = load_wav("aec_reference.wav")
mic, mic_sr = load_wav("aec_raw_mic.wav")
assert ref_sr == TARGET_SR and mic_sr == TARGET_SR, "Run test_aec_offline.py again first."

window_samples = int(WINDOW_S * TARGET_SR)
n_windows = min(len(ref), len(mic)) // window_samples

print(f"Checking delay in {n_windows} chunks of {WINDOW_S:.0f}s each:\n")

delays_ms = []
for i in range(n_windows):
    start = i * window_samples
    end = start + window_samples
    ref_chunk = ref[start:end]
    mic_chunk = mic[start:end]

    correlation = np.correlate(mic_chunk, ref_chunk, mode="full")
    best_lag = np.argmax(np.abs(correlation)) - (len(ref_chunk) - 1)
    delay_ms = (best_lag / TARGET_SR) * 1000
    delays_ms.append(delay_ms)
    print(f"  Chunk {i} ({i*WINDOW_S:.1f}s to {(i+1)*WINDOW_S:.1f}s): delay = {delay_ms:.1f}ms")

spread = max(delays_ms) - min(delays_ms)
print(f"\nDelay spread across all chunks: {spread:.1f}ms")
if spread > 20:
    print("This is a real, growing change. This points to drift as the cause.")
else:
    print("This stays close to flat. Drift is likely NOT the cause.")
    print("Something else may explain the echo return, such as loud-word distortion.")