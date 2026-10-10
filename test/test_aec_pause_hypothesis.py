"""
test_aec_pause_hypothesis.py -- Part 1: checks the ALREADY-SAVED
reference audio for a real, visible quiet gap. Part 2: re-runs the
live AEC test, but with ONE continuous phrase, no internal pause,
close to the same length. If Part 2 stays clean where the old one did
not, that confirms the pause -- not plain elapsed time -- is the
real cause.
"""
import sys
sys.path.append("..")
import time
import threading
import queue
import wave
import numpy as np
import sounddevice as sd
from aec_audio_processing import AudioProcessor
from tts import synthesize

TARGET_SR = 16000

# --- Part 1: free check on the already-saved file ---
print("=== Part 1: checking the OLD phrase for a quiet gap ===")
with wave.open("aec_reference.wav", "rb") as f:
    sr = f.getframerate()
    raw = f.readframes(f.getnframes())
ref_old = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

window_s = 0.25
window_samples = int(window_s * sr)
n_windows = len(ref_old) // window_samples
for i in range(n_windows):
    chunk = ref_old[i*window_samples:(i+1)*window_samples]
    rms = np.sqrt(np.mean(chunk**2))
    marker = "  <-- LOW" if rms < 0.02 else ""
    print(f"  {i*window_s:.2f}s-{(i+1)*window_s:.2f}s: RMS={rms:.4f}{marker}")
print("\nA clear LOW point near 2.0-3.0s would match the pause between")
print("the two sentences in the old phrase.\n")


# --- Part 2: live re-test with ONE phrase, no internal pause ---
def resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    target_n = int(len(audio) / orig_sr * target_sr)
    return np.interp(np.linspace(0, len(audio) - 1, target_n), np.arange(len(audio)), audio).astype(np.float32)


def to_pcm16_bytes(chunk):
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def from_pcm16_bytes(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def save_wav(path, audio_f32, sr):
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm16.tobytes())


print("=== Part 2: live AEC test, ONE continuous phrase, no mid-pause ===")
NEW_PHRASE = "the quick brown fox jumps over the lazy dog while running a long echo cancellation test today"
print(f"Synthesizing: {NEW_PHRASE!r}")
ref_pcm_raw, ref_sr = synthesize(NEW_PHRASE)
ref_16k = resample_linear(ref_pcm_raw, ref_sr, TARGET_SR)
print(f"Clip length: {len(ref_16k)/TARGET_SR:.2f}s")

ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
ap.set_stream_format(TARGET_SR, 1)
ap.set_reverse_stream_format(TARGET_SR, 1)
ap.set_stream_delay(50)
frame_size = ap.get_frame_size()
frame_seconds = frame_size / TARGET_SR

mic_frames = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    mic_frames.put(indata[:, 0].copy())


output_chunks = []
lock = threading.Lock()


def feed_reverse_stream():
    n_frames = len(ref_16k) // frame_size
    next_time = time.monotonic()
    for i in range(n_frames):
        chunk = ref_16k[i*frame_size:(i+1)*frame_size]
        with lock:
            ap.process_reverse_stream(to_pcm16_bytes(chunk))
        next_time += frame_seconds
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


print("Playing live and running AEC in real time. STAY QUIET.\n")
reverse_thread = threading.Thread(target=feed_reverse_stream, daemon=True)

with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                     blocksize=frame_size, callback=mic_callback):
    sd.play(ref_16k, samplerate=TARGET_SR)
    reverse_thread.start()
    start_time = time.monotonic()
    duration = len(ref_16k) / TARGET_SR + 1.0
    while time.monotonic() - start_time < duration:
        try:
            chunk = mic_frames.get(timeout=0.1)
        except queue.Empty:
            continue
        with lock:
            out = ap.process_stream(to_pcm16_bytes(chunk))
        output_chunks.append(from_pcm16_bytes(out))
    sd.wait()

aec_output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)
save_wav("aec_output_nopause.wav", aec_output, TARGET_SR)
print(f"Saved aec_output_nopause.wav ({len(aec_output)/TARGET_SR:.2f}s)")
print("Listen start to end. If this stays clean where the old phrase did not,")
print("that confirms the pause -- not plain elapsed time -- breaks the lock.")