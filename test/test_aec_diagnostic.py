"""
test_aec_diagnostic.py -- ONE thorough, live test that checks three
competing ideas at once: does poor cancellation track quiet moments,
loud moments, or a steady clock? Saves all three signals plus a table
comparing them window by window.
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
WINDOW_S = 0.25


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


PHRASE = "the quick brown fox jumps over the lazy dog while running a long echo cancellation test today"
print(f"Synthesizing: {PHRASE!r}")
ref_pcm_raw, ref_sr = synthesize(PHRASE)
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


raw_mic_chunks = []
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
        raw_mic_chunks.append(chunk)
        with lock:
            out = ap.process_stream(to_pcm16_bytes(chunk))
        output_chunks.append(from_pcm16_bytes(out))
    sd.wait()

raw_mic = np.concatenate(raw_mic_chunks) if raw_mic_chunks else np.array([], dtype=np.float32)
aec_output = np.concatenate(output_chunks) if output_chunks else np.array([], dtype=np.float32)

save_wav("diag_reference.wav", ref_16k, TARGET_SR)
save_wav("diag_raw_mic.wav", raw_mic, TARGET_SR)
save_wav("diag_aec_output.wav", aec_output, TARGET_SR)
print("Saved diag_reference.wav, diag_raw_mic.wav, diag_aec_output.wav\n")

window_samples = int(WINDOW_S * TARGET_SR)
n_windows = min(len(ref_16k), len(raw_mic), len(aec_output)) // window_samples

print(f"{'time':>10} {'ref RMS':>10} {'raw mic RMS':>13} {'AEC out RMS':>13} {'cut %':>8}")
for i in range(n_windows):
    s, e = i*window_samples, (i+1)*window_samples
    ref_rms = np.sqrt(np.mean(ref_16k[s:e]**2))
    raw_rms = np.sqrt(np.mean(raw_mic[s:e]**2))
    out_rms = np.sqrt(np.mean(aec_output[s:e]**2))
    cut_pct = (1 - out_rms/raw_rms) * 100 if raw_rms > 0.001 else float("nan")
    flag = "  <-- POOR CUT" if (raw_rms > 0.02 and cut_pct < 40) else ""
    print(f"{i*WINDOW_S:>9.2f}s {ref_rms:>10.4f} {raw_rms:>13.4f} {out_rms:>13.4f} {cut_pct:>7.1f}%{flag}")

print("\nFor every row marked POOR CUT, check its ref RMS:")
print("  LOW ref RMS there  -> supports the quiet-moment idea")
print("  HIGH ref RMS there -> supports a loud-word distortion idea")
print("  Evenly spaced, regardless of ref RMS -> supports an internal-timing idea")