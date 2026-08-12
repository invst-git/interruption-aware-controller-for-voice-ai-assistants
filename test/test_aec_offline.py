"""
test_aec_offline.py -- v2. Measures the real delay between reference
and mic via cross-correlation instead of guessing 50ms, then runs AEC
both ways so you can directly compare.
"""
import sys
sys.path.append("..")
import wave
import numpy as np
import sounddevice as sd
from aec_audio_processing import AudioProcessor
from tts import synthesize

TARGET_SR = 16000


def resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    target_n = int(len(audio) / orig_sr * target_sr)
    return np.interp(np.linspace(0, len(audio) - 1, target_n), np.arange(len(audio)), audio).astype(np.float32)


def save_wav(path, audio_f32, sr):
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm16.tobytes())


def to_pcm16_bytes(chunk):
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def from_pcm16_bytes(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def run_aec(ref, mic, stream_delay_ms):
    ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
    ap.set_stream_format(TARGET_SR, 1)
    ap.set_reverse_stream_format(TARGET_SR, 1)
    ap.set_stream_delay(stream_delay_ms)
    frame_size = ap.get_frame_size()
    n_frames = int(np.ceil(max(len(ref), len(mic)) / frame_size))
    ref_p = np.pad(ref, (0, n_frames * frame_size - len(ref)))
    mic_p = np.pad(mic, (0, n_frames * frame_size - len(mic)))
    out_chunks = []
    for i in range(n_frames):
        ap.process_reverse_stream(to_pcm16_bytes(ref_p[i*frame_size:(i+1)*frame_size]))
        out = ap.process_stream(to_pcm16_bytes(mic_p[i*frame_size:(i+1)*frame_size]))
        out_chunks.append(from_pcm16_bytes(out))
    return np.concatenate(out_chunks)


print("Synthesizing test phrase...")
TEST_PHRASE = "The quick brown fox jumps over the lazy dog. Testing echo cancellation."
ref_pcm, ref_sr = synthesize(TEST_PHRASE)
ref_pcm_padded = np.concatenate([ref_pcm, np.zeros(int(0.5 * ref_sr), dtype=np.float32)])

print(f"Playing {len(ref_pcm_padded)/ref_sr:.2f}s and recording at the same time. STAY QUIET.\n")
recording = sd.playrec(ref_pcm_padded.reshape(-1, 1), samplerate=ref_sr, channels=1, dtype="float32")
sd.wait()

ref_16k = resample_linear(ref_pcm_padded, ref_sr, TARGET_SR)
raw_mic_16k = resample_linear(recording[:, 0].copy(), ref_sr, TARGET_SR)
save_wav("aec_reference.wav", ref_16k, TARGET_SR)
save_wav("aec_raw_mic.wav", raw_mic_16k, TARGET_SR)

# --- Measure the real delay instead of guessing ---
correlation = np.correlate(raw_mic_16k, ref_16k, mode="full")
best_lag = np.argmax(np.abs(correlation)) - (len(ref_16k) - 1)
measured_delay_ms = (best_lag / TARGET_SR) * 1000
print(f"Measured delay (cross-correlation): {best_lag} samples = {measured_delay_ms:.1f}ms")
if best_lag < 0:
    print("NOTE: negative -- mic appears to LEAD the reference. Worth flagging if this shows up.")

aec_guessed = run_aec(ref_16k, raw_mic_16k, 50)
save_wav("aec_output_guessed_50ms.wav", aec_guessed, TARGET_SR)

safe_delay_ms = max(int(round(measured_delay_ms)), 0)
aec_measured = run_aec(ref_16k, raw_mic_16k, safe_delay_ms)
save_wav("aec_output_measured.wav", aec_measured, TARGET_SR)

raw_peak = np.abs(raw_mic_16k).max()
print(f"\nRaw mic peak:                      {raw_peak:.3f}")
print(f"AEC, guessed 50ms:                 {np.abs(aec_guessed).max():.3f}  "
      f"({(1 - np.abs(aec_guessed).max()/raw_peak)*100:.1f}% reduction)")
print(f"AEC, measured {safe_delay_ms}ms:              {np.abs(aec_measured).max():.3f}  "
      f"({(1 - np.abs(aec_measured).max()/raw_peak)*100:.1f}% reduction)")

print("\nListen in order: aec_raw_mic.wav -> aec_output_guessed_50ms.wav -> aec_output_measured.wav")