"""
test_aec_live.py -- tests AEC under REAL live timing, not saved files.
Plays a phrase out loud. At the same real time, feeds AEC both sound
streams as they truly happen: the exact sound going to the speaker,
and the real mic input, in matched, small, real-time steps.
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


print("Synthesizing test phrase...")
TEST_PHRASE = "The quick brown fox jumps over the lazy dog. Testing echo cancellation."
ref_pcm_raw, ref_sr = synthesize(TEST_PHRASE)
ref_16k = resample_linear(ref_pcm_raw, ref_sr, TARGET_SR)

ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
ap.set_stream_format(TARGET_SR, 1)
ap.set_reverse_stream_format(TARGET_SR, 1)
ap.set_stream_delay(50)  # now just a rough starting hint -- the live control part should track the rest
frame_size = ap.get_frame_size()
frame_seconds = frame_size / TARGET_SR

mic_frames = queue.Queue()


def mic_callback(indata, frames, time_info, status):
    mic_frames.put(indata[:, 0].copy())


output_chunks = []
lock = threading.Lock()


def feed_reverse_stream():
    """Feeds AEC the exact reference sound, paced to real time, to
    match when it is truly being sent to the speaker."""
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
save_wav("aec_output_live.wav", aec_output, TARGET_SR)
print(f"Saved aec_output_live.wav ({len(aec_output)/TARGET_SR:.2f}s)")
print("Listen to this one file, start to end.")
print("If it stays clean the whole way through -- not just in parts --")
print("that confirms real-time timing was the true, missing piece.")