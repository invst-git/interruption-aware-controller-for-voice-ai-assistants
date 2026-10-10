"""
test_aec_reset_check.py -- checks if a hard moment early in the clip
hurt the AEC filter's memory for the rest of the run. Re-runs AEC on
the SAME saved recording, but restarts the filter's memory partway
through, right after the rough patch. If the back half gets cleaner
this way, that confirms the early patch was the real cause.
"""
import wave
import numpy as np
from aec_audio_processing import AudioProcessor

TARGET_SR = 16000
RESET_AT_S = 2.0  # restart the filter's memory at this point
DELAY_MS = 546    # the stable value seen in three of four chunks


def load_wav(path):
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        raw = f.readframes(f.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def save_wav(path, audio_f32, sr):
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm16.tobytes())


def to_pcm16_bytes(chunk):
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def from_pcm16_bytes(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def new_ap():
    ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
    ap.set_stream_format(TARGET_SR, 1)
    ap.set_reverse_stream_format(TARGET_SR, 1)
    ap.set_stream_delay(DELAY_MS)
    return ap


ref, ref_sr = load_wav("aec_reference.wav")
mic, mic_sr = load_wav("aec_raw_mic.wav")
assert ref_sr == TARGET_SR and mic_sr == TARGET_SR, "Run test_aec_offline.py again first."

ap = new_ap()
frame_size = ap.get_frame_size()
n_frames = int(np.ceil(max(len(ref), len(mic)) / frame_size))
ref_p = np.pad(ref, (0, n_frames * frame_size - len(ref)))
mic_p = np.pad(mic, (0, n_frames * frame_size - len(mic)))
reset_frame = int((RESET_AT_S * TARGET_SR) / frame_size)

out_chunks = []
for i in range(n_frames):
    if i == reset_frame:
        print(f"Restarting filter memory at {RESET_AT_S:.1f}s...")
        ap = new_ap()
    ap.process_reverse_stream(to_pcm16_bytes(ref_p[i*frame_size:(i+1)*frame_size]))
    out = ap.process_stream(to_pcm16_bytes(mic_p[i*frame_size:(i+1)*frame_size]))
    out_chunks.append(from_pcm16_bytes(out))

aec_output = np.concatenate(out_chunks)
save_wav("aec_output_reset.wav", aec_output, TARGET_SR)
print(f"Raw mic peak: {np.abs(mic).max():.3f}  |  Reset-run peak: {np.abs(aec_output).max():.3f}")
print("Saved aec_output_reset.wav -- listen from 2s onward, and compare it")
print("directly against aec_output_measured.wav from the earlier run.")