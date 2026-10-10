"""
test_aec_duplex.py -- the corrected setup. ONE single stream drives
BOTH play and record through the SAME callback, for the SAME instant,
every time. This removes the separate-stream timing gap found in the
last test (mic caught pure silence for 0.75s while sound was already
playing -- proof the two old streams were never truly in step).
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
ref_16k = np.concatenate([ref_16k, np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)])
print(f"Clip length: {len(ref_16k)/TARGET_SR:.2f}s")

ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
ap.set_stream_format(TARGET_SR, 1)
ap.set_reverse_stream_format(TARGET_SR, 1)
ap.set_stream_delay(50)
frame_size = ap.get_frame_size()

play_pos = [0]
raw_mic_chunks = []
output_chunks = []


def duplex_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"[stream status] {status}")

    # This block's OUTGOING audio -- the exact samples about to reach the speaker.
    start = play_pos[0]
    end = start + frames
    block = ref_16k[start:end]
    if len(block) < frames:
        block = np.pad(block, (0, frames - len(block)))
    outdata[:, 0] = block
    play_pos[0] = end

    # Feed AEC the SAME block as the reference, for the SAME callback --
    # guaranteed aligned by the audio driver, not by our own guesswork.
    ap.process_reverse_stream(to_pcm16_bytes(block))

    # Feed AEC the mic input captured in this SAME callback.
    mic_block = indata[:, 0].copy()
    raw_mic_chunks.append(mic_block)
    out = ap.process_stream(to_pcm16_bytes(mic_block))
    output_chunks.append(from_pcm16_bytes(out))


print("Playing and recording through ONE single stream. STAY QUIET.\n")
with sd.Stream(samplerate=TARGET_SR, channels=1, dtype="float32",
               blocksize=frame_size, callback=duplex_callback):
    sd.sleep(int((len(ref_16k) / TARGET_SR + 0.5) * 1000))

raw_mic = np.concatenate(raw_mic_chunks)
aec_output = np.concatenate(output_chunks)
save_wav("duplex_raw_mic.wav", raw_mic, TARGET_SR)
save_wav("duplex_aec_output.wav", aec_output, TARGET_SR)

check_samples = int(0.75 * TARGET_SR)
print("Saved duplex_raw_mic.wav and duplex_aec_output.wav")
print(f"\nSanity check -- raw mic RMS in the first 0.75s:")
print(f"  {np.sqrt(np.mean(raw_mic[:check_samples]**2)):.4f}")
print("  (should be a REAL number now, not 0.0000, if this fixed the gap)")
print("\nListen to duplex_aec_output.wav, start to end.")