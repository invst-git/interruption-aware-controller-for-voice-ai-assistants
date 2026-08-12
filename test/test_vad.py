"""
test_vad.py -- standalone test to verify Silero VAD correctly detects
speech start/end from your microphone in real time, before we build
anything else on top of it.

Silero VAD requires EXACTLY 512-sample audio chunks at 16kHz -- this
isn't a tunable setting, it's a hard requirement of the model itself,
so the mic stream below is configured to match exactly.
"""
import torch
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # required exact value for Silero VAD at 16kHz -- do not change

print("Loading Silero VAD...")
model = load_silero_vad(onnx=True)  # ONNX runtime -- lighter and faster than full torch inference
vad_iterator = VADIterator(model, sampling_rate=SAMPLE_RATE)

print("Listening... speak, pause, speak again. Ctrl+C to stop.\n")


def callback(indata, frames, time_info, status):
    if status:
        print(f"[stream status] {status}")
    # .copy() matters: sounddevice reuses this buffer on the next callback,
    # and torch.from_numpy shares memory rather than copying it.
    chunk = torch.from_numpy(indata[:, 0].copy())
    result = vad_iterator(chunk, return_seconds=True)
    if result:
        if "start" in result:
            print(f">>> Speech STARTED at {result['start']:.2f}s")
        if "end" in result:
            print(f"<<< Speech ENDED at {result['end']:.2f}s")


try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopped.")