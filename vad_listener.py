"""vad_listener.py -- continuous microphone listener built on Silero VAD."""
import threading
import queue
import time
import numpy as np
import torch
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512


class VADListener:
    def __init__(self, min_silence_duration_ms: int = 500):
        self._model = load_silero_vad(onnx=True)
        self._vad_iterator = VADIterator(
            self._model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=min_silence_duration_ms,
        )
        self._stream = None
        self._events = queue.Queue()
        self._speech_active = False
        self._current_segment_chunks = []
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[VADListener] stream status: {status}")
        chunk_np = indata[:, 0].copy()
        result = self._vad_iterator(torch.from_numpy(chunk_np), return_seconds=True)
        with self._lock:
            if self._speech_active:
                self._current_segment_chunks.append(chunk_np)
            if result:
                if "start" in result and not self._speech_active:
                    self._speech_active = True
                    self._current_segment_chunks = [chunk_np]
                    self._events.put(("start", None))
                if "end" in result and self._speech_active:
                    self._speech_active = False
                    audio = np.concatenate(self._current_segment_chunks)
                    self._current_segment_chunks = []
                    self._events.put(("end", audio))

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK_SAMPLES, callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    @property
    def is_speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def get_partial_audio(self) -> np.ndarray:
        with self._lock:
            if not self._current_segment_chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._current_segment_chunks)

    def wait_for_utterance(self, timeout=None) -> np.ndarray:
        """Blocks until a complete speech segment happens. Tracks a real
        deadline so an intervening 'start' event can't silently reset
        the timeout clock."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0)
            if remaining is not None and remaining <= 0:
                raise queue.Empty()
            kind, payload = self._events.get(timeout=remaining)
            if kind == "end":
                return payload

    def poll_event(self, timeout=0.05):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def discard_active_segment(self):
        with self._lock:
            self._speech_active = False
            self._current_segment_chunks = []
            self._vad_iterator.reset_states()