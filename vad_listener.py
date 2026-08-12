"""
vad_listener.py -- continuous microphone listener built on Silero VAD.

Replaces Stage 1's crude RMS-threshold record_until_silence() with a
proper, always-on listener. Runs its own background thread reading the
mic in the exact 512-sample chunks Silero VAD requires, and tracks
speech segments as they happen.

Used two ways going forward:
  - Normal turn-taking (this step): call wait_for_utterance() to block
    until the user finishes a complete turn, and get back that segment's
    full audio.
  - During assistant playback (Step 4 / barge-in): poll is_speech_active
    and get_partial_audio() to catch the user starting to talk WHILE the
    assistant is still speaking, without waiting for them to finish.
"""
import threading
import queue
import numpy as np
import torch
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # required exact value for Silero VAD at 16kHz


class VADListener:
    def __init__(self, min_silence_duration_ms: int = 500):
        self._model = load_silero_vad(onnx=True)
        self._vad_iterator = VADIterator(
            self._model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self._stream = None
        self._events = queue.Queue()  # ("start", None) or ("end", audio_np_array)

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
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self._callback,
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
        """Audio accumulated so far for the CURRENTLY in-progress speech
        segment, if any. Not used yet -- this is what Step 4's barge-in
        detection will call during playback."""
        with self._lock:
            if not self._current_segment_chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._current_segment_chunks)

    def wait_for_utterance(self, timeout=None) -> np.ndarray:
        """Blocks until a complete speech segment (start -> end) happens,
        then returns its full audio. This is what replaces
        record_until_silence() for normal turn-taking."""
        while True:
            kind, payload = self._events.get(timeout=timeout)
            if kind == "end":
                return payload

    def poll_event(self, timeout=0.05):
        """Non-blocking check for the next start/end event, or None if
        nothing happened within `timeout`. Not used yet -- Step 4's
        barge-in loop needs this so it can check the controller between
        events instead of blocking entirely."""
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None