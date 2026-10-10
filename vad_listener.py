"""vad_listener.py -- continuous microphone listener built on Silero VAD,
now also handling TTS playback WITH echo-cancelled monitoring, through a
single shared duplex stream (play + record together, same clock).

Single-stream design exists because of a confirmed real bug: two
SEPARATE streams (one for play, one for record) do not start at the
same real moment -- proven directly in testing (mic read pure silence
for ~0.75s while playback had already begun). One shared stream removes
that gap, since one callback drives both jobs at the same instant, every time.
"""
import threading
import queue
import time
import numpy as np
import torch
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator
from aec_audio_processing import AudioProcessor

from config import AEC_STREAM_DELAY_MS

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512     # Silero VAD's required exact chunk size at 16kHz


def _resample_linear(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    target_n = int(len(audio) / orig_sr * target_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_n), np.arange(len(audio)), audio
    ).astype(np.float32)


def _to_pcm16_bytes(chunk):
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _from_pcm16_bytes(b):
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


class VADListener:
    def __init__(self, min_silence_duration_ms: int = 500):
        self._min_silence_ms = min_silence_duration_ms
        self._model = load_silero_vad(onnx=True)
        self._vad_iterator = VADIterator(
            self._model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=min_silence_duration_ms,
        )
        self._stream = None
        self._events = queue.Queue()
        self._speech_active = False
        self._current_segment_chunks = []
        self._lock = threading.Lock()

        # -- speaking-mode state (playback + AEC-monitored listening) --
        self._speaking = False
        self._stop_speaking_flag = threading.Event()
        self._speak_thread = None
        self._played_seconds = 0.0

    # ---------------- normal (idle) listening mode ----------------

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
            self._stream = None

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

    # ---------------- speaking mode: play + AEC-monitored listen ----------------

    def is_speaking(self) -> bool:
        return self._speaking

    def get_played_seconds(self) -> float:
        """How much of the most recent start_speaking() pcm has actually
        been sent to the output device, in seconds."""
        return self._played_seconds

    def start_speaking(self, pcm: np.ndarray, samplerate: int):
        """
        Plays `pcm` while listening through the SAME duplex stream, AEC
        cleaning the mic of the assistant's own voice before it reaches
        VAD/segment tracking. Runs on a background thread -- returns at
        once. Events during this window go through the SAME queue as
        normal listening, so poll_event()/get_partial_audio()/
        discard_active_segment() keep working exactly as before.
        """
        pcm_16k = _resample_linear(pcm, samplerate, SAMPLE_RATE)

        was_running = self._stream is not None
        if was_running:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._stop_speaking_flag.clear()
        self._played_seconds = 0.0
        self._speaking = True

        def _run():
            try:
                self._speak_duplex_session(pcm_16k)
            except Exception as e:
                print(f"[VADListener] ERROR during speaking session: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._speaking = False
                with self._lock:
                    self._speech_active = False
                    self._current_segment_chunks = []
                if was_running:
                    self.start()

        self._speak_thread = threading.Thread(target=_run, daemon=True)
        self._speak_thread.start()

    def stop_speaking(self):
        """Signals the current speaking session to stop playback at
        once (barge-in cancellation)."""
        self._stop_speaking_flag.set()

    def _speak_duplex_session(self, pcm: np.ndarray):
        ap = AudioProcessor(enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False)
        ap.set_stream_format(SAMPLE_RATE, 1)
        ap.set_reverse_stream_format(SAMPLE_RATE, 1)
        ap.set_stream_delay(AEC_STREAM_DELAY_MS)
        aec_frame_size = ap.get_frame_size()  # AEC's own fixed size -- 160 samples

        speak_vad_iterator = VADIterator(
            self._model, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=self._min_silence_ms,
        )
        speak_speech_active = False
        speak_segment_chunks = []
        play_pos = [0]
        stream_error = [None]

        # Silero VAD needs EXACTLY 512-sample chunks -- a different fixed
        # size from AEC's 160. 512 is not an even multiple of 160, so
        # cleaned audio is buffered across callbacks and released to VAD
        # in exact 512-sample windows, carrying any leftover forward.
        vad_buffer = np.array([], dtype=np.float32)

        def feed_vad(cleaned_chunk):
            nonlocal vad_buffer, speak_speech_active, speak_segment_chunks
            vad_buffer = np.concatenate([vad_buffer, cleaned_chunk])
            while len(vad_buffer) >= CHUNK_SAMPLES:
                vad_chunk = vad_buffer[:CHUNK_SAMPLES]
                vad_buffer = vad_buffer[CHUNK_SAMPLES:]
                result = speak_vad_iterator(torch.from_numpy(vad_chunk), return_seconds=True)
                with self._lock:
                    if speak_speech_active:
                        speak_segment_chunks.append(vad_chunk)
                    if result:
                        if "start" in result and not speak_speech_active:
                            speak_speech_active = True
                            speak_segment_chunks = [vad_chunk]
                            self._events.put(("start", None))
                        if "end" in result and speak_speech_active:
                            speak_speech_active = False
                            audio = np.concatenate(speak_segment_chunks)
                            speak_segment_chunks = []
                            self._events.put(("end", audio))
                    self._speech_active = speak_speech_active
                    self._current_segment_chunks = speak_segment_chunks

        def duplex_callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"[VADListener speaking] stream status: {status}")

            start = play_pos[0]
            end = start + frames
            block = pcm[start:end]
            if len(block) < frames:
                block = np.pad(block, (0, frames - len(block)))
            outdata[:, 0] = block
            play_pos[0] = end
            self._played_seconds = min(end, len(pcm)) / SAMPLE_RATE

            try:
                ap.process_reverse_stream(_to_pcm16_bytes(block))
                mic_block = indata[:, 0].copy()
                cleaned = _from_pcm16_bytes(ap.process_stream(_to_pcm16_bytes(mic_block)))
            except Exception as e:
                stream_error[0] = e
                return

            feed_vad(cleaned)

        n_frames_total = int(np.ceil(len(pcm) / aec_frame_size))
        duration_s = n_frames_total * aec_frame_size / SAMPLE_RATE + 0.3

        with sd.Stream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=aec_frame_size, callback=duplex_callback):
            start_time = time.monotonic()
            while time.monotonic() - start_time < duration_s:
                if self._stop_speaking_flag.is_set():
                    break
                if play_pos[0] >= len(pcm):
                    time.sleep(0.1)
                    break
                if stream_error[0] is not None:
                    print(f"[VADListener speaking] error: {stream_error[0]}")
                    break
                time.sleep(0.02)
        # A segment still open when this ends is left in shared state --
        # the caller (main.py) already decides what to do with that.