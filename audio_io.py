"""audio_io.py -- TTS playback with mid-playback stop support.

Microphone input lives entirely in vad_listener.py; this file only
handles output. AudioPlayer.stop() is what makes instant barge-in
cancellation possible -- unused until Step 4, but the reason local
Piper was chosen over a cloud TTS API.
"""
import threading
import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = None

    def play(self, pcm_data: np.ndarray, samplerate: int, blocking: bool = True):
        self._stop_flag.clear()
        started = threading.Event()

        def _run():
            sd.play(pcm_data, samplerate=samplerate)
            started.set()  # signal: sd.play() has now truly begun
            while True:
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    return
                if self._stop_flag.is_set():
                    sd.stop()
                    return
                sd.sleep(50)

        if blocking:
            _run()
        else:
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            started.wait(timeout=2.0)  # do not return until sound has truly started

    def stop(self):
        self._stop_flag.set()
        sd.stop()

    def is_playing(self) -> bool:
        try:
            stream = sd.get_stream()
        except RuntimeError:
            return False  # no sound has started yet -- this is not an error state
        return stream is not None and stream.active