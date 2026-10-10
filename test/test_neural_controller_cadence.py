"""
test_neural_controller_cadence.py -- headless (no mic, no real ASR)
regression test for the neural-controller correctness bug in
main.py::speak_and_monitor().

Background: NeuralStreamingTurnClassifier caches its verdict per
word-list (add_word() marks it dirty, nothing else does). Mid-utterance
partial checks run live (every PARTIAL_CHECK_INTERVAL_S, same cadence
as the rule-based controller -- accepted latency cost on CPU hardware,
expected to be fine on GPU). If a mid-utterance check landed on
BACKCHANNEL from a short prefix, and "end" then called finalize()
without feeding it anything new, finalize() returned that SAME stale
cached verdict instead of reflecting the now-complete segment --
confirmed directly in scratchpad/confirm_stale_finalize.py
(finalize() returning in ~0ms with the identical mid-utterance
verdict). A real interruption whose start sounded like a backchannel
("yeah... no wait, stop") got silently dropped.

The fix: on "end", main.py always resets the classifier and re-feeds
it the freshly transcribed COMPLETE word list before calling
finalize() -- the end-of-utterance decision is always made on the
real, full transcript, never a cached partial one.

This test drives the real, patched speak_and_monitor() with a stubbed
listener and ASR, no mic or real Whisper needed. Cases:

  1. A prefix that reads as BACKCHANNEL mid-utterance, but the complete
     segment is a real interruption -- must correctly flip to STOP,
     with the audio returned matching the original ended segment.
  2. A genuine backchannel from start to finish -- must NOT stop.
  3. A long (>5 word) complete segment -- MAX_PREFIX_WORDS shortcut
     fires with zero model calls at finalize (see the comment next to
     MAX_PREFIX_WORDS in neural_controller.py for why that's still
     safe: run_turn()'s rule-based re-check is the second opinion).

Uses the real, trained NeuralController (real inference) -- needs
checkpoints/best_model_clean.pt, same as the live app.
"""
import os
import sys
import types
import time
import queue
import numpy as np

sys.path.append("..")  # lets this run as `python test/test_neural_controller_cadence.py`


def _stub_module(name, **attrs):
    """This sandbox has no audio hardware and doesn't have
    faster-whisper/sounddevice/silero-vad/piper installed -- only
    speak_and_monitor()'s control flow is under test here, so those
    packages are stubbed out just enough that `import main` succeeds.
    None of their real behavior is exercised by this test."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(mod, attr, value)
    sys.modules[name] = mod


_stub_module("faster_whisper", WhisperModel=object)
_stub_module("sounddevice", InputStream=object, Stream=object)
_stub_module("silero_vad", load_silero_vad=lambda **kw: None, VADIterator=object)
_stub_module("aec_audio_processing", AudioProcessor=object)
_stub_module("piper", PiperVoice=type("PiperVoice", (), {"load": staticmethod(lambda *a, **kw: None)}))

import main as main_module
import neural_controller as neural_controller_module
from neural_controller import NeuralController

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model_clean.pt")
SR = 16000
PARTIAL_CHECK_INTERVAL_S = main_module.PARTIAL_CHECK_INTERVAL_S


class StubListener:
    """Simulates a segment that's still being spoken: while `hold_s`
    hasn't elapsed, is_speech_active is True, get_partial_audio()
    returns `partial_audio`, and no event is delivered -- long enough
    for at least one real mid-utterance partial check to run, exactly
    as happens live while the user keeps talking. After the hold, the
    "end" event delivers `end_audio` as the complete segment."""

    def __init__(self, partial_audio, end_audio, hold_s, played_seconds=2.0):
        self._partial_audio = partial_audio
        self._end_audio = end_audio
        self._hold_s = hold_s
        self._start_delivered = False
        self._end_delivered = False
        self._speaking = True
        self._played_seconds = played_seconds
        self._t_started = None

    def start_speaking(self, pcm, sr):
        pass

    def is_speaking(self):
        return self._speaking

    def get_played_seconds(self):
        return self._played_seconds

    def poll_event(self, timeout=0.05):
        if not self._start_delivered:
            self._start_delivered = True
            self._t_started = time.monotonic()
            return "start", None
        if not self._end_delivered and time.monotonic() - self._t_started >= self._hold_s:
            self._end_delivered = True
            self._speaking = False
            return "end", self._end_audio
        time.sleep(min(timeout, 0.02))
        return None

    @property
    def is_speech_active(self):
        return not self._end_delivered

    def get_partial_audio(self):
        return self._partial_audio if self._start_delivered and not self._end_delivered else np.array([], dtype=np.float32)

    def wait_for_utterance(self, timeout=None):
        raise queue.Empty()  # the neural path must not need this

    def discard_active_segment(self):
        pass

    def stop_speaking(self):
        pass


# audio payloads are just markers -- the stub ASR below maps them to text
NO_AUDIO = np.array([], dtype=np.float32)
BACKCHANNEL_PREFIX_AUDIO = np.full(int(SR * 0.3), 111.0, dtype=np.float32)
REAL_INTERRUPT_FULL_AUDIO = np.full(int(SR * 1.2), 222.0, dtype=np.float32)
PURE_BACKCHANNEL_AUDIO = np.full(int(SR * 0.3), 333.0, dtype=np.float32)
LONG_INTERRUPT_AUDIO = np.full(int(SR * 1.5), 444.0, dtype=np.float32)

TEXT_BY_MARKER = {
    111.0: "yeah",                                            # sounds like a backchannel mid-utterance...
    222.0: "yeah no wait stop",                                # ...but the complete segment is a real interruption
    333.0: "yeah",                                             # genuine backchannel, start to finish
    444.0: "no actually I think the price is way too high",   # >5 words -- MAX_PREFIX_WORDS shortcut
}


class StubASR:
    def transcribe(self, audio, **kwargs):
        marker = float(audio[0]) if len(audio) else 0.0
        text = TEXT_BY_MARKER.get(marker, "")
        Seg = type("Seg", (), {"text": text})
        return ([Seg()] if text else []), None


def run_case(name, partial_audio, end_audio, neural_controller, expect_stop, min_infer_calls=1):
    # Hold long enough that at least one real mid-utterance partial
    # check fires before "end" -- matching live behavior now that
    # mid-utterance checking is restored for the neural controller too.
    listener = StubListener(partial_audio, end_audio, hold_s=PARTIAL_CHECK_INTERVAL_S * 1.5)
    asr = StubASR()

    infer_calls = []
    orig_infer = neural_controller_module.NeuralStreamingTurnClassifier._infer

    def counting_infer(self):
        infer_calls.append(1)
        return orig_infer(self)

    neural_controller_module.NeuralStreamingTurnClassifier._infer = counting_infer
    try:
        result_audio, played_s = main_module.speak_and_monitor(
            listener, asr, neural_controller, pcm=np.zeros(10), sr=SR,
        )
    finally:
        neural_controller_module.NeuralStreamingTurnClassifier._infer = orig_infer

    ok = True
    if expect_stop:
        if result_audio is None:
            print(f"[{name}] FAIL: expected STOP, got no interruption at all -- "
                  f"this is exactly the stale-cache bug (real interruption dropped)")
            ok = False
        elif not np.array_equal(result_audio[:len(end_audio)], end_audio):
            print(f"[{name}] FAIL: interrupt_audio does not match the complete ended "
                  f"segment -- wrong audio captured")
            ok = False
    else:
        if result_audio is not None:
            print(f"[{name}] FAIL: expected no interruption (backchannel), but got a STOP")
            ok = False

    if len(infer_calls) < min_infer_calls:
        print(f"[{name}] FAIL: expected at least {min_infer_calls} real model inference(s), "
              f"got {len(infer_calls)}")
        ok = False

    if ok:
        print(f"[{name}] PASS ({len(infer_calls)} inference call(s))")
    return ok


def main():
    print("Loading neural controller (real weights, real inference -- no mic needed)...")
    controller = NeuralController(CHECKPOINT_PATH)

    results = [
        run_case(
            "mid-utterance BACKCHANNEL prefix, but complete segment is a real interruption -> STOP",
            BACKCHANNEL_PREFIX_AUDIO, REAL_INTERRUPT_FULL_AUDIO, controller, expect_stop=True,
        ),
        run_case(
            "genuine backchannel start to finish -> no stop",
            PURE_BACKCHANNEL_AUDIO, PURE_BACKCHANNEL_AUDIO, controller, expect_stop=False,
        ),
        # >5-word finished segment: MAX_PREFIX_WORDS shortcut fires on the
        # complete transcript at finalize (see the comment in
        # neural_controller.py next to MAX_PREFIX_WORDS) -- STOP without
        # a model call there; run_turn()'s rule-based
        # classify_complete_utterance() re-check is the second opinion
        # that protects against a hallucinated word count.
        run_case(
            "long real interruption -> STOP via word-count shortcut at finalize",
            NO_AUDIO, LONG_INTERRUPT_AUDIO, controller, expect_stop=True, min_infer_calls=0,
        ),
    ]

    if all(results):
        print("\nAll cases passed.")
    else:
        print("\nSome cases FAILED -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
