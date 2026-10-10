"""main.py -- real interruption handling, now with echo-cancelled
in-app playback and monitoring, both through vad_listener.py's single
duplex stream.

Controller choice: config.USE_NEURAL_CONTROLLER selects between the
rule-based StreamingTurnClassifier and the trained neural controller.
Defaults to the rule-based one -- the neural path has not yet been
tested against real microphone audio, only offline test rows.

NEURAL CONTROLLER CADENCE, read before changing speak_and_monitor()
--------------------------------------------------------------------
Both controllers get live, mid-utterance checks every
PARTIAL_CHECK_INTERVAL_S (0.3s) while the user is still talking. For
the neural controller on today's CPU-only deployment,
classifier.decision measures ~1-3s per call (see
benchmark_neural_inference.py plus live timing under
config.DEBUG_TIMING) -- slower than a human's short-interruption
duration, and slower than PARTIAL_CHECK_INTERVAL_S. That's a known,
accepted latency cost of this hardware, not something this file works
around; it's expected to shrink to a non-issue once inference runs on
a real GPU -- either locally, or via config.NEURAL_CONTROLLER_MODE =
"remote" calling modal_controller.py's Modal GPU service (see
neural_controller.py's module docstring for that split).

What this file DOES guarantee regardless of how slow or fast a single
call is: NeuralStreamingTurnClassifier caches its verdict per
word-list (add_word() marks it dirty, nothing else does), so a
mid-utterance check landing on BACKCHANNEL leaves a stale cached
verdict sitting there. If "end" simply called finalize() on top of
that unchanged word list, it would silently return the stale verdict
instead of reflecting the now-complete segment -- confirmed directly
in scratchpad/confirm_stale_finalize.py (finalize() returning in
~0ms with the identical mid-utterance verdict). So on "end", this loop
always resets the classifier and re-feeds it the freshly transcribed,
complete word list before calling finalize() -- the end-of-utterance
decision is always made on the real, full transcript, never a cached
partial one. This applies to both controller types (reset() is part
of the shared classifier interface), though it only ever changes
StreamingTurnClassifier's answer in the rare case partial checks
missed trailing words -- its decision was never cache-based to begin
with."""
import time
import queue
import numpy as np
from faster_whisper import WhisperModel

from config import (
    ASR_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE, SAMPLE_RATE, VAD_MIN_SILENCE_MS,
    PARTIAL_CHECK_INTERVAL_S, MIN_PARTIAL_AUDIO_S, MAX_INTERRUPT_WAIT_S, CONTINUATION_GRACE_S,
    MIN_RESUME_TAIL_S, USE_NEURAL_CONTROLLER, NEURAL_CONTROLLER_CHECKPOINT,
    NEURAL_CONTROLLER_THRESHOLD, DEBUG_TIMING, NEURAL_CONTROLLER_MODE,
    MODAL_CONTROLLER_URL, MODAL_CONTROLLER_API_KEY, NEURAL_REMOTE_TIMEOUT_S,
)
from vad_listener import VADListener
from llm import generate_response
from tts import synthesize
from candor_fd_controller import (
    StreamingTurnClassifier, TurnDecision, classify_complete_utterance, is_likely_echo,
)


_T0 = time.monotonic()


def _log_t(msg):
    if DEBUG_TIMING:
        print(f"  [t={time.monotonic() - _T0:7.3f}s] {msg}")


def _transcribe_words(asr_model, audio, fast=False):
    if len(audio) < SAMPLE_RATE * MIN_PARTIAL_AUDIO_S:
        return []
    kwargs = {"language": "en"}
    if fast:
        kwargs["beam_size"] = 1
    segments, _ = asr_model.transcribe(audio, **kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    return text.split() if text else []


def _capture_full_interruption(listener, initial_audio, overall_deadline):
    combined = [initial_audio] if len(initial_audio) > 0 else []
    while True:
        remaining = overall_deadline - time.monotonic()
        wait_time = min(CONTINUATION_GRACE_S, remaining)
        if wait_time <= 0:
            break
        try:
            combined.append(listener.wait_for_utterance(timeout=wait_time))
        except queue.Empty:
            break
    return np.concatenate(combined) if combined else initial_audio


def _new_classifier(listener, neural_controller):
    """The one place a fresh per-utterance classifier gets created.
    Both classifier types expose add_word()/decision/finalize()/words/
    reset(), so nothing downstream needs to know which one it got."""
    if neural_controller is not None:
        return neural_controller.new_classifier(listener.get_played_seconds())
    return StreamingTurnClassifier()


def speak_and_monitor(listener: VADListener, asr_model, neural_controller, pcm, sr):
    listener.start_speaking(pcm, sr)
    classifier = None
    words_fed = 0
    last_check = 0.0

    while listener.is_speaking():
        event = listener.poll_event(timeout=0.05)
        if event:
            kind, payload = event
            _log_t(f"VAD '{kind}' event reached main.py")
            if kind == "start":
                classifier = _new_classifier(listener, neural_controller)
                words_fed = 0
                last_check = 0.0
            elif kind == "end" and classifier is not None:
                # Always re-transcribe the complete, just-finished segment
                # and reset the classifier onto that word list before
                # deciding -- see the NEURAL CONTROLLER CADENCE note at
                # the top of this file for why: a mid-utterance check may
                # have already cached a verdict on a partial prefix, and
                # that verdict must not be what finalize() answers with.
                final_words = _transcribe_words(asr_model, payload, fast=True)
                classifier.reset()
                for w in final_words:
                    classifier.add_word(w)
                t_decision = time.monotonic()
                decision = classifier.finalize()
                _log_t(f"classifier.finalize() -> {decision} "
                       f"({(time.monotonic() - t_decision) * 1000:.0f}ms)")
                if decision == TurnDecision.STOP:
                    listener.stop_speaking()
                    _log_t("listener.stop_speaking() called")
                    print("  [!] Real interruption -- stopping")
                    deadline = time.monotonic() + MAX_INTERRUPT_WAIT_S
                    interrupt_audio = _capture_full_interruption(listener, payload, deadline)
                    return interrupt_audio, listener.get_played_seconds()
                classifier = None

        now = time.monotonic()
        if (classifier is not None and listener.is_speech_active
                and now - last_check >= PARTIAL_CHECK_INTERVAL_S):
            last_check = now
            words = _transcribe_words(asr_model, listener.get_partial_audio(), fast=True)
            for w in words[words_fed:]:
                classifier.add_word(w)
            words_fed = len(words)
            t_decision = time.monotonic()
            decision = classifier.decision
            _log_t(f"classifier.decision (mid-utterance) -> {decision} "
                   f"({(time.monotonic() - t_decision) * 1000:.0f}ms)")
            if decision == TurnDecision.STOP:
                listener.stop_speaking()
                _log_t("listener.stop_speaking() called")
                print("  [!] Real interruption -- stopping, waiting for you to finish")
                deadline = time.monotonic() + MAX_INTERRUPT_WAIT_S
                try:
                    remaining = max(deadline - time.monotonic(), 0.1)
                    first_segment = listener.wait_for_utterance(timeout=remaining)
                except queue.Empty:
                    first_segment = listener.get_partial_audio()
                    listener.discard_active_segment()
                interrupt_audio = _capture_full_interruption(listener, first_segment, deadline)
                return interrupt_audio, listener.get_played_seconds()

    return None, None


def _speak_with_resume(listener, asr_model, neural_controller, pcm, sr):
    """Speak pcm and monitor for interruptions. If a stop turns out to be
    a false positive (no real words in the careful re-transcription),
    resume speaking the unplayed remainder instead of going silent.
    Returns (interrupt_audio, interrupt_text) for a real interruption,
    or (None, None) once the reply has fully gotten through."""
    remaining_pcm = pcm
    while True:
        interrupt_audio, played_s = speak_and_monitor(listener, asr_model, neural_controller, remaining_pcm, sr)
        if interrupt_audio is None:
            return None, None

        interrupt_text = ""
        if len(interrupt_audio) >= SAMPLE_RATE * 0.3:
            print("[Transcribing interruption...]")
            interrupt_text = " ".join(_transcribe_words(asr_model, interrupt_audio, fast=False))
        if interrupt_text:
            return interrupt_audio, interrupt_text

        cut_sample = int(played_s * sr)
        remaining_pcm = remaining_pcm[cut_sample:]
        if len(remaining_pcm) < sr * MIN_RESUME_TAIL_S:
            print("  [!] No speech found in the interruption -- likely self-echo, reply was almost done")
            return None, None

        print("  [!] No speech found in the interruption -- likely self-echo, resuming reply")


def run_turn(asr_model, listener, neural_controller, conversation_history, user_text):
    next_input = user_text
    while True:
        conversation_history.append({"role": "user", "content": next_input})
        print("[Thinking...]")
        reply = generate_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}")

        print("[Speaking...]")
        pcm, sr = synthesize(reply)
        interrupt_audio, interrupt_text = _speak_with_resume(listener, asr_model, neural_controller, pcm, sr)

        if interrupt_audio is None:
            return

        if is_likely_echo(interrupt_text, reply):
            print(f'  [!] ("{interrupt_text}") looks like mic echo -- ignoring, resuming')
            return

        # This safety re-check always uses the rule, regardless of which
        # controller ran live. The neural controller was trained on
        # short causal prefixes with a timing feature that doesn't apply
        # to a finished, arbitrary-length utterance -- this check needs
        # a tool built for that shape of input, not the streaming one.
        if classify_complete_utterance(interrupt_text) == TurnDecision.BACKCHANNEL:
            print(f'  [!] ("{interrupt_text}") re-checked as a backchannel -- resuming')
            return

        print(f"You interrupted with: {interrupt_text}")
        cut_off_reply = conversation_history.pop()["content"]
        next_input = (
            f'[You interrupted the assistant while it was saying: "{cut_off_reply}"]\n'
            f"{interrupt_text}"
        )


def main():
    print("Loading Faster-Whisper model...")
    asr_model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)

    neural_controller = None
    if USE_NEURAL_CONTROLLER:
        if NEURAL_CONTROLLER_MODE == "remote":
            print("Connecting to remote (Modal) neural controller...")
            from neural_controller import RemoteNeuralController
            neural_controller = RemoteNeuralController(
                MODAL_CONTROLLER_URL, MODAL_CONTROLLER_API_KEY,
                threshold=NEURAL_CONTROLLER_THRESHOLD, timeout_s=NEURAL_REMOTE_TIMEOUT_S,
            )
        else:
            print("Loading local neural controller...")
            from neural_controller import NeuralController
            neural_controller = NeuralController(NEURAL_CONTROLLER_CHECKPOINT, threshold=NEURAL_CONTROLLER_THRESHOLD)

    listener = VADListener(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    listener.start()
    conversation_history = []

    print("\nCANDOR-FD -- live app, echo-cancelled duplex audio")
    if neural_controller is None:
        print("Controller: rule-based")
    else:
        print(f"Controller: neural ({NEURAL_CONTROLLER_MODE})")
    print("Ctrl+C to quit.\n")
    try:
        while True:
            print("[Listening...]")
            audio = listener.wait_for_utterance()
            if len(audio) < SAMPLE_RATE * 0.3:
                continue
            print("[Transcribing...]")
            text = " ".join(_transcribe_words(asr_model, audio, fast=False))
            if not text:
                continue
            print(f"You said: {text}")
            run_turn(asr_model, listener, neural_controller, conversation_history, text)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()