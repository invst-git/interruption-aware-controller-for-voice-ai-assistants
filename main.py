"""main.py -- Step 4, hardened: real interruption handling with full-
utterance capture (not just the first fragment), a backchannel safety
re-check, and an iterative (not recursive) interruption chain."""
import time
import queue
import numpy as np
from faster_whisper import WhisperModel
from candor_fd_controller import is_likely_echo
from config import (
    ASR_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE, SAMPLE_RATE, VAD_MIN_SILENCE_MS,
    PARTIAL_CHECK_INTERVAL_S, MIN_PARTIAL_AUDIO_S, MAX_INTERRUPT_WAIT_S, CONTINUATION_GRACE_S,
)
from vad_listener import VADListener
from audio_io import AudioPlayer
from llm import generate_response
from tts import synthesize
from candor_fd_controller import StreamingTurnClassifier, TurnDecision, classify_complete_utterance


def _transcribe_words(asr_model, audio, fast=False):
    if len(audio) < SAMPLE_RATE * MIN_PARTIAL_AUDIO_S:
        return []
    kwargs = {"language": "en"}
    if fast:
        kwargs["beam_size"] = 1
    segments, _ = asr_model.transcribe(audio, **kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    return text.split() if text else []


def _capture_full_interruption(listener: VADListener, initial_audio, overall_deadline):
    """Collects the user's COMPLETE interrupting utterance -- waits
    briefly after a pause to see if more speech follows, instead of
    treating the first short burst as the whole thing."""
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


def speak_and_monitor(player: AudioPlayer, listener: VADListener, asr_model, pcm, sr):
    player.play(pcm, sr, blocking=False)
    classifier = None
    words_fed = 0
    last_check = 0.0

    while player.is_playing():
        event = listener.poll_event(timeout=0.05)
        if event:
            kind, payload = event
            if kind == "start":
                classifier = StreamingTurnClassifier()
                words_fed = 0
                last_check = 0.0
            elif kind == "end" and classifier is not None:
                if not classifier.words:
                    for w in _transcribe_words(asr_model, payload, fast=True):
                        classifier.add_word(w)
                if classifier.finalize() == TurnDecision.STOP:
                    player.stop()
                    print("  [!] Real interruption -- stopping")
                    deadline = time.monotonic() + MAX_INTERRUPT_WAIT_S
                    return _capture_full_interruption(listener, payload, deadline)
                classifier = None

        now = time.monotonic()
        if (classifier is not None and listener.is_speech_active
                and now - last_check >= PARTIAL_CHECK_INTERVAL_S):
            last_check = now
            words = _transcribe_words(asr_model, listener.get_partial_audio(), fast=True)
            for w in words[words_fed:]:
                classifier.add_word(w)
            words_fed = len(words)
            if classifier.decision == TurnDecision.STOP:
                player.stop()
                print("  [!] Real interruption -- stopping, waiting for you to finish")
                deadline = time.monotonic() + MAX_INTERRUPT_WAIT_S
                try:
                    remaining = max(deadline - time.monotonic(), 0.1)
                    first_segment = listener.wait_for_utterance(timeout=remaining)
                except queue.Empty:
                    first_segment = listener.get_partial_audio()
                    listener.discard_active_segment()
                return _capture_full_interruption(listener, first_segment, deadline)

    return None  # playback ended on its own -- no controller logic applies past this point


def run_turn(asr_model, listener, player, conversation_history, user_text):
    next_input = user_text
    while True:
        conversation_history.append({"role": "user", "content": next_input})
        print("[Thinking...]")
        reply = generate_response(conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}")

        print("[Speaking...]")
        pcm, sr = synthesize(reply)
        interrupt_audio = speak_and_monitor(player, listener, asr_model, pcm, sr)

        if interrupt_audio is None or len(interrupt_audio) < SAMPLE_RATE * 0.3:
            return

        print("[Transcribing interruption...]")
        interrupt_text = " ".join(_transcribe_words(asr_model, interrupt_audio, fast=False))
        if is_likely_echo(interrupt_text, reply):
            print(f'  [!] ("{interrupt_text}") looks like mic echo of the assistant\'s '
                  f"own voice -- ignoring, resuming")
            return
        if not interrupt_text:
            return

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
    listener = VADListener(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    listener.start()
    player = AudioPlayer()
    conversation_history = []

    print("\nCANDOR-FD -- Step 4 (hardened)")
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
            run_turn(asr_model, listener, player, conversation_history, text)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()