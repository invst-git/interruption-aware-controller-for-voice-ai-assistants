"""
main.py -- Step 4: real interruption handling.

Wires candor_fd_controller.py into the loop built in Step 3. While the
app speaks, new user speech is checked live, word by word, using
periodic re-transcription of the growing in-progress audio (Faster-
Whisper has no true live streaming mode, so this is the closest
practical stand-in). A STOP verdict halts playback at once. A
BACKCHANNEL verdict does nothing -- the app keeps talking.

Core rule: the controller only runs WHILE the app is speaking. The
instant playback stops, for any reason, plain listening takes over --
no controller logic applies past that point.
"""
import time
import queue
from faster_whisper import WhisperModel

from config import (
    ASR_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE, SAMPLE_RATE, VAD_MIN_SILENCE_MS,
    PARTIAL_CHECK_INTERVAL_S, MIN_PARTIAL_AUDIO_S, MAX_INTERRUPT_WAIT_S,
)
from vad_listener import VADListener
from audio_io import AudioPlayer
from llm import generate_response
from tts import synthesize
from candor_fd_controller import StreamingTurnClassifier, TurnDecision


def _transcribe_words(asr_model, audio, fast=False):
    """Runs Faster-Whisper on `audio`, returns a plain list of words.
    fast=True uses greedy decoding for quick live checks (speed matters
    more than perfect accuracy here). fast=False is the normal, higher-
    quality mode used for text that actually reaches the LLM."""
    if len(audio) < SAMPLE_RATE * MIN_PARTIAL_AUDIO_S:
        return []
    kwargs = {"language": "en"}
    if fast:
        kwargs["beam_size"] = 1
    segments, _ = asr_model.transcribe(audio, **kwargs)
    text = " ".join(seg.text for seg in segments).strip()
    return text.split() if text else []


def speak_and_monitor(player: AudioPlayer, listener: VADListener, asr_model, pcm, sr):
    """
    Plays the reply while checking, live, for a real interruption.
    Returns interrupt_audio: the full captured audio of the user's
    interrupting turn, or None if none happened.
    """
    player.play(pcm, sr, blocking=False)

    classifier = None
    words_fed = 0
    last_check = 0.0
    interrupt_audio = None

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
                    # Segment closed before our first live check ever ran
                    # (very short sound). Check its real content directly,
                    # instead of judging on zero words -- a bare "0 words"
                    # would wrongly read as a real interrupt otherwise.
                    for w in _transcribe_words(asr_model, payload, fast=True):
                        classifier.add_word(w)
                decision = classifier.finalize()
                if decision == TurnDecision.STOP:
                    player.stop()
                    interrupt_audio = payload
                    print("  [!] Real interruption -- stopping")
                    break
                classifier = None  # was BACKCHANNEL -- keep talking

        now = time.monotonic()
        if (classifier is not None
                and listener.is_speech_active
                and now - last_check >= PARTIAL_CHECK_INTERVAL_S):
            last_check = now
            words = _transcribe_words(asr_model, listener.get_partial_audio(), fast=True)
            for w in words[words_fed:]:
                classifier.add_word(w)
            words_fed = len(words)
            if classifier.decision == TurnDecision.STOP:
                player.stop()
                print("  [!] Real interruption -- stopping, waiting for you to finish")
                try:
                    interrupt_audio = listener.wait_for_utterance(timeout=MAX_INTERRUPT_WAIT_S)
                except queue.Empty:
                    interrupt_audio = listener.get_partial_audio()
                    listener.discard_active_segment()
                break

    # If playback ended on its own, with a segment still open, we
    # deliberately do nothing here. No controller logic applies once the
    # app isn't speaking -- the caller's normal listening step picks up
    # this same open segment on its own, as an ordinary next turn.
    return interrupt_audio


def run_turn(asr_model, listener, player, conversation_history, user_text):
    """Handles one full turn: reply, speak, and react if interrupted."""
    conversation_history.append({"role": "user", "content": user_text})
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
    if not interrupt_text:
        return  # false alarm -- likely noise, not real speech

    print(f"You interrupted with: {interrupt_text}")
    # The reply just above was cut off -- you may not have heard all of
    # it, so it's removed here rather than kept as a normal, fully-heard
    # turn. What it WAS going to say is folded into the next user message
    # as plain context instead.
    cut_off_reply = conversation_history.pop()["content"]
    new_user_text = (
        f'[You interrupted the assistant while it was saying: "{cut_off_reply}"]\n'
        f"{interrupt_text}"
    )
    run_turn(asr_model, listener, player, conversation_history, new_user_text)


def main():
    print("Loading Faster-Whisper model...")
    asr_model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)

    listener = VADListener(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    listener.start()
    player = AudioPlayer()
    conversation_history = []

    print("\nCANDOR-FD -- Step 4 (real interruption handling)")
    print("Backchannels (yeah/mhm/okay) should NOT stop it. Real interruptions should.")
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