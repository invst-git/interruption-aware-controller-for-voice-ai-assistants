"""main.py -- Step 3: concurrent listening + non-blocking playback.

Turn-taking runs through VADListener. TTS plays on a background thread
while the main thread keeps polling the mic, so talking mid-playback is
DETECTED and logged in real time. It does not yet stop the assistant --
that decision is Step 4 (wiring in candor_fd_controller.py)."""
import numpy as np
from faster_whisper import WhisperModel

from config import (
    ASR_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE, SAMPLE_RATE, VAD_MIN_SILENCE_MS,
)
from vad_listener import VADListener
from audio_io import AudioPlayer
from llm import generate_response
from tts import synthesize


def speak_and_monitor(player: AudioPlayer, listener: VADListener, pcm, sr):
    player.play(pcm, sr, blocking=False)
    interruption_in_progress = False
    while player.is_playing():
        event = listener.poll_event(timeout=0.05)
        if event:
            kind, payload = event
            if kind == "start" and not interruption_in_progress:
                interruption_in_progress = True
                print("  [!] User started talking while assistant is speaking")
            elif kind == "end" and interruption_in_progress:
                duration = len(payload) / SAMPLE_RATE
                print(f"  [!] User's speech ended ({duration:.2f}s) -- not acting yet (Step 4)")
                interruption_in_progress = False
    if interruption_in_progress:
        print("  [!] Playback finished while user was still talking -- "
              "will pick up their full utterance next")


def main():
    print("Loading Faster-Whisper model...")
    asr_model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)

    listener = VADListener(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    listener.start()
    player = AudioPlayer()
    conversation_history = []

    print("\nCANDOR-FD -- Step 3 (concurrent listening, no interruption REACTION yet)")
    print("Talk, pause, get a reply. While it speaks, try talking -- you'll see it")
    print("DETECTED live even though it keeps talking. Ctrl+C to quit.\n")

    try:
        while True:
            print("[Listening...]")
            audio = listener.wait_for_utterance()
            if len(audio) < SAMPLE_RATE * 0.3:
                continue

            print("[Transcribing...]")
            segments, _ = asr_model.transcribe(audio, language="en")
            text = " ".join(seg.text for seg in segments).strip()
            if not text:
                continue
            print(f"You said: {text}")

            conversation_history.append({"role": "user", "content": text})
            print("[Thinking...]")
            reply = generate_response(conversation_history)
            conversation_history.append({"role": "assistant", "content": reply})
            print(f"Assistant: {reply}")

            print("[Speaking...]")
            pcm, sr = synthesize(reply)
            speak_and_monitor(player, listener, pcm, sr)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()