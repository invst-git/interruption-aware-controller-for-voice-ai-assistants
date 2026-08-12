"""
test_vad_listener.py -- verify VADListener captures whole speech
segments correctly, matching what you'd expect from your own turns.
"""
import sys
sys.path.append("..")  # lets us import vad_listener.py from the project root

from vad_listener import VADListener

listener = VADListener(min_silence_duration_ms=500)
listener.start()

print("Listening for utterances. Speak a full sentence, pause, repeat. Ctrl+C to stop.\n")

try:
    while True:
        audio = listener.wait_for_utterance()
        duration = len(audio) / 16000
        print(f"Got an utterance: {len(audio)} samples ({duration:.2f}s)")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    listener.stop()