"""
benchmark_neural_inference.py -- measures how long a single, live-style
call to the trained controller takes on THIS machine's CPU.

Run this FIRST, before any more integration work. The model trained on
a qBraid A100. The live app runs on a CPU-only Windows machine. If a
single call is too slow, checking every ~300ms (the rule-based
controller's live cadence) will not work as designed, and the check
frequency needs to change before wiring anything into main.py.

Needs neural_controller.py and candor_fd_controller.py in the same
folder, plus best_model_clean.pt in this folder (or edit the path
below).
"""
import time

import numpy as np

from neural_controller import NeuralController

CHECKPOINT_PATH = "checkpoints/best_model_clean.pt"


def main():
    print("Loading checkpoint (slow, one-time cost -- not what's being measured)...")
    controller = NeuralController(CHECKPOINT_PATH)

    test_utterances = [
        ("yeah", 3.5),
        ("I think", 8.0),
        ("wait what", 1.2),
        ("no actually I", 0.5),
        ("I'm not sure about", 4.0),
    ]

    print("\nWarm-up call (first call is always slower, not representative)...")
    warm = controller.new_classifier(time_into_other_turn=2.0)
    warm.add_word("warm")
    _ = warm.decision

    print("\nTiming individual live-style calls (one at a time, matching real use):")
    times = []
    for text, time_into_other_turn in test_utterances:
        classifier = controller.new_classifier(time_into_other_turn)
        start = time.perf_counter()
        for word in text.split():
            classifier.add_word(word)
            _ = classifier.decision  # forces inference on this word, like the live app does
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms / len(text.split()))
        print(f"  {text!r:25} ({len(text.split())} words) "
              f"-> {elapsed_ms:6.1f} ms total, {elapsed_ms/len(text.split()):5.1f} ms/word")

    avg = np.mean(times)
    worst = np.max(times)
    print(f"\nAverage per-word call: {avg:.1f} ms   Worst: {worst:.1f} ms")
    print("\nFor context: the rule-based controller checks roughly every 300ms")
    print("during live playback, while also running mic capture, echo")
    print("cancellation, and speech recognition at the same time.")

    if avg > 150:
        print("\nThis is a meaningful fraction of that 300ms budget, on top of")
        print("everything else already competing for CPU time. Consider checking")
        print("less often than the rule-based version did, rather than assuming")
        print("the same cadence will work.")
    else:
        print("\nThis looks comfortably fast enough to check at the same cadence")
        print("the rule-based controller already uses.")


if __name__ == "__main__":
    main()
