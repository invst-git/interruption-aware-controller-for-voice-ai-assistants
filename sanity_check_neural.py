"""
sanity_check_neural.py -- checks the neural controller makes sensible
decisions on known text, before real audio enters the picture at all.
No microphone, no speaker -- just the model and some hand-picked words.

Not expected to match the rule-based controller exactly on every case
-- it's a different, learned decision boundary, not a lookup of the
same rule. The point is catching an obviously broken setup (wrong
tokenizer, wrong feature order, wrong threshold) cheaply, before adding
the complexity of real audio on top of an untested model.
"""
from neural_controller import NeuralController
from candor_fd_controller import TurnDecision

CHECKPOINT_PATH = "checkpoints/best_model_clean.pt"

# (words fed one at a time, time_into_other_turn in seconds, expected)
CASES = [
    (["yeah"], 5.0, TurnDecision.BACKCHANNEL),
    (["mhm", "okay"], 5.0, TurnDecision.BACKCHANNEL),
    (["gotcha"], 5.0, TurnDecision.BACKCHANNEL),
    (["wait"], 5.0, TurnDecision.STOP),
    (["no", "actually", "I", "think"], 5.0, TurnDecision.STOP),
    (["what", "about", "the", "price"], 5.0, TurnDecision.STOP),
    (["I'm", "not", "sure"], 5.0, TurnDecision.STOP),
]


def main():
    print("Loading checkpoint...")
    controller = NeuralController(CHECKPOINT_PATH)

    print(f"\n{'words':40} {'expected':13} {'got':13} match")
    print("-" * 80)
    n_match = 0
    for words, time_into_other_turn, expected in CASES:
        classifier = controller.new_classifier(time_into_other_turn)
        for w in words:
            classifier.add_word(w)
        got = classifier.decision
        ok = got == expected
        n_match += int(ok)
        print(f"{' '.join(words):40} {expected.name:13} {got.name:13} {'OK' if ok else 'MISMATCH'}")

    print(f"\n{n_match}/{len(CASES)} matched the expected direction.")
    print("A mismatch or two is not alarming by itself -- this is a learned")
    print("boundary, not a rule lookup. If MOST go the wrong way, or every")
    print("case comes back the same regardless of the words, stop here and")
    print("investigate before moving on to real audio.")


if __name__ == "__main__":
    main()
