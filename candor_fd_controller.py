"""candor_fd_controller.py -- streaming backchannel vs. interruption
classifier, based on CANDOR's Backbiter rule. Lexicon (v3) validated
against 557,864 real conversation turns. Not wired into the live loop
until Step 4."""
from dataclasses import dataclass, field
from enum import Enum


class TurnDecision(Enum):
    PENDING = "PENDING"
    BACKCHANNEL = "BACKCHANNEL"
    STOP = "STOP"


BACKCHANNEL_WORDS = {
    "yeah", "mhm", "mm", "hmm", "yep", "yup", "huh", "yes",
    "wow", "oh", "okay", "ah", "exactly", "alright", "gotcha",
}

PROHIBITED_STARTS = {
    "i'm", "i", "no", "wait", "stop", "actually", "but", "sorry", "so",
}


def _clean(word: str) -> str:
    return word.lower().strip(".,!?;:")


@dataclass
class StreamingTurnClassifier:
    words: list = field(default_factory=list)
    decision: TurnDecision = TurnDecision.PENDING

    def reset(self):
        self.words = []
        self.decision = TurnDecision.PENDING

    def add_word(self, word: str) -> TurnDecision:
        self.words.append(word)
        n = len(self.words)
        first = _clean(self.words[0])

        if first in PROHIBITED_STARTS:
            self.decision = TurnDecision.STOP
        elif n > 3:
            self.decision = TurnDecision.STOP
        else:
            bc_count = sum(1 for w in self.words if _clean(w) in BACKCHANNEL_WORDS)
            majority = (bc_count / n) > 0.5
            if n == 3 and not majority:
                self.decision = TurnDecision.STOP
            elif majority:
                self.decision = TurnDecision.BACKCHANNEL
            else:
                self.decision = TurnDecision.PENDING
        return self.decision

    def finalize(self) -> TurnDecision:
        if self.decision != TurnDecision.PENDING:
            return self.decision
        self.decision = (
            TurnDecision.BACKCHANNEL if _is_backchannel_final(self.words) else TurnDecision.STOP
        )
        return self.decision


def _is_backchannel_final(words: list) -> bool:
    n = len(words)
    if n == 0 or n > 3:
        return False
    if _clean(words[0]) in PROHIBITED_STARTS:
        return False
    bc_count = sum(1 for w in words if _clean(w) in BACKCHANNEL_WORDS)
    return (bc_count / n) > 0.5


def classify_complete_utterance(text: str) -> TurnDecision:
    words = text.split()
    return TurnDecision.BACKCHANNEL if _is_backchannel_final(words) else TurnDecision.STOP