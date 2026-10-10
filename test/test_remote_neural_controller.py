"""
test_remote_neural_controller.py -- headless test for
RemoteNeuralController/RemoteNeuralStreamingTurnClassifier in
neural_controller.py, with requests.Session.post mocked (no real
network call, no Modal account needed to run this).

Covers:
  1. A normal successful call: STOP/BACKCHANNEL decoded from the mocked
     JSON response, and the request body/auth header sent are correct.
  2. The MAX_PREFIX_WORDS shortcut still applies locally -- a >5-word
     segment must NOT make an HTTP call at all.
  3. reset() correctly clears cached state (same contract as the local
     NeuralStreamingTurnClassifier and StreamingTurnClassifier).
  4. Network failure (timeout/connection error) falls back to the
     rule-based classify_complete_utterance() heuristic instead of
     raising or hanging -- and does not crash speak_and_monitor()'s
     control flow, which has no other error handling around
     classifier.decision/finalize().

Mocks requests.Session.post (class-level) rather than requests.post --
RemoteNeuralController keeps one requests.Session per instance and
reuses it for every call (measured against the real deployed endpoint:
~1.4-3.5s/call opening a fresh connection each time vs ~350-400ms/call
reusing one), so the client code never calls the bare module function.
"""
import sys
sys.path.append("..")

import requests

import neural_controller as nc
from candor_fd_controller import TurnDecision


class FakeResponse:
    def __init__(self, status_code=200, json_body=None):
        self.status_code = status_code
        self._json = json_body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


def run(name, fn):
    try:
        fn()
        print(f"[{name}] PASS")
        return True
    except AssertionError as e:
        print(f"[{name}] FAIL: {e}")
        return False


def patch_session_post(fake_post):
    """Patches requests.Session.post at the class level (so it applies
    regardless of which Session instance RemoteNeuralController created)
    and returns the original for restoration."""
    orig = requests.Session.post
    requests.Session.post = fake_post
    return orig


def test_successful_call_and_request_shape():
    calls = []

    def fake_post(self, url, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return FakeResponse(200, {"decision": "STOP"})

    orig = patch_session_post(fake_post)
    try:
        controller = nc.RemoteNeuralController("https://example.modal.run/predict", "secret-token", threshold=0.7, timeout_s=2.5)
        calls.clear()  # __init__ fires its own warm-up call -- not what this test is checking
        classifier = controller.new_classifier(time_into_other_turn=3.0)
        for w in ["wait", "stop"]:
            classifier.add_word(w)
        decision = classifier.decision

        assert decision == TurnDecision.STOP, f"expected STOP, got {decision}"
        assert len(calls) == 1, f"expected exactly 1 HTTP call, got {len(calls)}"
        url, body, headers, timeout = calls[0]
        assert url == "https://example.modal.run/predict"
        assert body == {"words": ["wait", "stop"], "time_into_other_turn": 3.0, "threshold": 0.7}
        assert headers == {"Authorization": "Bearer secret-token"}
        assert timeout == 2.5

        # Same word list, still not dirty -- must NOT call again (cache hit).
        _ = classifier.decision
        assert len(calls) == 1, "decision should be cached until add_word()/reset()"
    finally:
        requests.Session.post = orig


def test_max_prefix_words_shortcut_skips_network():
    def fake_post(self, *a, **kw):
        raise AssertionError("should not have called the network for a >5-word segment")

    orig = patch_session_post(fake_post)
    try:
        controller = nc.RemoteNeuralController("https://example.modal.run/predict", "secret-token")
        classifier = controller.new_classifier(time_into_other_turn=1.0)
        for w in "no actually I think the price is too high".split():
            classifier.add_word(w)
        decision = classifier.decision
        assert decision == TurnDecision.STOP
    finally:
        requests.Session.post = orig


def test_reset_clears_cache_and_words():
    def fake_post(self, url, json, headers, timeout):
        text = " ".join(json["words"])
        return FakeResponse(200, {"decision": "BACKCHANNEL" if text == "yeah" else "STOP"})

    orig = patch_session_post(fake_post)
    try:
        controller = nc.RemoteNeuralController("https://example.modal.run/predict", "secret-token")
        classifier = controller.new_classifier(time_into_other_turn=2.0)
        classifier.add_word("yeah")
        assert classifier.decision == TurnDecision.BACKCHANNEL

        classifier.reset()
        assert classifier.words == []
        for w in ["yeah", "no", "wait"]:
            classifier.add_word(w)
        assert classifier.finalize() == TurnDecision.STOP
    finally:
        requests.Session.post = orig


def test_network_failure_falls_back_to_rule_based():
    def fake_post(self, *a, **kw):
        raise requests.exceptions.Timeout("simulated network failure")

    orig = patch_session_post(fake_post)
    try:
        controller = nc.RemoteNeuralController("https://example.modal.run/predict", "secret-token", timeout_s=0.01)
        classifier = controller.new_classifier(time_into_other_turn=1.0)
        for w in ["wait", "no", "stop"]:
            classifier.add_word(w)
        # Must not raise -- falls back to classify_complete_utterance("wait no stop").
        decision = classifier.decision
        assert decision == TurnDecision.STOP  # "wait"/"no" are PROHIBITED_STARTS in the rule-based fallback
    finally:
        requests.Session.post = orig


def main():
    results = [
        run("successful call + request shape + caching", test_successful_call_and_request_shape),
        run("MAX_PREFIX_WORDS shortcut skips the network", test_max_prefix_words_shortcut_skips_network),
        run("reset() clears cache and words", test_reset_clears_cache_and_words),
        run("network failure falls back to rule-based heuristic", test_network_failure_falls_back_to_rule_based),
    ]
    if all(results):
        print("\nAll cases passed.")
    else:
        print("\nSome cases FAILED -- see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
