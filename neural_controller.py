"""
neural_controller.py -- loads the trained DeBERTa-based controller and
wraps it with roughly the same interface as candor_fd_controller.py's
StreamingTurnClassifier (add_word, decision, finalize, words, reset),
so main.py can swap between controllers with minimal change to the
calling code. Two implementations of that interface live here:

  - NeuralController: runs the model in-process with torch (local GPU
    or CPU, via device auto-detection). Needs torch/transformers
    installed (requirements-local-neural.txt) -- ControllerModel and
    NeuralController/NeuralStreamingTurnClassifier are only DEFINED
    below if that import succeeds, so this module stays importable
    without them.
  - RemoteNeuralController: calls modal_controller.py's deployed Modal
    GPU service over HTTPS instead -- for when the machine running
    main.py (mic/speaker/ASR) doesn't itself have a GPU. Only the
    model forward pass moves to Modal; audio hardware has to stay
    local regardless, so this is a split architecture, not a full
    lift-and-shift. Needs only `requests` -- see requirements.txt (the
    lean, default local requirements) vs requirements-local-neural.txt.
    See modal_controller.py's docstring for the service side, and its
    "COST NOTE" for the warm-container trade-off.

CADENCE WARNING, READ BEFORE WIRING THIS IN
--------------------------------------------
A single forward pass measured ~1s on a CPU-only machine (see
benchmark_neural_inference.py) -- meaningfully slower than the
rule-based check it replaces, and slower than PARTIAL_CHECK_INTERVAL_S.
On a real GPU (local or via RemoteNeuralController -> Modal), this
should no longer be the bottleneck; verify with the same benchmark
approach on whatever's actually running inference before assuming it.

THRESHOLD
---------
Defaults to 0.5 here only because that has to be the shipped default of
some value. This is exactly the number flagged repeatedly during
evaluation as needing a deliberate choice, not an inherited one -- pass
a considered value in, don't leave this at 0.5 for real use.
"""
import time
import requests

from candor_fd_controller import TurnDecision, classify_complete_utterance

MAX_TOKEN_LEN = 24
MAX_PREFIX_WORDS = 5  # never trained past this many words -- see the cutoff below

# torch/transformers/numpy are only needed by the LOCAL path
# (NeuralController et al.) -- RemoteNeuralController below needs none
# of them, just `requests`. Importing them eagerly at module level
# would make `from neural_controller import RemoteNeuralController`
# fail on a lean machine with no torch installed (exactly the machine
# this class exists for), so they're deferred and the local classes
# are only defined if the import actually succeeds.
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    _LOCAL_TORCH_AVAILABLE = True
except ImportError:
    _LOCAL_TORCH_AVAILABLE = False


if _LOCAL_TORCH_AVAILABLE:
    class ControllerModel(nn.Module):
        """Must exactly match the architecture in train_controller.py --
        this loads weights trained against that exact structure."""

        def __init__(self, encoder_name, n_numeric, dropout=0.1):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)
            hidden = self.encoder.config.hidden_size
            self.numeric_proj = nn.Sequential(
                nn.Linear(n_numeric, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden + 64, 256), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(256, 1),
            )

        def forward(self, input_ids, attention_mask, features):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            joined = torch.cat([cls, self.numeric_proj(features)], dim=1)
            return self.head(joined).squeeze(-1)


    class NeuralController:
        """Loads the checkpoint once, at startup. Call new_classifier() for
        each new utterance to monitor -- do not construct this class itself
        more than once, loading DeBERTa is the expensive part."""

        def __init__(self, checkpoint_path, threshold=0.5, device=None):
            # Auto-detects CUDA so the same code path is CPU-only-safe here
            # and actually uses the GPU once this runs on hardware that has
            # one -- previously this always loaded onto and ran on the CPU
            # regardless of what was available, which would have made moving
            # to a GPU box a no-op for inference speed.
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[NeuralController] loading onto device: {self.device}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
            self.model = ControllerModel(ckpt["encoder_name"], len(ckpt["numeric_cols"]))
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(self.device)
            self.model.eval()
            self.feat_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
            self.feat_std = np.array(ckpt["feat_std"], dtype=np.float32)
            self.threshold = threshold

        def new_classifier(self, time_into_other_turn):
            """time_into_other_turn: how many seconds the assistant had
            already been speaking when the user's speech was first detected
            starting. Fixed for the whole utterance -- read it once, at the
            moment speech starts, from listener.get_played_seconds()."""
            return NeuralStreamingTurnClassifier(self, time_into_other_turn)


    class NeuralStreamingTurnClassifier:
        def __init__(self, controller, time_into_other_turn):
            self._controller = controller
            self._time_into_other_turn = time_into_other_turn
            self.words = []
            self._cached_decision = None
            self._dirty = True

        def add_word(self, word):
            self.words.append(word)
            self._dirty = True

        def reset(self):
            """Clears any words fed so far (e.g. from mid-utterance partial
            checks) so the next add_word()/decision reflects only what's fed
            after this call -- used by main.py at end-of-utterance to force
            a fresh evaluation on the complete transcript instead of
            whatever prefix was cached from live checks."""
            self.words = []
            self._cached_decision = None
            self._dirty = True

        @property
        def decision(self):
            if len(self.words) > MAX_PREFIX_WORDS:
                # Backbiter caps a backchannel at 3 words, and the training
                # data confirms every k>=4 PREFIX example is a real turn --
                # true whether this is called mid-utterance (a growing
                # prefix) or main.py's end-of-utterance reset()+finalize()
                # (the complete word list of a finished segment, which is
                # also a k>=4 case whenever it's this long). Either way this
                # short-circuits to STOP without inference. An ASR
                # hallucination inflating the end-of-utterance word count is
                # the one way this could fire wrongly; run_turn()'s
                # classify_complete_utterance() re-check applies the same
                # word-count reasoning independently as a second opinion
                # before actually cutting off the reply, so that case still
                # can't let a real backchannel through unchecked.
                return TurnDecision.STOP
            if self._dirty:
                self._cached_decision = self._infer()
                self._dirty = False
            return self._cached_decision

        def finalize(self):
            return self.decision

        def _infer(self):
            if not self.words:
                return TurnDecision.BACKCHANNEL
            text = " ".join(self.words)
            raw_feats = np.array(
                [len(self.words), self._time_into_other_turn], dtype=np.float32,
            )
            norm_feats = (raw_feats - self._controller.feat_mean) / self._controller.feat_std
            device = self._controller.device
            with torch.no_grad():
                enc = self._controller.tokenizer(
                    text, truncation=True, max_length=MAX_TOKEN_LEN,
                    padding="max_length", return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                feats = torch.tensor(norm_feats[None, :], dtype=torch.float32, device=device)
                logits = self._controller.model(input_ids, attention_mask, feats)
                prob = torch.sigmoid(logits).item()
            return TurnDecision.STOP if prob >= self._controller.threshold else TurnDecision.BACKCHANNEL


class RemoteNeuralController:
    """Same public interface as NeuralController (new_classifier()), but
    every real inference call goes to modal_controller.py's deployed
    Modal GPU service over HTTPS instead of running torch here. Nothing
    ML-heavy (torch, transformers, the checkpoint) needs to be present
    on this machine to use this class -- only `requests`."""

    def __init__(self, endpoint_url, api_key, threshold=0.5, timeout_s=5.0):
        if not endpoint_url or not api_key:
            raise ValueError(
                "RemoteNeuralController needs both an endpoint URL and an API "
                "key -- see config.py's MODAL_CONTROLLER_URL/MODAL_CONTROLLER_API_KEY "
                "(set via .env) and modal_controller.py's deploy instructions."
            )
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.threshold = threshold
        self.timeout_s = timeout_s
        # A fresh requests.post() call opens a new TLS connection every
        # time -- measured ~1.4-3.5s/call to the deployed endpoint
        # because of that alone. A reused Session (keep-alive) measured
        # ~350-400ms/call after the first, since the TLS handshake only
        # happens once. One session per controller (i.e. per process),
        # shared by every classifier it creates.
        self._session = requests.Session()
        self._warm_up()

    def _warm_up(self):
        """Pays the TLS handshake AND Modal cold-start cost (measured:
        ~1.1s handshake, ~5.7s cold start if no container is already
        warm) once, at startup, instead of on the user's first live
        interruption. Best-effort -- a startup network hiccup here just
        means the first real call pays the cost instead, not a crash."""
        try:
            t0 = time.monotonic()
            self._session.post(
                self.endpoint_url,
                json={"words": ["warmup"], "time_into_other_turn": 0.0, "threshold": self.threshold},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=max(self.timeout_s, 30.0),
            )
            print(f"[RemoteNeuralController] warmed up in {time.monotonic() - t0:.1f}s")
        except Exception as e:
            print(f"[RemoteNeuralController] warm-up call failed ({e}) -- "
                  f"the first real interruption will pay this cost instead")

    def new_classifier(self, time_into_other_turn):
        return RemoteNeuralStreamingTurnClassifier(self, time_into_other_turn)


class RemoteNeuralStreamingTurnClassifier:
    """Mirrors NeuralStreamingTurnClassifier's logic exactly (same
    MAX_PREFIX_WORDS shortcut, same caching), except _infer() is an
    HTTPS call to Modal instead of a local torch forward pass. The
    shortcut staying local means short/obvious segments (>5 words)
    never pay a network round trip at all -- only genuinely ambiguous
    short prefixes do."""

    def __init__(self, controller, time_into_other_turn):
        self._controller = controller
        self._time_into_other_turn = time_into_other_turn
        self.words = []
        self._cached_decision = None
        self._dirty = True

    def add_word(self, word):
        self.words.append(word)
        self._dirty = True

    def reset(self):
        self.words = []
        self._cached_decision = None
        self._dirty = True

    @property
    def decision(self):
        if len(self.words) > MAX_PREFIX_WORDS:
            return TurnDecision.STOP
        if self._dirty:
            self._cached_decision = self._infer()
            self._dirty = False
        return self._cached_decision

    def finalize(self):
        return self.decision

    def _infer(self):
        if not self.words:
            return TurnDecision.BACKCHANNEL
        controller = self._controller
        try:
            resp = controller._session.post(
                controller.endpoint_url,
                json={
                    "words": self.words,
                    "time_into_other_turn": self._time_into_other_turn,
                    "threshold": controller.threshold,
                },
                headers={"Authorization": f"Bearer {controller.api_key}"},
                timeout=controller.timeout_s,
            )
            resp.raise_for_status()
            return TurnDecision.STOP if resp.json()["decision"] == "STOP" else TurnDecision.BACKCHANNEL
        except Exception as e:
            # A network blip or a cold-starting/unreachable Modal
            # container must not hang or silently guess -- fall back to
            # the same rule-based heuristic run_turn() already uses as
            # its own second opinion on a finished utterance. This
            # trades model accuracy for availability on this one check;
            # it does not retry, so a slow-but-eventually-successful
            # call isn't waited out here.
            print(f"[RemoteNeuralController] inference call failed ({e}) -- "
                  f"falling back to the rule-based heuristic for this check")
            return classify_complete_utterance(" ".join(self.words))
