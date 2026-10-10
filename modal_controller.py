"""modal_controller.py -- deploys the neural turn-controller's DeBERTa
model as a Modal GPU service.

Why this exists: the live app's mic/speaker have to stay on your local
Linux machine (Modal can't reach local audio hardware), so only the
model inference call itself moves to the cloud. main.py keeps running
locally; neural_controller.py's RemoteNeuralController calls this
service over HTTPS instead of running torch in-process. See that
file's module docstring for the client side of this split.

Deploy with:
    modal deploy modal_controller.py

That prints this app's web endpoint URL -- put it in your .env as
MODAL_CONTROLLER_URL, alongside MODAL_CONTROLLER_API_KEY (see
setup_modal_secret.py, which creates both the Modal-side secret and
prints the value to put in .env).

COST NOTE: gpu="T4" is Modal's cheapest GPU tier, billed per second
while a container is running. scaledown_window=120 means a container
that served a request stays warm for 2 minutes of idle before scaling
back to zero -- so a burst of interruptions in one conversation mostly
avoids repeated cold starts, but the FIRST call after a long idle gap
pays a cold start (loading torch + DeBERTa onto a fresh GPU container,
likely several seconds) before the model responds. Setting
min_containers=1 below would keep a container running 24/7 (eliminates
cold starts entirely, but bills continuously even when idle) -- left
off by default; uncomment only if you want to pay for that.
"""
import modal
from fastapi import Request, HTTPException
from pydantic import BaseModel

app = modal.App("candor-fd-neural-controller")

MAX_TOKEN_LEN = 24

# Encoder name matches what benchmark_neural_inference.py/sanity_check_neural.py's
# transformers LOAD REPORT actually printed when loading this checkpoint
# (microsoft/deberta-v3-base). Baked into the image at build time so a
# cold-starting container doesn't hit the HuggingFace Hub on the
# request path -- that download would otherwise dominate the first
# call's latency (and fail outright with no network egress).
ENCODER_NAME = "microsoft/deberta-v3-base"

image = (
    modal.Image.debian_slim(python_version="3.11")
    # sentencepiece (+ its own protobuf dependency) is required to load
    # DeBERTa-v3's tokenizer (its spm.model vocab file) -- neither is a
    # hard dependency of transformers itself, so both must be listed
    # explicitly or AutoTokenizer.from_pretrained fails outright on a
    # fresh install.
    # requests: not used server-side, but neural_controller.py imports
    # it at module level (for RemoteNeuralController's client code) and
    # modal_controller.py imports ControllerModel from that same
    # module -- omitting it here crash-loops @modal.enter() on every
    # container start with ModuleNotFoundError.
    .pip_install("torch", "transformers", "sentencepiece", "protobuf", "numpy", "fastapi[standard]", "requests")
    .run_commands(
        f'python -c "from transformers import AutoModel, AutoTokenizer; '
        f"AutoModel.from_pretrained('{ENCODER_NAME}'); "
        f"AutoTokenizer.from_pretrained('{ENCODER_NAME}')\""
    )
    # neural_controller.py's ControllerModel class defines the exact
    # architecture the checkpoint was trained against -- imported here
    # rather than redefined, so the two can't drift apart.
    .add_local_file("neural_controller.py", "/root/neural_controller.py")
    .add_local_file("candor_fd_controller.py", "/root/candor_fd_controller.py")
    .add_local_file("checkpoints/best_model_clean.pt", "/root/checkpoint.pt")
)


class PredictRequest(BaseModel):
    words: list[str]
    time_into_other_turn: float
    threshold: float = 0.5


class PredictResponse(BaseModel):
    decision: str  # "STOP" or "BACKCHANNEL"


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=120,
    secrets=[modal.Secret.from_name("candor-fd-api-key")],
    # min_containers=1,  # uncomment to keep a container always warm (extra, continuous cost)
)
class ControllerService:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from neural_controller import ControllerModel

        ckpt = torch.load("/root/checkpoint.pt", map_location="cuda", weights_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
        self.model = ControllerModel(ckpt["encoder_name"], len(ckpt["numeric_cols"]))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to("cuda")
        self.model.eval()
        self.feat_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
        self.feat_std = np.array(ckpt["feat_std"], dtype=np.float32)
        self._torch = torch
        self._np = np

    def _check_auth(self, request: Request):
        import os
        expected = os.environ["API_KEY"]
        got = request.headers.get("authorization", "")
        if got != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @modal.fastapi_endpoint(method="POST")
    def predict(self, item: PredictRequest, request: Request) -> PredictResponse:
        self._check_auth(request)

        torch, np = self._torch, self._np
        text = " ".join(item.words)
        raw_feats = np.array([len(item.words), item.time_into_other_turn], dtype=np.float32)
        norm_feats = (raw_feats - self.feat_mean) / self.feat_std
        with torch.no_grad():
            enc = self.tokenizer(
                text, truncation=True, max_length=MAX_TOKEN_LEN,
                padding="max_length", return_tensors="pt",
            )
            input_ids = enc["input_ids"].to("cuda")
            attention_mask = enc["attention_mask"].to("cuda")
            feats = torch.tensor(norm_feats[None, :], dtype=torch.float32, device="cuda")
            logits = self.model(input_ids, attention_mask, feats)
            prob = torch.sigmoid(logits).item()
        decision = "STOP" if prob >= item.threshold else "BACKCHANNEL"
        return PredictResponse(decision=decision)
