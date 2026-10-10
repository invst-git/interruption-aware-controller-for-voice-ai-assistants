# CANDOR-FD Voice Assistant — Linux Setup (End-to-End)

A real-time, interruptible voice assistant: continuous mic listening with
echo-cancelled playback (so it can tell your voice apart from its own while
it's talking), Whisper for transcription, Claude for responses, Piper for
speech, and a trained neural model that decides whether you're backchanneling
("mhm", "yeah") or actually trying to interrupt.

This guide assumes **nothing is installed yet** on the Linux machine. Follow
it top to bottom.

## How it's split

```
┌─────────────────────────────┐         ┌───────────────────────────┐
│  Your Linux machine          │  HTTPS  │  Modal (cloud GPU)         │
│  - microphone / speaker      │ ──────▶ │  - DeBERTa turn-controller │
│  - VAD, echo cancellation    │ ◀────── │  - runs modal_controller.py│
│  - Whisper (ASR)             │         └───────────────────────────┘
│  - Claude API call           │
│  - Piper (TTS)               │         ┌───────────────────────────┐
│  - main.py orchestration     │ ──────▶ │  Anthropic API (Claude)    │
└─────────────────────────────┘         └───────────────────────────┘
```

The neural "should I stop talking?" model runs on a Modal GPU over HTTPS, not
on your Linux box — so this machine needs **no GPU, no CUDA, no torch**. It
only needs a microphone, a speaker, and internet access.

---

## What you need before starting

- A Linux machine with a working microphone and speaker, and sudo access.
- An **Anthropic API key** — sign up / create one at
  https://console.anthropic.com/settings/keys (used for the LLM that
  generates the assistant's replies).
- A **Modal account** — sign up at https://modal.com (used to run the neural
  turn-controller on a GPU; has a free tier). If you already have a working
  deployment from a previous machine, skip Part 2 below and just reuse its
  URL and key in Part 3.

---

## Part 1 — Clone the repo

```bash
git clone https://github.com/invst-git/interruption-aware-controller-for-voice-ai-assistants.git voice-ai
cd voice-ai
```

Everything below assumes you're inside this `voice-ai/` directory.

---

## Part 2 — One-time: deploy the neural controller to Modal

**Skip this entire part if you already have a working Modal deployment**
from a previous machine (you'd already have a `MODAL_CONTROLLER_URL` and
`MODAL_CONTROLLER_API_KEY`) — jump to Part 3 and reuse those values.

This is a one-time cloud deployment, not something that runs continuously on
your hardware — it can be done from this Linux machine, or from any machine
with Python. It needs the model checkpoint file
(`checkpoints/best_model_clean.pt`, ~700MB), which is **not** in this git
repo (too large for GitHub) — you must have it from wherever the model was
trained/saved, and place it at `checkpoints/best_model_clean.pt` inside the
repo before step 4 below.

1. **Install Python 3.11+ and pip**, if not already present:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-pip python3-venv
   ```

2. **Set up a virtual environment and install the deploy-side tools**
   (this does NOT install torch locally — torch only runs inside the Modal
   container, not on this machine):
   ```bash
   python3 -m venv .deploy-venv
   source .deploy-venv/bin/activate
   pip install modal fastapi pydantic
   ```

3. **Authenticate the Modal CLI** (opens a browser to log in / link this
   machine to your Modal account):
   ```bash
   modal setup
   ```

4. **Place the checkpoint** at `checkpoints/best_model_clean.pt` if you
   haven't already (see the note above this list).

5. **Create the API auth secret** that the deployed service checks incoming
   requests against:
   ```bash
   python setup_modal_secret.py
   ```
   This prints a line like `MODAL_CONTROLLER_API_KEY=<random-value>` — **copy
   it**, you'll need it in Part 3. (Re-running this later with `--force`
   rotates the key and invalidates the old one everywhere.)

6. **Deploy the service**:
   ```bash
   modal deploy modal_controller.py
   ```
   This builds a container image (installs torch/transformers inside it,
   pre-downloads the DeBERTa-v3-base tokenizer/weights, bakes in your
   checkpoint) and prints a URL at the end that looks like:
   ```
   Created web endpoint for ControllerService.predict =>
   https://<your-workspace>--candor-fd-neural-controller-controllerservi-xxxxxx.modal.run
   ```
   **Copy that full URL** — you'll need it in Part 3 as `MODAL_CONTROLLER_URL`.

   First deploy typically takes 1–3 minutes (installing torch/CUDA packages
   inside the build). Re-deploying after a code change is much faster
   (Modal caches unchanged layers).

7. You can leave `.deploy-venv` behind now — it's not needed to run the app
   itself, only to deploy/redeploy the Modal service.

**Cost note:** the deployed service scales to zero when idle (no charge), and
bills per-second of GPU time while handling requests. A cold start (container
scaling up from zero) took 6–23 seconds in testing; a warm call takes
roughly 350–650ms. `modal_controller.py` has a commented-out `min_containers=1`
option if you'd rather pay to keep it always warm and eliminate cold starts
— leave it alone unless you want that trade-off.

---

## Part 3 — Get the files git doesn't track

Three files are gitignored (secrets/binaries don't belong in GitHub) and
need to be placed in the repo root yourself.

### 3.1 Create `.env`

Create a file named `.env` in the repo root with:

```ini
ANTHROPIC_API_KEY=<your Anthropic API key from console.anthropic.com>
MODAL_CONTROLLER_URL=<the URL printed at the end of Part 2, step 6>
MODAL_CONTROLLER_API_KEY=<the key printed by Part 2, step 5>
```

(`NEURAL_CONTROLLER_MODE` defaults to `"remote"` in `config.py` — you don't
need to set it unless you want to switch to running the model locally
instead, which needs a local GPU and `requirements-local-neural.txt`; not
covered by this guide.)

### 3.2 Get the Piper voice model

Two files: `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json`.

If you have them on another machine already, `scp` is simplest:
```bash
# run FROM the machine that already has the files, pointing AT this Linux box
scp en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json user@linux-host:~/voice-ai/
```

Otherwise, download them straight from the
[Piper voices collection on Hugging Face](https://huggingface.co/rhasspy/piper-voices)
(verified working as of writing this guide):
```bash
curl -L -o en_US-lessac-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
curl -L -o en_US-lessac-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

### 3.3 Checkpoints folder — not needed here

`checkpoints/best_model_clean.pt` is only read if `NEURAL_CONTROLLER_MODE` is
set to `"local"`. In the default `"remote"` mode this machine never touches
it — don't bother copying it here (it was only needed in Part 2, and only if
you were deploying from this machine).

---

## Part 4 — Install Docker

Skip if `docker --version` already works.

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
Log all the way out and back in (or run `newgrp docker` in your current
shell) for the group membership to take effect. Verify:
```bash
docker run hello-world
```

---

## Part 5 — Check your audio devices

```bash
sudo apt install -y alsa-utils   # if arecord/aplay aren't already present
ls -l /dev/snd
arecord -l   # lists capture (microphone) devices
aplay -l     # lists playback (speaker) devices
```
If `/dev/snd` doesn't exist, or these list nothing, fix that at the OS level
before continuing — Docker can only pass through a device that already works
on the host.

Also add yourself to the host's audio group now (needed in Part 7 if
passthrough has permission issues):
```bash
sudo usermod -aG audio $USER
```
(log out/in again if you just ran this)

---

## Part 6 — Build the image

```bash
docker build -t candor-fd .
```
No GPU/CUDA download here — this installs PortAudio, ffmpeg, and the Python
dependencies for the local side only. Should take a couple of minutes.

---

## Part 7 — Run it

```bash
docker run --rm -it \
  --device /dev/snd \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/en_US-lessac-medium.onnx:/app/en_US-lessac-medium.onnx \
  -v $(pwd)/en_US-lessac-medium.onnx.json:/app/en_US-lessac-medium.onnx.json \
  -v whisper-cache:/root/.cache/huggingface \
  candor-fd
```

The `whisper-cache` volume keeps the ~3GB Whisper "large" model (downloaded
automatically on first run, see below) across container restarts, so you
don't re-download it every single run.

### What you should see on startup

```
Loading Faster-Whisper model...
Connecting to remote (Modal) neural controller...
[RemoteNeuralController] warmed up in N.Ns
CANDOR-FD -- live app, echo-cancelled duplex audio
Controller: neural (remote)
Ctrl+C to quit.

[Listening...]
```

Then just talk — the first turn plays the assistant's reply, and you can
interrupt it mid-sentence to test the barge-in behavior.

### First-run expectations

- **Faster-Whisper's "large" model downloads automatically** the first time
  (~3GB from Hugging Face) — needs internet, takes a few minutes. With the
  `whisper-cache` volume above, this only happens once.
- **The first neural-controller check pays a Modal cold start** if the
  container has been idle more than 2 minutes (6–23s observed). The
  `warmed up in N.Ns` line at startup is this cost being paid proactively, at
  boot, instead of during your first live interruption.

---

## Troubleshooting

**`PortAudioError: No Default Input/Output Device`**
PortAudio can't see a usable device inside the container.
- Make sure you passed `--device /dev/snd` and that you're in the `audio`
  group on the host (Part 5).
- If your desktop uses PipeWire/PulseAudio (most modern distros do) and raw
  ALSA passthrough still doesn't expose the right device, share the Pulse
  socket instead:
  ```bash
  -v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
  -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native
  ```
  and add `libpulse0` to the `apt-get install` line in the `Dockerfile`,
  then rebuild (`docker build -t candor-fd .`).

**No sound / mic not detected, but no error printed**
Check what PortAudio actually sees from inside the container and compare
against the host's `arecord -l` / `aplay -l`:
```bash
docker run --rm -it --device /dev/snd candor-fd \
  python -c "import sounddevice as sd; print(sd.query_devices())"
```

**`RemoteNeuralController needs both an endpoint URL and an API key`**
`.env` didn't mount correctly, or a value in it is empty. Check (don't paste
this output anywhere shared — it contains your API keys):
```bash
docker run --rm -v $(pwd)/.env:/app/.env candor-fd cat /app/.env
```

**Connection refused / timeout calling Modal**
No outbound internet from the container, or a firewall is blocking it:
```bash
docker run --rm candor-fd python -c "import requests; print(requests.get('https://modal.com').status_code)"
```

**Modal deploy fails with a missing-package error**
If you changed `modal_controller.py`, double check any new import is also
added to the `.pip_install(...)` list inside it — the deploy machine's own
Python environment is separate from the one that gets built inside the
Modal container.

**Audio works but barge-in feels slow / assistant talks over you**
That's largely bounded by Modal round-trip latency (~350–650ms warm, longer
on a cold start) plus Faster-Whisper's own transcription time — this is a
known, accepted trade-off of this architecture, not a bug. See
`neural_controller.py`'s and `main.py`'s module docstrings for the full
reasoning if you want to tune `PARTIAL_CHECK_INTERVAL_S`, Modal's
`scaledown_window`, or the GPU tier in `modal_controller.py`.

---

## Alternative: running without Docker

If audio passthrough turns out to be more trouble than it's worth, you can
run `main.py` directly in a virtual environment on the host instead — this
sidesteps every container audio question since it talks to the host's audio
system directly:

```bash
sudo apt install -y libportaudio2 libsndfile1 ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
(Needs the same `.env`, `en_US-lessac-medium.onnx`, and
`en_US-lessac-medium.onnx.json` files from Part 3 in the repo root.)

---

## Configuration reference

All settings live in `config.py`; the ones read from `.env` (via
`python-dotenv`) are:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | — | Claude API calls for response generation |
| `MODAL_CONTROLLER_URL` | yes, if `NEURAL_CONTROLLER_MODE=remote` | — | Your deployed Modal endpoint from Part 2 |
| `MODAL_CONTROLLER_API_KEY` | yes, if `NEURAL_CONTROLLER_MODE=remote` | — | Auth token checked by `modal_controller.py` |
| `NEURAL_CONTROLLER_MODE` | no | `remote` | `remote` (Modal GPU) or `local` (needs a local GPU + `requirements-local-neural.txt`) |

Other knobs worth knowing about, directly in `config.py`:
- `USE_NEURAL_CONTROLLER` — set `False` to use the cheap rule-based
  backchannel/interruption classifier instead of the neural one.
- `DEBUG_TIMING` — set `False` once you're done diagnosing latency; it
  currently prints timestamped `[t=...]` lines for every VAD event and
  classifier decision.
- `NEURAL_REMOTE_TIMEOUT_S` — how long to wait for a Modal response before
  falling back to the rule-based heuristic for that one check.
