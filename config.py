"""config.py -- central settings for the CANDOR-FD voice pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- LLM (Claude API) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 300

# --- ASR (Faster-Whisper) ---
ASR_MODEL_SIZE = "large"
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"

# --- Audio ---
SAMPLE_RATE = 16000
VAD_MIN_SILENCE_MS = 500   # pause length before a turn counts as finished

# --- TTS (Piper) ---
# --- TTS (Piper) ---
PIPER_VOICE_MODEL = str(_PROJECT_ROOT / "en_US-lessac-medium.onnx")
# --- Assistant persona ---
SYSTEM_PROMPT = (
    "You are a helpful, concise spoken voice assistant. Keep responses "
    "conversational and reasonably short, since you are being spoken aloud, "
    "not read as text. Avoid long lists or complex formatting."
)

PARTIAL_CHECK_INTERVAL_S = 0.3   # how often to re-check in-progress speech during playback
MIN_PARTIAL_AUDIO_S = 0.15       # skip clips shorter than this -- too short to hold a full word
MAX_INTERRUPT_WAIT_S = 15        # safety cap: max wait for an interrupting turn to finish
CONTINUATION_GRACE_S = 0.9  # after a pause, how long to wait for the user to keep talking before assuming they're done
MIN_RESUME_TAIL_S = 0.5     # false stop this close to the end of a reply isn't worth resuming
AEC_STREAM_DELAY_MS = 50  # rough starting hint only -- AEC's own live control part adjusts from here
USE_NEURAL_CONTROLLER = True   # flip to True once the benchmark and a live test both pass
NEURAL_CONTROLLER_CHECKPOINT = "checkpoints/best_model_clean.pt"
NEURAL_CONTROLLER_THRESHOLD = 0.5   # the untested default -- tune deliberately before relying on it
DEBUG_TIMING = True   # timestamped VAD-event / classifier-decision / stop_speaking logging in main.py

# --- Neural controller deployment: local torch process, or Modal's GPU
# service over HTTPS (see modal_controller.py + neural_controller.py's
# RemoteNeuralController). "remote" needs no torch/transformers/CUDA on
# this machine at all -- only `requests` and the two env vars below.
NEURAL_CONTROLLER_MODE = os.environ.get("NEURAL_CONTROLLER_MODE", "remote")  # "local" | "remote"
MODAL_CONTROLLER_URL = os.environ.get("MODAL_CONTROLLER_URL", "")
MODAL_CONTROLLER_API_KEY = os.environ.get("MODAL_CONTROLLER_API_KEY", "")
NEURAL_REMOTE_TIMEOUT_S = 5.0   # per-call cap before falling back to the rule-based heuristic