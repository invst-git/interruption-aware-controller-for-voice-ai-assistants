"""config.py -- central settings for the CANDOR-FD voice pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

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
PIPER_VOICE_MODEL = "en_US-lessac-medium.onnx"  # wherever download_voices put it

# --- Assistant persona ---
SYSTEM_PROMPT = (
    "You are a helpful, concise spoken voice assistant. Keep responses "
    "conversational and reasonably short, since you are being spoken aloud, "
    "not read as text. Avoid long lists or complex formatting."
)

PARTIAL_CHECK_INTERVAL_S = 0.3   # how often to re-check in-progress speech during playback
MIN_PARTIAL_AUDIO_S = 0.15       # skip clips shorter than this -- too short to hold a full word
MAX_INTERRUPT_WAIT_S = 15        # safety cap: max wait for an interrupting turn to finish