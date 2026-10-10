# CANDOR-FD voice pipeline -- LOCAL image.
#
# Inference now runs remotely on Modal (see modal_controller.py) with
# config.NEURAL_CONTROLLER_MODE="remote" (the default) -- so this image
# does NOT need a GPU, CUDA, torch, or transformers at all. It just
# needs mic/speaker access and a network path to Modal's endpoint.
#
# ASSUMPTIONS (read before building):
#   - This app opens a real microphone + speaker via sounddevice/
#     PortAudio. Docker does NOT virtualize audio hardware -- you must
#     pass the host's ALSA device through explicitly at `docker run`
#     time (see the run command below). This is a hard requirement,
#     not an optional flag, and is most reliable on a native Linux
#     Docker host (not Docker Desktop on Windows/Mac).
#   - .env (ANTHROPIC_API_KEY, MODAL_CONTROLLER_URL,
#     MODAL_CONTROLLER_API_KEY), en_US-lessac-medium.onnx(.json), and
#     checkpoints/ are all gitignored on purpose -- mount them at
#     runtime instead of baking them into the image (see the run
#     command below). checkpoints/ is only actually read if you switch
#     to NEURAL_CONTROLLER_MODE="local".
FROM python:3.11-slim

WORKDIR /app

# System deps: PortAudio for sounddevice, libsndfile for audio I/O,
# ffmpeg for faster-whisper's audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

CMD ["python", "main.py"]

# --- Build & run ---
#
#   docker build -t candor-fd .
#   docker run --rm -it \
#     --device /dev/snd \
#     -v $(pwd)/.env:/app/.env \
#     -v $(pwd)/en_US-lessac-medium.onnx:/app/en_US-lessac-medium.onnx \
#     -v $(pwd)/en_US-lessac-medium.onnx.json:/app/en_US-lessac-medium.onnx.json \
#     candor-fd
#
# Add `-v $(pwd)/checkpoints:/app/checkpoints` only if you switch
# NEURAL_CONTROLLER_MODE to "local" (and install
# requirements-local-neural.txt into a torch/transformers/CUDA image
# instead of this one -- see that file's header).
