"""test_aec_library.py -- confirms the real API before we touch main.py."""
from aec_audio_processing import AudioProcessor

ap = AudioProcessor(enable_aec=True, enable_ns=True, enable_agc=True, enable_vad=True)
ap.set_stream_format(sample_rate_in=16000, channel_count_in=1, sample_rate_out=16000, channel_count_out=1)
ap.set_reverse_stream_format(16000, 1)
ap.set_stream_delay(50)

print("AudioProcessor loaded.\n")
print("Full method list:")
for name in dir(ap):
    if not name.startswith("_"):
        print(f"  {name}")

print("\nLook for a method with 'reverse' in the name -- that's almost")
print("certainly how reference audio gets fed in. Paste this full list back.")